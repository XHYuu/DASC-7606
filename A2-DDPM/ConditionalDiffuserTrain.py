import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from ConditionalDiffuser import ConditionalDiffuser
import math
import argparse
import datetime
from pathlib import Path
from tqdm import tqdm


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def create_mnist_dataloaders(batch_size, image_size=28, num_workers=4):
    preprocess = transforms.Compose([transforms.Resize(image_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])  # [0,1] to [-1,1]
    train_dataset = MNIST(root="./mnist_data",
                          train=True,
                          download=False,
                          transform=preprocess
                          )
    test_dataset = MNIST(root="./mnist_data",
                         train=False,
                         download=False,
                         transform=preprocess
                         )

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=40)
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet', default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of DDPM', default=1000)
    parser.add_argument('--numbers', type=int, help='classes of number', default=10)
    parser.add_argument('--model_ema_steps', type=int, help='ema model evaluation interval', default=10)
    parser.add_argument('--model_ema_decay', type=float, help='ema model decay', default=0.995)
    parser.add_argument('--cpu', action='store_true', help='cpu training')

    args = parser.parse_args()

    return args


def main(args):
    device = "cpu" if args.cpu else "cuda"
    train_dataloader, _ = create_mnist_dataloaders(batch_size=args.batch_size, image_size=28)
    model = ConditionalDiffuser(
        timesteps=args.timesteps,
        number_class=args.numbers,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4]
    ).to(device)

    # torchvision ema setting
    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(optimizer, args.lr, total_steps=args.epochs * len(train_dataloader), pct_start=0.25,
                           anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')

    global_steps = 0
    # create project name with current time
    exp_name = datetime.datetime.now().strftime("%m%d-%H%M%S")
    exp_path = Path("logs") / exp_name
    exp_path.mkdir(parents=True)
    (exp_path / "ckpt").mkdir(exist_ok=True)
    (exp_path / "img").mkdir(exist_ok=True)

    ckpt_list = []
    loss_record = []
    for i in range(args.epochs):
        model.train()
        # leave_option = False if i < args.epochs - 1 else True
        training_progress = tqdm(train_dataloader, desc='Training Progress', leave=True)
        loss: Tensor = torch.tensor(0.0, device=device)
        for image, label in training_progress:
            noise = torch.randn_like(image).to(device)
            image = image.to(device)
            label = label.to(device)
            pred = model(image, noise, label)
            loss = loss_fn(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
            global_steps += 1
            training_progress.set_description(f"epoch-{i} loss: {loss.detach().cpu().item():.4f}")
            loss_record.append(loss.detach().cpu().item())
        ckpt = {
            "model": model.state_dict(),
            "model_ema": model_ema.state_dict()
        }
        ckpt_path = exp_path / "ckpt" / f"{i}.pt"
        torch.save(ckpt, ckpt_path)
        ckpt_list.insert(0, ckpt_path)
        if len(ckpt_list) > 5:
            remove_ckpt = ckpt_list.pop()
            remove_ckpt.unlink()

        model_ema.eval()
        samples = model_ema.module.sampling(args.n_samples, 10, device=device)
        save_image(samples, exp_path / "img" / f"{i}.png", nrow=10)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    iterations = list(map(lambda x: x, range(1, len(loss_record) + 1)))
    ax1.plot(iterations, loss_record, marker='.', color='blue', label='Loss')
    ax1.set_xlabel('Iteration', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.tick_params(axis='y')
    plt.title('Loss Curves', fontsize=16)
    fig.legend(bbox_to_anchor=(0.92, 0.75), bbox_transform=ax1.transAxes)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(exp_path / "img" / "Conditional_loss_image.png")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
