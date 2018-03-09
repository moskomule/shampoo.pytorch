from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cifar_resnet import resnet20
from utils import Trainer
from shampoo import Shampoo


def get_dataloader(batch_size, root="~/.torch/data/cifar10"):
    root = Path(root).expanduser()
    if not root.exists():
        root.mkdir()
    root = str(root)

    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    data_augmentation = [transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip()]

    train_loader = DataLoader(
            datasets.CIFAR10(root, train=True, download=True,
                             transform=transforms.Compose(data_augmentation + to_normalized_tensor)),
            batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
            datasets.CIFAR10(root, train=False, transform=transforms.Compose(to_normalized_tensor)),
            batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def main(batch_size, lr, momentum, epsilon, update_freq):
    train_loader, test_loader = get_dataloader(batch_size)

    model = resnet20()
    optimizer = Shampoo(params=model.parameters(), lr=lr, momentum=momentum,
                        weight_decay=1e-4, epsilon=epsilon, update_freq=update_freq)
    trainer = Trainer(model, optimizer, F.cross_entropy)
    trainer.loop(200, train_loader, test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batchsize", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--epsilon", type=float, default=1e-4)
    p.add_argument("--update_freq", type=int, default=1)
    args = p.parse_args()
    main(args.batchsize, args.lr, args.momentum, args.epsilon, args.update_freq)
