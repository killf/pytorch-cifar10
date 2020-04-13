import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from old.resnet import ResNet18 as resnet18

import random
import numpy as np

GLOBAL_SEED = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)


set_seed(GLOBAL_SEED)
cudnn.benchmark = False
cudnn.deterministic = True

LR = 0.1
EPOCHS = 300
DATA_DIR = "../data"


def train():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = CIFAR10(DATA_DIR, train=True, transform=train_transforms, download=True)
    test_dataset = CIFAR10(DATA_DIR, train=False, transform=test_transforms, download=True)

    train_data = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
    test_data = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    for epoch in range(EPOCHS):
        net.train()
        correct, total = 0, 0
        for step, (x, y_true) in enumerate(train_data):
            x, y_true = x.to(device), y_true.to(device)

            optimizer.zero_grad()

            y_pred = net(x)
            loss = criterion(y_pred, y_true)
            loss.backward()

            optimizer.step()

            y_pred = torch.argmax(y_pred, dim=-1)
            correct += (y_pred == y_true).sum().item()
            total += y_true.size(0)
            if step % 100 == 0:
                print(f"Epoch:{epoch} Step:{step}, Loss:{loss.item():05f}, Acc:{correct / total:05f}")

        scheduler.step()

        net.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for x, y_true in test_data:
                x, y_true = x.to(device), y_true.to(device)

                y_pred = net(x)

                y_pred = torch.argmax(y_pred, dim=-1)
                correct += (y_pred == y_true).float().sum().item()
                total += y_true.size(0)

            print(f"[VAL]Epoch:{epoch}, Acc:{correct / total:05f}")
            print()


if __name__ == '__main__':
    train()
