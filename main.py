import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from models.resnet import resnet18
from utils import *

LR = 0.1
EPOCHS = 50
START_EPOCH = 0
DATA_DIR = "data"
BATCH_SIZE = 128
NUM_WORKERS = 16
SEED = 0
set_seed(SEED)


def main():
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

    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                            worker_init_fn=worker_init_fn)
    test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                           worker_init_fn=worker_init_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    for epoch in range(START_EPOCH, START_EPOCH + EPOCHS):
        train_one_epoch(epoch, train_data, net, optimizer, criterion, scheduler, device)
        test_one_epoch(epoch, test_data, net, device)


def train_one_epoch(epoch, data_loader, net, optimizer, criterion, scheduler, device):
    net.train()
    correct, total = 0, 0
    for step, (x, y_true) in enumerate(data_loader):
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


def test_one_epoch(epoch, data_loader, net, device):
    net.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for x, y_true in data_loader:
            x, y_true = x.to(device), y_true.to(device)

            y_pred = net(x)

            y_pred = torch.argmax(y_pred, dim=-1)
            correct += (y_pred == y_true).float().sum().item()
            total += y_true.size(0)

        print(f"[VAL]Epoch:{epoch}, Acc:{correct / total:05f}")
        print()


if __name__ == '__main__':
    main()
