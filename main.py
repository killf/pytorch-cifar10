import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets

from utils import *
from loss import *
import models

LR = 0.1
LR_MILESTONES = [135, 185, 240]
EPOCHS = 240
START_EPOCH = 0
DATA_DIR = "data"
DATASET = "CIFAR10"
BATCH_SIZE = 128
NUM_WORKERS = 16
MODEL_NAME = "resnet18"
MODEL_FILE = f"output/{MODEL_NAME}.pkl"
LOG_FILE = "log.txt"


def main():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = torchvision.datasets.__dict__[DATASET]
    train_dataset = dataset(DATA_DIR, train=True, transform=train_transforms, download=True)
    test_dataset = dataset(DATA_DIR, train=False, transform=test_transforms, download=True)

    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.__dict__[MODEL_NAME](pretrained=False).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    # scheduler = MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=0.1)

    best_acc = 0
    open(LOG_FILE, "w", encoding="utf8")

    for epoch in range(START_EPOCH, START_EPOCH + EPOCHS):
        if epoch == 0:
            optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        elif epoch == 135:
            optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        elif epoch == 185:
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

        train_acc = train_one_epoch(epoch, train_data, net, optimizer, criterion, device)
        test_acc = test_one_epoch(epoch, test_data, net, device)

        with open(LOG_FILE, "a", encoding="utf8") as f:
            f.write(f"{epoch}\t{train_acc}\t{test_acc}\n")

        if test_acc > best_acc:
            folder = os.path.dirname(MODEL_FILE)
            if not os.path.exists(folder):
                os.makedirs(folder)

            torch.save(net.state_dict(), MODEL_FILE)
            best_acc = test_acc

        # scheduler.step()

    print(f"best_acc:{best_acc:05f}")
    if os.path.exists(MODEL_FILE):
        os.rename(MODEL_FILE, f"{MODEL_FILE}.{best_acc:05f}")


def train_one_epoch(epoch, data_loader, net, optimizer, criterion, device):
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

    return correct / total


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

        acc = correct / total
        print(f"[VAL]Epoch:{epoch}, Acc:{acc:05f}")
        print()

    return acc


def print_config():
    print(f"model: {MODEL_NAME}")
    print(f"dataset: {DATASET}")
    print(f"lr: {LR} {LR_MILESTONES}")
    print(f"epochs: {EPOCHS}")
    print(f"workers: {NUM_WORKERS}")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"output: {MODEL_FILE}")
    print(f"log: {LOG_FILE}")
    print()


if __name__ == '__main__':
    print_config()
    main()
