import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
from torchmirror import model, dataset

LR = 0.1
EPOCHS = 500
BATCH_SIZE = 32
NUM_WORKERS = 7
DATA_DIR = "../data"
MODEL_FILE = "../models/resnet18_ex.pkl"


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

    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    net = resnet18(num_classes=10)
    if torch.cuda.is_available():
        net = net.cuda()

    if os.path.exists(MODEL_FILE):
        net.load_state_dict(torch.load(MODEL_FILE))

    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)

    for epoch in range(EPOCHS):
        net.train()
        for step, (x, y_true) in enumerate(train_data, 1):
            if torch.cuda.is_available():
                x = x.cuda()
                y_true = y_true.cuda()

            optimizer.zero_grad()
            y_pred = net(x)
            loss = F.cross_entropy(y_pred, y_true)

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                y_pred = torch.argmax(y_pred, dim=-1)
                acc = (y_pred == y_true).float().mean()
                print(f"Epoch:{epoch} Step:{step}, Loss:{loss:05f}, Acc:{acc:05f}")

        net.eval()
        with torch.no_grad():
            acc, count = 0, 0
            for x, y_true in test_data:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y_true = y_true.cuda()

                y_pred = net(x)
                y_pred = torch.argmax(y_pred, dim=-1)
                acc += (y_pred == y_true).float().sum()
                count += y_true.size(0)

            acc /= count
            print(f"[VAL]Epoch:{epoch}, Acc:{acc:05f}")
            print()

        scheduler.step(epoch)
        torch.save(net.state_dict(), MODEL_FILE)


if __name__ == '__main__':
    train()
