import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50
from torchmirror import model, dataset

LR = 0.001
EPOCHS = 30
BATCH_SIZE = 32
DATA_DIR = "data"
MODEL_FILE = "models/resnet50.pkl"


def train():
    train_transforms = transforms.Compose([
        transforms.RandomGrayscale(0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = CIFAR10(DATA_DIR, train=True, transform=train_transforms, download=True)
    test_dataset = CIFAR10(DATA_DIR, train=False, transform=test_transforms, download=True)

    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=7)
    test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    net = resnet50(num_classes=10)
    if torch.cuda.is_available():
        net = net.cuda()

    if os.path.exists(MODEL_FILE):
        net.load_state_dict(torch.load(MODEL_FILE))

    optimizer = torch.optim.Adam(net.parameters(), LR)
    for epoch in range(EPOCHS):
        for step, (x, y_true) in enumerate(train_data, 1):
            if torch.cuda.is_available():
                x = x.cuda()
                y_true = y_true.cuda()

            y_pred = net(x)
            loss = F.cross_entropy(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                y_pred = torch.argmax(y_pred, dim=-1)
                acc = (y_pred == y_true).float().mean()
                print(f"Epoch:{epoch} Step:{step}, Loss:{loss:05f}, Acc:{acc:05f}")

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

        torch.save(net.state_dict(), MODEL_FILE)


if __name__ == '__main__':
    train()
