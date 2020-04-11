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
import torch.backends.cudnn as cudnn

LR = 0.1
EPOCHS = 5
BATCH_SIZE = 128
NUM_WORKERS = 2
DATA_DIR = "../data"
MODEL_FILE = "models/resnet18_ex2.pkl"


def main():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet18(num_classes=10).to(device)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    for epoch in range(EPOCHS):
        net.train()
        correct, total = 0., 0.
        for step, (x, y_true) in enumerate(trainloader, 1):
            x = x.to(device)
            y_true = y_true.to(device)

            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            _, predicted = y_pred.max(1)
            total += y_true.size(0)
            correct += predicted.eq(y_true).sum().item()
            if step % 100 == 0:
                print(f"Epoch:{epoch} Step:{step}, Loss:{loss.item():05f}, Acc:{correct / total:05f}")

        del correct
        del total
        del x
        del y_true
        del loss
        del predicted

        net.eval()
        with torch.no_grad():
            correct, total = 0., 0.
            for x, y_true in testloader:
                x = x.to(device)
                y_true = y_true.to(device)

                y_pred = net(x)

                _, predicted = y_pred.max(1)
                total += y_true.size(0)
                correct += predicted.eq(y_true).sum().item()

        print(f"[VAL]Epoch:{epoch}, Acc:{correct / total:05f}")
        print()

        del correct
        del total
        del x
        del y_true
        del predicted


if __name__ == '__main__':
    main()
