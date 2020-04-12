import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

import models

DATA_DIR = "../data"
DATASET = "CIFAR10"
BATCH_SIZE = 32
NUM_WORKERS = 4
MODEL_NAME = "dpn92"
MODEL_FILE = f"../output/dpn92.pkl.0.943200"
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_de_transforms = transforms.Compose([
    transforms.Normalize((-0.4914 / 0.2023, - 0.4822 / 0.1994, - 0.4465 / 0.2010),
                         (1 / 0.2023, 1 / 0.1994, 1 / 0.2010)),
    transforms.ToPILImage(),
])

test_dataset = CIFAR10(DATA_DIR, train=False, download=True, transform=test_transforms)
test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

device = torch.device(DATA_DIR if torch.cuda.is_available() else "cpu")

net = models.__dict__[MODEL_NAME]().to(device)
net.load_state_dict(torch.load(MODEL_FILE))


def main():
    err_ls = []

    net.eval()
    with torch.no_grad():
        correct, total, index = 0, 0, {i: 0 for i in range(10)}
        for x, y_true in test_data:
            x, y_true = x.to(device), y_true.to(device)

            y_pred = net(x)

            y_pred = torch.argmax(y_pred, dim=-1)
            correct += (y_pred == y_true).float().sum().item()
            total += y_true.size(0)

            err = (y_pred != y_true).cpu().numpy()
            for i in range(y_true.size(0)):
                if err[i]:
                    img = x[i].cpu()
                    img = test_de_transforms(img)

                    y_true_i, y_pred_i = y_true[i].item(), y_pred[i].item()
                    file = f"images/err/{y_true_i}_{CLASSES[y_true_i]}/{index[y_true_i]}_{y_pred_i}.png"
                    folder = os.path.dirname(file)
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    img.save(file)
                    index[y_true_i] += 1

        acc = correct / total
        print(f"Acc:{acc:05f}")
        print()


if __name__ == '__main__':
    main()
