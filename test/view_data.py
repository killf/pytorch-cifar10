import os

from torchvision.datasets import CIFAR10

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train_dataset = CIFAR10("../data", train=True, download=True)
test_dataset = CIFAR10("../data", train=False, download=True)


def show_labels_count(dataset):
    cls = {i: 0 for i in range(10)}
    for image, label in dataset:
        cls[label] += 1
    print(cls)

#
# show_labels_count(train_dataset)
# show_labels_count(test_dataset)


def extract(dataset, name):
    index = {i: 0 for i in range(10)}
    for image, label in dataset:
        file = f"images/{name}/{label}_{CLASSES[label]}/{index[label]}.png"
        folder = os.path.dirname(file)
        if not os.path.exists(folder):
            os.makedirs(folder)

        image.save(file)
        index[label] += 1
    print(index)


extract(train_dataset, "train")
extract(test_dataset, "test")
