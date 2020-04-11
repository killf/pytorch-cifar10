from torchvision.datasets import CIFAR10

train_dataset = CIFAR10("data", train=True, download=True)
test_dataset = CIFAR10("data", train=False, download=True)


def show_labels_count(dataset):
    cls = {i: 0 for i in range(10)}
    for image, label in dataset:
        cls[label] += 1
    print(cls)


show_labels_count(train_dataset)
show_labels_count(test_dataset)
