import torchvision
from torchvision import transforms


def load_cifar(num_classes, size=(112, 112)):
    # data preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
    ])

    if num_classes == 100:
        # Load training dataset
        train_set = torchvision.datasets.CIFAR100(
            root='./CIFAR10/train_set',
            train=True,
            download=True,
            transform=transform
        )

        # Load test set
        test_set = torchvision.datasets.CIFAR100(
            root="./CIFAR10/test_set",
            train=False,
            download=True,
            transform=transform,
        )

    elif num_classes == 10:
        # Load training dataset
        train_set = torchvision.datasets.CIFAR10(
            root='./CIFAR10/train_set',
            train=True,
            download=True,
            transform=transform
        )

        # Load test set
        test_set = torchvision.datasets.CIFAR10(
            root="./CIFAR10/test_set",
            train=False,
            download=True,
            transform=transform,
        )

    else:
        raise Exception(f'CIFAR{num_classes} not found!')

    return train_set, test_set
