from torchvision.datasets import MNIST
from torchvision import transforms


def load_data():

    transform = transforms.ToTensor()

    train_dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    X_train = train_dataset.data.float() / 255.0
    y_train = train_dataset.targets

    X_test = test_dataset.data.float() / 255.0
    y_test = test_dataset.targets

    return X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy()