from torchvision import datasets, transforms

N_CLASSES = 10

def load_train_data():
    train_data = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())

    return train_data.data.numpy(), train_data.targets.numpy()


def load_test_data():
    test_data = datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())

    return test_data.data.numpy(), test_data.targets.numpy()