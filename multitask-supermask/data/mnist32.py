import os
import torch
from torchvision import datasets, transforms

from args import args

class MNIST32:
    def __init__(self):
        super(MNIST32, self).__init__()

        data_root = os.path.join(args.data, "mnist")

        use_cuda = torch.cuda.is_available()

        train_dataset = datasets.MNIST(
                data_root,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(32), transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_root,
                train=False,
                transform=transforms.Compose(
                    [transforms.Resize(32), transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )

class FashionMNIST32:
    def __init__(self):
        super(FashionMNIST32, self).__init__()

        data_root = os.path.join(args.data, "mnist")

        use_cuda = torch.cuda.is_available()

        train_dataset = datasets.FashionMNIST(
                data_root,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(32), transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
        self.val_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                data_root,
                train=False,
                transform=transforms.Compose(
                    [transforms.Resize(32), transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
