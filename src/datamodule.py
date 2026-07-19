import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir="data",
        batch_size=64,
        num_workers=4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(root=self.data_dir, train=True, transform=self.transform)

            self.mnist_train, self.mnist_valid = random_split(
                dataset=mnist_full,
                lengths=[55_000, 5_000],
            )

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                root=self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.mnist_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
