import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

import torchvision
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset


class customDataLoader(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        # ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
         #["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        # ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        #["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

class custom_dataset(VisionDataset):
    """
    A dataset that returns a split of the CIFAR10 dataset.
    input: dataset_name, train, transform, target_transform, download, split_number, split_id, iid
    """

    def __init__(
        self,
        dataset_name,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        split_number=10,
        split_id=0,
        iid=True,
    ):
        super(custom_dataset, self).__init__(root="./data", transform=transform, target_transform=target_transform)
        self.dataset_name = dataset_name
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.split_number = split_number
        self.split_id = split_id
        self.iid = iid
        self.data = []
        self.targets = []
        self.data, self.targets = self.get_data()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_data(self):
        if self.dataset_name == "cifar10":
            dataset = torchvision.datasets.CIFAR10(root="./data", train=self.train, download=self.download)
        elif self.dataset_name == "cifar100":
            dataset = torchvision.datasets.CIFAR100(root="./data", train=self.train, download=self.download)
        elif self.dataset_name == "svhn":
            dataset = torchvision.datasets.svhn(root="./data", train=self.train, download=self.download)
        else:
            raise ValueError("Dataset not found")

        if self.iid:
            data = dataset.data
            targets = dataset.targets
            data_len = len(data)
            data_len_per_split = data_len // self.split_number
            data = data[self.split_id * data_len_per_split : (self.split_id + 1) * data_len_per_split]
            targets = targets[self.split_id * data_len_per_split : (self.split_id + 1) * data_len_per_split]
        else:
            # non-iid split
            Adata = [[] for _ in range(10)]
            Atargets = [[] for _ in range(10)]
            data=[]
            targets=[]
            data_len = len(dataset.data)
            class_num = 10
            if self.dataset_name == "cifar100":
                class_num = 100
            data_len_per_class = data_len // class_num
            
            # for i in range(class_num):
            #     data.append(dataset.data[i * data_len_per_class : (i + 1) * data_len_per_class])
            #     targets.append(dataset.targets[i * data_len_per_class : (i + 1) * data_len_per_class])
            
            for n in range(data_len):
                Adata[dataset.targets[n]].append(dataset.data[n])
                Atargets[dataset.targets[n]].append(dataset.targets[n])

            numdata = max( 10 // self.split_number,1)
            for d in Adata[self.split_id * numdata : (self.split_id + 1) * numdata]:
                data += d
            for t in Atargets[self.split_id * numdata : (self.split_id + 1) * numdata]:
                targets += t
        return data, targets
    
if __name__ == "__main__":
    dataset = custom_dataset("cifar10", train=True, split_number=4, split_id=0, iid=False)
    print(len(dataset))
    print(dataset[0])
    print(dataset[-1])
    
    dataset = custom_dataset("cifar10", train=True, split_number=4, split_id=1, iid=False)
    print(len(dataset))
    print(dataset[0])
    print(dataset[-1])
    
    dataset = custom_dataset("cifar10", train=True, split_number=4, split_id=2, iid=False)
    print(len(dataset))
    print(dataset[0])
    print(dataset[-1])
    
    dataset = custom_dataset("cifar10", train=True, split_number=4, split_id=3, iid=False)
    print(len(dataset))
    print(dataset[0])
    print(dataset[-1])