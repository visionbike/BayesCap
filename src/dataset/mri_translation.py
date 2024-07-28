from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["MTDataset"]


class MTDataset(Dataset):
    """
    Customize the dataset loading function and prepare mri image translation data in advanced,
    can act as supervised or un-supervised based on file path lists.
    """

    def __init__(self,
                 root1: str, root2: str,
                 flist1: list, flist2: list,
                 transform1: None = None, transform2: None = None,
                 do_aug: bool = False) -> None:
        """

        :param flist1:
        :param flist2:
        :param transform1:
        :param transform2:
        :param do_aug:
        """
        super().__init__()
        self.flist1 = flist1
        self.flist2 = flist2
        self.transform1 = transform1
        self.transform2 = transform2
        self.do_aug = do_aug

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        img_path_1 = self.flist1[index]
        img1 = np.load(str(Path(img_path_1)))
        img_path_2 = self.flist2[index]
        img2 = np.load(str(Path(img_path_2)))
        if self.transform1 is not None:
            img1 = self.transform1(img1)
            img2 = self.transform2(img2)
        if self.do_aug:
            p1 = random.random()
            if p1 < 0.5:
                img1, img2 = torch.fliplr(img1), torch.fliplr(img2)
            p2 = random.random()
            if p2 < 0.5:
                img1, img2 = torch.flipud(img1), torch.flipud(img2)
        return img1, img2

    def __len__(self) -> int:
        return len(self.flist1)
