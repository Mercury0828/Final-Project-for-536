
from trojanvision.datasets import ImageSet
from torchvision.datasets import VisionDataset

import torch
import torchvision.transforms as transforms
import numpy as np
import glob
import os


class _IRaven(VisionDataset):
    def __init__(self, root: str, configure: str = 'center_single',
                 train: bool = True, **kwargs) -> None:
        super().__init__(root=root, **kwargs)
        self.train = train
        if train:
            npz_list = glob.glob(os.path.join(
                root, f'{configure}_train_*.npz'))
        else:
            npz_list_1 = glob.glob(os.path.join(
                root, f'{configure}_valid_*.npz'))
            npz_list_2 = glob.glob(os.path.join(
                root, f'{configure}_test_*.npz'))
            npz_list = npz_list_1 + npz_list_2
        data_list: list[torch.Tensor] = []
        label_list: list[int] = []
        meta_list: list[torch.Tensor] = []
        # structure_list: list[np.ndarray] = []

        for npz_path in npz_list:
            npz = np.load(npz_path)
            np_data: np.ndarray = npz['image']
            meta_data: np.ndarray = npz['meta_target']
            # structure_data: np.ndarray = npz['structure']
            data = torch.from_numpy(np_data).to(
                dtype=torch.float).div(255)
            label = npz['target']
            data_list.append(data)
            label_list.extend(label)
            meta_list.append(torch.from_numpy(meta_data).float())
            # structure_list.append(structure_data)
        self.data = torch.cat(data_list)
        self.targets = label_list
        self.meta_targets = torch.cat(meta_list)
        # self.structures = np.concatenate(structure_list)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # embedding = torch.zeros((6, 300), dtype=torch.float)
        # indicator = torch.zeros(1, dtype=torch.float)
        # element_idx = 0
        # for element in self.structures[index]:
        #     if element != '/':
        #         embedding[element_idx, :] = torch.tensor(
        #             self.embeddings.get(element), dtype=torch.float)
        #         element_idx += 1
        # if element_idx == 6:
        #     indicator[0] = 1.

        if self.train:
            idx = torch.randperm(8)
            new_img = img.clone()
            new_img[8:] = img[8:][idx]
            new_target = idx.tolist().index(target)
            img, target = new_img, new_target

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, self.meta_targets[index]

    def __len__(self) -> int:
        return len(self.data)


class IRaven(ImageSet):
    name = 'i-raven'
    num_classes = 8

    def __init__(self, configure: str = 'center_single',
                 data_shape: list[int] = [16, 80, 80], **kwargs):
        self.configure = configure
        self.data_shape = data_shape
        super().__init__(**kwargs)
        self.param_list['i-raven'] = ['configure']

    def get_transform(self, mode: str, **kwargs) -> transforms.Compose:
        return transforms.Compose([transforms.Resize(self.data_shape[-2:])])

    def _get_org_dataset(self, mode: str, configure: str = None,
                         **kwargs):
        configure = configure or self.configure
        return _IRaven(self.folder_path, configure=configure,
                       train=(mode == 'train'), **kwargs)
