# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseLoader(DataLoader):
    """
    Base class for all data loaders.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=1, collate_fn=default_collate, validation_split=0.1):
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'collate_fn': collate_fn
        }
        self.validation_split = validation_split
        self.sampler, self.valid_sampler = self.split_sampler(self.validation_split)
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def split_sampler(self, split):
        if split == 0:
            return None, None
        else:
            n_samples = len(self.dataset)
            idx_samples = np.arange(n_samples)
            np.random.seed(0)
            np.random.shuffle(idx_samples)

            if isinstance(split, int):
                assert 0 < split < n_samples, "validation set size should be configured between the 0 and the length of the dataset."
                len_valid = split
            else:
                assert 0 < split < 1, "validation set size should be configured between the 0 and the 1."
                len_valid = int(n_samples * split)

            valid_idx = idx_samples[0:len_valid]
            train_idx = np.delete(idx_samples, np.arange(0, len_valid))
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            # turn off shuffle option which is mutually exclusive with sampler
            self.init_kwargs['shuffle'] = False

            return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
