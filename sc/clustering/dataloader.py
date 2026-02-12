from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from torchvision import transforms
import numpy as np


class AuxSpectraDataset(Dataset):
    def __init__(self, csv_fn, split_portion, train_val_test_ratios=(0.8, 0.15, 0.0),
             n_aux=0, transform=None, shuffle=True, random_seed=0):
        

        self.metadata = self._process_metadata(csv_fn, train_val_test_ratios)
        full_df = pd.read_csv(csv_fn, index_col=[0, 1], comment='#')
        if shuffle:
            full_df = full_df.sample(frac=1.0, random_state=random_seed)
        
        self.grid = np.array([float(col.strip('ENE_')) for col in full_df.columns if col.startswith('ENE_')])
        n_train_val_test = [int(len(full_df) * ratio)
                            for ratio in train_val_test_ratios]
        n_train_val_test[-1] = int(len(full_df)) - sum(n_train_val_test[:-1])
        portion_options = ['train', 'val', 'test']
        assert split_portion in portion_options
        i_prev = portion_options.index(split_portion)
        start = sum(n_train_val_test[:i_prev])
        stop = sum(n_train_val_test[:i_prev+1])
        df = full_df.iloc[start:stop]
    
        assert "ENE_" in df.columns.to_list()[n_aux]
        if n_aux > 0:
            assert "ENE_" not in df.columns.to_list()[n_aux-1]
            assert "AUX_" in df.columns.to_list()[0]
            assert "AUX_" in df.columns.to_list()[n_aux-1]
        data = df.to_numpy()
        self.spec = data[:, n_aux:]
        if n_aux > 0:
            self.aux = data[:, :n_aux]
        else:
            self.aux = None
        self.transform = transform
        self.atom_index = df.index.to_list()
    
    def _process_metadata(self, file_path, split_ratio):
        metadata = {
                "path": file_path,
                "train_test_val_split_ratio": split_ratio
        }
        return metadata
    
    def __len__(self):
        return self.spec.shape[0]


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.aux is None:
            sample = self.spec[idx], np.zeros((0,), dtype=np.float32)
        else:
            sample = self.spec[idx], self.aux[idx]
        if self.transform is not None:
            sample = [self.transform(
                x) if x is not None else None for x in sample]
        return sample


class ToTensor(object):
    def __call__(self, sample):
        return torch.Tensor(sample)


def get_dataloaders(csv_fn, batch_size, train_val_test_ratios=(0.8, 0.15, 0.0), n_aux=0,
                    shuffle=True, random_seed=0):

    transform_list = transforms.Compose([ToTensor()])
    ds_train, ds_val, ds_test = [
        AuxSpectraDataset(
            csv_fn, p, train_val_test_ratios,
            transform=transform_list, n_aux=n_aux,
            shuffle=shuffle, random_seed=random_seed
        )
        for p in ["train", "val", "test"]
    ]
    

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(ds_val, batch_size=batch_size,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader
