from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from torchvision import transforms
import numpy as np


# dont forget to adjust train/val split 
# dont forget to adjust random seed for shuffling jjj into bins 

class AuxSpectraDataset(Dataset):
    def __init__(self, csv_fn, split_portion, train_val_test_ratios=(0.8, 0.2, 0.0),
                 n_aux=0, transform=None):

        self.metadata = self._process_metadata(csv_fn, train_val_test_ratios)

        full_df = pd.read_csv(csv_fn, index_col=[0, 1], comment='#')

        # energy grid
        self.grid = np.array([float(col.strip('ENE_')) for col in full_df.columns if col.startswith('ENE_')])

        # ---- bin-based split (jjj groups) ----
        df = self._split_df_by_jjj_bins(full_df, split_portion, train_val_test_ratios, n_bins=128)

        # ---- same as before from here ----
        assert "ENE_" in df.columns.to_list()[n_aux]
        if n_aux > 0:
            assert "ENE_" not in df.columns.to_list()[n_aux-1]
            assert "AUX_" in df.columns.to_list()[0]
            assert "AUX_" in df.columns.to_list()[n_aux-1]

        data = df.to_numpy()
        self.spec = data[:, n_aux:]
        self.aux = data[:, :n_aux] if n_aux > 0 else None
        self.transform = transform
        self.atom_index = df.index.to_list()

    def _process_metadata(self, file_path, split_ratio):
        return {"path": file_path, "train_test_val_split_ratio": split_ratio}

    def _get_jjj_values(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns jjj as integers in [0,127] for each row.
        Supports either MultiIndex (i, jjj) or single index "i_jjj".
        """
        if getattr(df.index, "nlevels", 1) >= 2:
            jjj = df.index.get_level_values(1)
            # allow '000' strings or ints
            return jjj.astype(str).astype(int).to_numpy()
        else:
            # single index like "0_000"
            ids = df.index.astype(str)
            return np.array([int(s.split("_")[1]) for s in ids], dtype=int)

    def _split_df_by_jjj_bins(self, full_df, split_portion, ratios, n_bins=128):
        portion_options = ["train", "val", "test"]
        assert split_portion in portion_options

        # decide how many *bins* go to each split
        counts = [int(ratios[0] * n_bins), int(ratios[1] * n_bins), int(ratios[2] * n_bins)]
        # put remainder into train (so we never miss a bin due to truncation)
        counts[0] += (n_bins - sum(counts))
        assert sum(counts) == n_bins and all(c >= 0 for c in counts)

        # bin IDs are 0..127 (some bins may have 0 rows; that's fine)
        # seed 
        rng = np.random.default_rng(22)
        all_bins = np.arange(n_bins, dtype=int)
        shuffled = rng.permutation(all_bins)
        train_bins = set(shuffled[:counts[0]])
        val_bins   = set(shuffled[counts[0]:counts[0] + counts[1]])
        test_bins  = set(shuffled[counts[0] + counts[1]:])
        


        split_bins = {"train": train_bins, "val": val_bins, "test": test_bins}[split_portion]

        jjj_vals = self._get_jjj_values(full_df)
        mask = np.isin(jjj_vals, list(split_bins))
        return full_df.loc[mask]

    def __len__(self):
        return self.spec.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.aux is None:
            sample = self.spec[idx], np.array([0.0])
        else:
            sample = self.spec[idx], self.aux[idx]
        if self.transform is not None:
            sample = [self.transform(x) if x is not None else None for x in sample]
        return sample


class ToTensor(object):
    def __call__(self, sample):
        return torch.Tensor(sample)


def get_dataloaders(csv_fn, batch_size, train_val_test_ratios=(0.8, 0.2, 0), n_aux=0):
    transform_list = transforms.Compose([ToTensor()])
    ds_train,  ds_val, ds_test = [AuxSpectraDataset(
        csv_fn, p, train_val_test_ratios, transform=transform_list, n_aux=n_aux)
        for p in ["train", "val", "test"]]

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(ds_val, batch_size=batch_size,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, num_workers=0, pin_memory=False)

    print("len of train, val, and test loaders")
    print(len(train_loader), len(val_loader), len(test_loader))

    return train_loader, val_loader, test_loader
