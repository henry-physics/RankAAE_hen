from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
from torchvision import transforms
import numpy as np
from pathlib import Path
from io import StringIO

### !!!! IMPORTANT 
### currently, the train/val/test split are passed in from trainer and are NOT set in this file. 

# ----------------- utilities -----------------

def _atomic_write_text(path: Path, text: str):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)  # atomic on POSIX + Windows


def _atomic_write_csv(path: Path, df: pd.DataFrame):
    buf = StringIO()
    df.to_csv(buf, index=False)
    _atomic_write_text(path, buf.getvalue())


def _get_jjj_values(df: pd.DataFrame) -> np.ndarray:
    """
    Returns jjj as integers in [0,127] for each row.
    Supports either MultiIndex (i, jjj) or single index "i_jjj".
    """
    if getattr(df.index, "nlevels", 1) >= 2:
        jjj = df.index.get_level_values(1)
        return jjj.astype(str).astype(int).to_numpy()
    else:
        ids = df.index.astype(str)
        return np.array([int(s.split("_")[1]) for s in ids], dtype=int)

def make_split_bins(train_val_test_ratios=(0.8, 0.2, 0.0), available_bins=None, seed=22):
    """
    Returns {"train": set(...), "val": set(...), "test": set(...)} where the bins
    come ONLY from available_bins (e.g. unique jjj values found in the CSV).
    """
    assert len(train_val_test_ratios) == 3

    if available_bins is None:
        raise ValueError("Pass available_bins (e.g. unique jjj from the dataframe).")

    available_bins = np.array(sorted(set(int(x) for x in available_bins)), dtype=int)
    n = len(available_bins)

    counts = [
        int(train_val_test_ratios[0] * n),
        int(train_val_test_ratios[1] * n),
        int(train_val_test_ratios[2] * n),
    ]
    counts[0] += (n - sum(counts))  # remainder -> train
    assert sum(counts) == n and all(c >= 0 for c in counts)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(available_bins)

    train_bins = set(shuffled[:counts[0]])
    val_bins   = set(shuffled[counts[0]:counts[0] + counts[1]])
    test_bins  = set(shuffled[counts[0] + counts[1]:])

    return {"train": train_bins, "val": val_bins, "test": test_bins}



def split_df_by_bins(full_df: pd.DataFrame, split_bins: dict, portion: str):
    assert portion in split_bins
    jjj_vals = _get_jjj_values(full_df)
    mask = np.isin(jjj_vals, list(split_bins[portion]))
    return full_df.loc[mask]

def save_split_ids(full_df: pd.DataFrame, dfs_by_split: dict, split_bins: dict,
                   out_dir=".", prefix="split", extra_meta=None):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "n_total_rows": int(len(full_df)),
        "n_rows": {k: int(len(v)) for k, v in dfs_by_split.items()},
        "bins": {k: sorted(list(v)) for k, v in split_bins.items()},
        "index_names": list(full_df.index.names),
    }
    if extra_meta:
        manifest.update(extra_meta)

    # write 3 CSVs: ids for each split
    for split_name, df_split in dfs_by_split.items():
        idx_df = df_split.index.to_frame(index=False)
        _atomic_write_csv(out_dir / f"{prefix}_ids_{split_name}.csv", idx_df)




# ----------------- Dataset -----------------

class AuxSpectraDataset(Dataset):
    def __init__(self, df: pd.DataFrame, n_aux=0, transform=None):
        # energy grid
        self.grid = np.array([float(col.strip("ENE_")) for col in df.columns if col.startswith("ENE_")])

        assert "ENE_" in df.columns.to_list()[n_aux]
        if n_aux > 0:
            assert "ENE_" not in df.columns.to_list()[n_aux - 1]
            assert "AUX_" in df.columns.to_list()[0]
            assert "AUX_" in df.columns.to_list()[n_aux - 1]

        data = df.to_numpy()
        self.spec = data[:, n_aux:]
        self.aux = data[:, :n_aux] if n_aux > 0 else None
        self.transform = transform
        self.atom_index = df.index.to_list()

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

def get_dataloaders(csv_fn, batch_size, train_val_test_ratios=(0.8, 0.2, 0.0),
                    n_aux=0, seed=22, save_split_files=True, out_dir="."):

    transform_list = transforms.Compose([ToTensor()])

    # read ONCE
    full_df = pd.read_csv(csv_fn, index_col=[0, 1], comment="#")

    # ONLY bins that actually exist in the CSV
    existing_jjj = np.unique(_get_jjj_values(full_df))

    split_bins = make_split_bins(train_val_test_ratios, available_bins=existing_jjj, seed=seed)

    
    dfs = {p: split_df_by_bins(full_df, split_bins, p) for p in ["train", "val", "test"]}


    if save_split_files:
        save_split_ids(
            full_df,
            dfs_by_split=dfs,
            split_bins=split_bins,
            out_dir=out_dir,
            prefix="split",
            extra_meta={
                "source_csv": str(Path(csv_fn).resolve()),
                "ratios": list(train_val_test_ratios),
                "seed": int(seed),
                "existing_jjj": [int(x) for x in existing_jjj],
            }
        )

    ds_train = AuxSpectraDataset(dfs["train"], n_aux=n_aux, transform=transform_list)
    ds_val   = AuxSpectraDataset(dfs["val"],   n_aux=n_aux, transform=transform_list)
    ds_test  = AuxSpectraDataset(dfs["test"],  n_aux=n_aux, transform=transform_list)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=0)

    print("len of train, val, and test loaders")
    print(len(train_loader), len(val_loader), len(test_loader))

    return train_loader, val_loader, test_loader
