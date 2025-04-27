import torch
import os
from tqdm import tqdm
from typing import Tuple

from models.data_utils.dataset import ASPEDDataset

IGNORE_INDEX = -1

def find_valid_indices_per_dataset(concat_dataset, segment_length=10):
    all_valid_indices = []
    cumulative_offset = 0

    for dataset_idx, ds in enumerate(tqdm(concat_dataset.datasets, desc="Finding valid indices")):
        labels = ds.labels
        usable_len = len(labels) - segment_length

        ds_valid_indices = []
        for idx in range(usable_len):
            label_segment = labels[idx:idx + segment_length]
            if (label_segment != IGNORE_INDEX).all():
                ds_valid_indices.append(cumulative_offset + idx)

        all_valid_indices.extend(ds_valid_indices)

        cumulative_offset += usable_len

    return torch.tensor(all_valid_indices)


def split_valid_indices(valid_indices: torch.Tensor, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    shuffled = valid_indices[torch.randperm(valid_indices.shape[0])]

    num = shuffled.shape[0] // 10
    train_idx = shuffled[:8*num]
    val_idx = shuffled[8*num:9*num]
    test_idx = shuffled[9*num:]

    return train_idx, val_idx, test_idx


def save_split_indices(train_idx, val_idx, test_idx, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'train': train_idx, 'val': val_idx, 'test': test_idx}, save_path)


if __name__ == "__main__":
    TEST_DIR = ["/media/chan/backup_SSD2/ASPED_v1_npy/Session_5242023", "/media/chan/backup_SSD2/ASPED_v1_npy/Session_6012023", 
                "/media/chan/backup_SSD2/ASPED_v1_npy/Session_6072023", "/media/chan/backup_SSD2/ASPED_v1_npy/Session_6212023",
                "/media/chan/backup_SSD2/ASPED_v1_npy/Session_6282023"] # V.A
    # TEST_DIR = [
    #     "/media/backup_SSD/ASPED_v2_npy/Session_07262023",
    #     "/media/backup_SSD/ASPED_v2_npy/Session_08092023",
    #     "/media/backup_SSD/ASPED_v2_npy/Session_11072023",
    #     "/media/backup_SSD/ASPED_v2_npy/Session_11282023",
    # ] # V.B
    save_path = "indices/ASPED_va.pth"
    segment_length = 10

    # 1. Load entire dataset
    concat_dataset = ASPEDDataset.from_dirs_vb(TEST_DIR, segment_length=segment_length)

    # 2. Find valid indices
    valid_indices = find_valid_indices_per_dataset(concat_dataset, segment_length=segment_length)
    print(f"✅ Found {len(valid_indices)} valid indices!")

    # 3. Split
    train_idx, val_idx, test_idx = split_valid_indices(valid_indices, seed=42)

    # 4. Save
    save_split_indices(train_idx, val_idx, test_idx, save_path)
    print(f"✅ Saved splits to {save_path}")
