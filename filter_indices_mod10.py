import torch
import os

def filter_indices_mod10(load_path, save_path, segment_length=10):
    splits = torch.load(load_path)
    print(f"✅ Loaded splits from {load_path}")

    filtered_splits = {}
    for split_name, indices in splits.items():
        filtered = indices[indices % segment_length == 0]
        filtered_splits[split_name] = filtered
        print(f"Split '{split_name}': {len(indices)} -> {len(filtered)} after filtering (mod {segment_length})")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(filtered_splits, save_path)
    print(f"✅ Saved filtered splits to {save_path}")

if __name__ == "__main__":
    load_path = "/media/backup_SSD/PedestrianDetection/indices/ASPED_va.pth"
    save_path = "/media/backup_SSD/PedestrianDetection/indices/ASPED_va_mod10.pth"

    filter_indices_mod10(load_path, save_path, segment_length=10)
