import torch
import pandas as pd
import os
from tqdm import tqdm
import bisect

def split_vehicle_indices(indices_path, dataset_dirs, save_dir, segment_length=10):
    IGNORE_INDEX = -1

    # 1. Load previously saved indices
    splits = torch.load(indices_path)
    print(f"Splits keys: {splits.keys()}")

    # 2. Load view_recorder1_6m columns from CSVs
    view_labels_list = []
    usable_lens = []

    for d in tqdm(dataset_dirs, desc="Loading label CSVs"):
        label_dir = d.replace('ASPED_v2_npy', 'ASPED_v2_npy_labels_with_bus')
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

        label_files = []
        for root, dirs, files in os.walk(label_dir):
            for f in files:
                if f.endswith('.csv'):
                    label_files.append(os.path.join(root, f))
        
        if len(label_files) == 0:
            raise FileNotFoundError(f"No CSV files found under {label_dir} (including subfolders)")

        labels = []
        for csv_path in sorted(label_files):
            df = pd.read_csv(csv_path)
            if 'view_recorder1_6m' not in df.columns:
                raise ValueError(f"'view_recorder1_6m' missing in {csv_path}")
            labels.append(torch.tensor(df['view_recorder1_6m'].values, dtype=torch.long))

        labels = torch.cat(labels, dim=0)
        view_labels_list.append(labels)
        usable_lens.append(len(labels) - segment_length)

    cumulative_sizes = [0] + list(torch.cumsum(torch.tensor(usable_lens), dim=0)[:-1])
    print(f"Total datasets: {len(view_labels_list)}")
    print(f"Cumulative usable sizes: {cumulative_sizes}")

    # 3. Split each (train / val / test) separately
    split_w_vehicle = dict()
    split_wo_vehicle = dict()

    for split_name in ['train', 'val', 'test']:
        all_indices = splits[split_name]
        w_vehicle = []
        wo_vehicle = []

        for idx in tqdm(all_indices, desc=f"Processing {split_name}"):
            idx = idx.item()

            dataset_idx = bisect.bisect_right(cumulative_sizes, idx) - 1
            local_idx = idx - cumulative_sizes[dataset_idx]

            if not (0 <= local_idx < usable_lens[dataset_idx]):
                continue  # usable length 넘어가면 skip

            view_labels = view_labels_list[dataset_idx]
            segment_views = view_labels[local_idx : local_idx + segment_length]

            if (segment_views == IGNORE_INDEX).any():
                print(f"❌ {split_name} {idx} has IGNORE_INDEX")
                input()
                continue

            if (segment_views == 1).any():
                w_vehicle.append(idx)
            else:
                wo_vehicle.append(idx)

        split_w_vehicle[split_name] = torch.tensor(w_vehicle, dtype=torch.long)
        split_wo_vehicle[split_name] = torch.tensor(wo_vehicle, dtype=torch.long)

        print(f"✅ [{split_name}] Found {len(w_vehicle)} WITH vehicle, {len(wo_vehicle)} WITHOUT vehicle.")

    # 4. Save results
    os.makedirs(save_dir, exist_ok=True)
    torch.save(split_w_vehicle, os.path.join(save_dir, "ASPED_vb_mod10_wVehicle_6m.pth"))
    torch.save(split_wo_vehicle, os.path.join(save_dir, "ASPED_vb_mod10_woVehicle_6m.pth"))

    print(f"\n✅ Saved at {save_dir}")

if __name__ == "__main__":
    indices_path = "/media/backup_SSD/PedestrianDetection/indices/ASPED_vb_mod10.pth"
    dataset_dirs = [
        "/media/backup_SSD/ASPED_v2_npy/Session_07262023",
        "/media/backup_SSD/ASPED_v2_npy/Session_08092023",
        "/media/backup_SSD/ASPED_v2_npy/Session_11072023",
        "/media/backup_SSD/ASPED_v2_npy/Session_11282023",
    ]
    save_dir = "/media/backup_SSD/PedestrianDetection/indices"
    split_vehicle_indices(indices_path, dataset_dirs, save_dir)
