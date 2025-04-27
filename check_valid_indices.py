
import torch
from tqdm import tqdm
from models.data_utils.dataset import ASPEDDataset
import bisect

IGNORE_INDEX = -1

def check_indices_and_labels(concat_dataset, indices_path, segment_length=10):
    print(f"Loading indices from {indices_path}")
    splits = torch.load(indices_path)

    problems = dict()

    cumulative_sizes = []
    total = 0
    for ds in concat_dataset.datasets:
        usable_len = len(ds.labels) - segment_length
        cumulative_sizes.append(total + usable_len)
        total += usable_len

    for split_name in ['train', 'val', 'test']:
        split_indices = splits[split_name]
        print(f"\nChecking split '{split_name}' with {len(split_indices)} samples...")

        for idx in tqdm(split_indices, desc=f"Checking {split_name}"):
            idx = idx.item()

            # Global idx → (dataset_idx, sample_idx) 매핑
            dataset_idx = bisect.bisect_right(cumulative_sizes, idx)

            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - cumulative_sizes[dataset_idx - 1]

            dataset = concat_dataset.datasets[dataset_idx]
            usable_len = len(dataset.labels) - segment_length

            if not (0 <= sample_idx < usable_len):
                print(f"❌ sample_idx {sample_idx} out of usable range in dataset {dataset_idx} (usable_len={usable_len})")
                if split_name not in problems:
                    problems[split_name] = []
                problems[split_name].append((idx, dataset_idx, sample_idx))
                continue

            label_segment = dataset.labels[sample_idx:sample_idx+segment_length]

            if torch.any(label_segment == IGNORE_INDEX):
                if split_name not in problems:
                    problems[split_name] = []
                problems[split_name].append((idx, dataset_idx, sample_idx))

    if problems:
        print("\n❌ Found problematic samples!")
        for split_name, bad_samples in problems.items():
            print(f"  Split '{split_name}': {len(bad_samples)} problematic samples")
        return problems
    else:
        print("\n✅ All indices and labels are clean!")
        return None


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
    indices_path = "indices/ASPED_va.pth"
    segment_length = 10

    concat_dataset = ASPEDDataset.from_dirs_vb(TEST_DIR, segment_length=segment_length)

    problems = check_indices_and_labels(concat_dataset, indices_path, segment_length=segment_length)

    if problems:
        for split_name, bad_samples in problems.items():
            print(f"\n=== Examples from split '{split_name}': ===")
            for idx, dataset_idx, sample_idx in bad_samples[:5]:  # 최대 5개 샘플만 출력
                print(f"  idx={idx}, dataset_idx={dataset_idx}, sample_idx={sample_idx}")
