import argparse
import csv
import os
import re
import time
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


DATASET_META = {
    "UCM": (
        "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
        "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
        "harbor", "intersection", "mediumresidential", "mobilehomepark",
        "overpass", "parkinglot", "river", "runway", "sparseresidential",
        "storagetanks", "tenniscourt",
    ),
    "AID": (
        "airport", "bareland", "baseballfield", "beach", "bridge", "center",
        "church", "commercial", "denseresidential", "desert", "farmland",
        "forest", "industrial", "meadow", "mediumresidential", "mountain",
        "park", "parking", "playground", "pond", "port", "railwaystation",
        "resort", "river", "school", "sparseresidential", "square", "stadium",
        "storagetanks", "viaduct",
    ),
    "NWPU": (
        "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
        "bridge", "chaparral", "church", "circular_farmland", "cloud",
        "commercial_area", "dense_residential", "desert", "forest", "freeway",
        "golf_course", "ground_track_field", "harbor", "industrial_area",
        "intersection", "island", "lake", "meadow", "medium_residential",
        "mobile_home_park", "mountain", "overpass", "palace", "parking_lot",
        "railway", "railway_station", "rectangular_farmland", "river",
        "roundabout", "runway", "sea_ice", "ship", "snowberg",
        "sparse_residential", "stadium", "storage_tank", "tennis_court",
        "terrace", "thermal_power_station", "wetland",
    ),
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def default_loader(path):
    return Image.open(path).convert("RGB")


class PathImageDataset(Dataset):
    def __init__(self, samples, transform=None, loader=default_loader):
        self.samples = list(samples)
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, class_name = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, path, label, class_name


def build_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


def collate_batch(batch):
    images, paths, labels, class_names = zip(*batch)
    return torch.stack(images, dim=0), list(paths), list(labels), list(class_names)


def infer_dataset_from_path(path):
    normalized_parts = [part.upper() for part in os.path.normpath(path).split(os.sep)]
    for dataset in DATASET_META:
        if dataset in normalized_parts:
            return dataset
    return None


def infer_ipc_from_path(path):
    for part in reversed(os.path.normpath(path).split(os.sep)):
        match = re.fullmatch(r"IPC_(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def safe_name_from_path(path, max_parts=8):
    normalized = os.path.normpath(os.path.abspath(path))
    parts = [part for part in normalized.split(os.sep) if part and part not in (os.sep,)]
    lower_parts = [part.lower() for part in parts]
    if "generated_data" in lower_parts:
        selected = parts[lower_parts.index("generated_data") + 1:]
    else:
        selected = parts[-max_parts:]
    raw_name = "__".join(selected) if selected else "distilled"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._-")
    return safe_name or "distilled"


def find_class_dir(distilled_dir, label, class_name):
    candidates = [
        os.path.join(distilled_dir, f"{label}_{class_name}"),
        os.path.join(distilled_dir, class_name),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    prefix = f"{label}_"
    for entry in sorted(os.listdir(distilled_dir)):
        entry_path = os.path.join(distilled_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry == class_name or entry.startswith(prefix) or entry.endswith(f"_{class_name}"):
            return entry_path
    return None


def collect_distilled_samples(distilled_dir, dataset, ipc=None):
    classnames = DATASET_META[dataset]
    samples = []
    for label, class_name in enumerate(classnames):
        class_dir = find_class_dir(distilled_dir, label, class_name)
        if class_dir is None:
            raise FileNotFoundError(
                f"Class folder not found for label={label}, class={class_name} under {distilled_dir}"
            )

        image_paths = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(glob(os.path.join(class_dir, f"*{ext}")))
            image_paths.extend(glob(os.path.join(class_dir, f"*{ext.upper()}")))
        image_paths = sorted(set(image_paths))

        if ipc is not None and len(image_paths) < ipc:
            raise ValueError(f"Class {label}_{class_name} has {len(image_paths)} images, need IPC={ipc}")

        selected_paths = image_paths[:ipc] if ipc is not None else image_paths
        samples.extend((path, label, class_name) for path in selected_paths)

    if not samples:
        raise ValueError(f"No distilled images found in {distilled_dir}")
    return samples


def load_real_train_samples_by_label(root_dir, dataset):
    split_path = os.path.join("dataset", f"{dataset}_train.txt")
    classnames = DATASET_META[dataset]
    samples_by_label = {label: [] for label in range(len(classnames))}
    with open(split_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rel_path, label_text = line.split()[:2]
            label = int(label_text)
            full_path = rel_path if os.path.isabs(rel_path) else os.path.join(root_dir, rel_path)
            samples_by_label[label].append((full_path, label, classnames[label]))

    total = sum(len(samples) for samples in samples_by_label.values())
    if total == 0:
        raise ValueError(f"No real training images found in {split_path}")
    missing = [label for label, samples in samples_by_label.items() if not samples]
    if missing:
        raise ValueError(f"Real train split has no samples for labels: {missing}")
    return samples_by_label


def gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    return window_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def ssim_batch(x, y, window_size=11, sigma=1.5, data_range=1.0):
    channels = x.shape[1]
    window = gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    padding = window_size // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
    return ssim_map.mean(dim=(1, 2, 3))




def build_real_loaders(samples_by_label, transform, args):
    loaders = {}
    for label, samples in samples_by_label.items():
        loaders[label] = DataLoader(
            PathImageDataset(samples, transform=transform),
            batch_size=args.real_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
    return loaders


def topk_records(values, paths, top_k, largest):
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=largest)
    records = []
    for idx in order[:top_k]:
        path = paths[idx]
        records.append({
            "name": os.path.basename(path),
            "path": path,
            "value": float(values[idx]),
        })
    return records


def evaluate_privacy(args):
    top_k = 5
    distilled_dir = os.path.abspath(args.distilled_dir)
    dataset = (args.dataset or infer_dataset_from_path(distilled_dir) or "").upper()
    if dataset not in DATASET_META:
        raise ValueError("Could not infer dataset from --distilled_dir. Pass --dataset UCM, AID, or NWPU.")

    ipc = args.IPC if args.IPC is not None else infer_ipc_from_path(distilled_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    transform = build_transform(args.resolution)

    distilled_samples = collect_distilled_samples(distilled_dir, dataset, ipc=ipc)
    real_samples_by_label = load_real_train_samples_by_label(args.root_dir, dataset)
    real_loaders_by_label = build_real_loaders(real_samples_by_label, transform, args)
    real_count = sum(len(samples) for samples in real_samples_by_label.values())

    distilled_loader = DataLoader(
        PathImageDataset(distilled_samples, transform=transform),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    ssim_topk_mean_scores = []
    detail_rows = []
    timing = {"ssim": 0.0, "data": 0.0, "total": time.perf_counter()}

    print(
        f"[INFO] Dataset={dataset}, Distilled={len(distilled_samples)}, RealTrain={real_count}, "
        f"SearchSpace=same_class, Resolution={args.resolution}, Device={device}, "
        f"Metric=NN-SSIM, TopK={top_k}"
    )

    with torch.no_grad():
        for distilled_images, distilled_paths, distilled_labels, distilled_classes in tqdm(distilled_loader, desc="Distilled images"):
            distilled_image = distilled_images.to(device, non_blocking=True)
            distilled_path = distilled_paths[0]
            distilled_label = int(distilled_labels[0])
            distilled_class = distilled_classes[0]
            real_loader = real_loaders_by_label[distilled_label]

            ssim_values_all = []
            real_paths_all = []

            real_iter_start = time.perf_counter()
            for real_images, real_paths, _, _ in real_loader:
                timing["data"] += time.perf_counter() - real_iter_start
                real_images = real_images.to(device, non_blocking=True)
                real_paths_all.extend(real_paths)

                metric_start = time.perf_counter()
                ssim_values = ssim_batch(distilled_image.expand(real_images.shape[0], -1, -1, -1), real_images)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timing["ssim"] += time.perf_counter() - metric_start
                ssim_values_all.extend(ssim_values.detach().cpu().tolist())

                real_iter_start = time.perf_counter()

            ssim_top = topk_records(ssim_values_all, real_paths_all, top_k, largest=True)

            ssim_top5_mean = ""
            if ssim_top:
                ssim_top5_mean = float(np.mean([record["value"] for record in ssim_top]))
                ssim_topk_mean_scores.append(ssim_top5_mean)

            row = {
                "distilled_name": os.path.basename(distilled_path),
                "distilled_path": distilled_path,
                "distilled_label": distilled_label,
                "distilled_class": distilled_class,
                "same_class_real_count": len(real_samples_by_label[distilled_label]),
                "ssim_top5_mean": ssim_top5_mean,
            }
            for rank in range(1, top_k + 1):
                ssim_record = ssim_top[rank - 1] if rank <= len(ssim_top) else None
                row[f"ssim_top{rank}_name"] = "" if ssim_record is None else ssim_record["name"]
                row[f"ssim_top{rank}_path"] = "" if ssim_record is None else ssim_record["path"]
                row[f"ssim_top{rank}_value"] = "" if ssim_record is None else ssim_record["value"]
            detail_rows.append(row)

    timing["total"] = time.perf_counter() - timing["total"]
    ssim_topk_array = np.array(ssim_topk_mean_scores, dtype=np.float64)
    summary = {
        "dataset": dataset,
        "distilled_dir": distilled_dir,
        "ipc": "all" if ipc is None else ipc,
        "num_distilled": len(distilled_samples),
        "num_real_train": real_count,
        "search_space": "same_class",
        "top_k": top_k,
        "resolution": args.resolution,
        "nn_ssim_top5_mean": "" if ssim_topk_array.size == 0 else float(ssim_topk_array.mean()),
        "nn_ssim_top5_std": "" if ssim_topk_array.size == 0 else float(ssim_topk_array.std()),
        "time_total_sec": timing["total"],
        "time_ssim_sec": timing["ssim"],
        "time_data_sec": timing["data"],
    }
    return summary, detail_rows


def format_summary_lines(summary):
    lines = []
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    return lines


def write_summary_txt(path, summary):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("NN-SSIM Summary\n")
        handle.write("===============\n")
        handle.write("\n".join(format_summary_lines(summary)))
        handle.write("\n")


def write_details_csv(path, rows, top_k):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "distilled_name", "distilled_path", "distilled_label", "distilled_class", "same_class_real_count",
        "ssim_top5_mean",
    ]
    for rank in range(1, top_k + 1):
        fieldnames.extend([
            f"ssim_top{rank}_name", f"ssim_top{rank}_path", f"ssim_top{rank}_value",
        ])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    summary, detail_rows = evaluate_privacy(args)
    safe_name = safe_name_from_path(args.distilled_dir)

    output_txt = args.output_txt
    if output_txt is None:
        output_txt = os.path.join(args.output_dir, f"{summary['dataset']}__{safe_name}__privacy.txt")

    write_summary_txt(output_txt, summary)

    print(
        f"Dataset={summary['dataset']}, Distilled_Dir={summary['distilled_dir']}, IPC={summary['ipc']}, "
        f"Images={summary['num_distilled']}, RealTrain={summary['num_real_train']}, "
        f"SearchSpace={summary['search_space']}, "
        f"NN-SSIM Top5Mean={summary['nn_ssim_top5_mean']:.6f}, "
        f"Top5Std={summary['nn_ssim_top5_std']:.6f}, "
        f"TXT={output_txt}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to the IPC directory containing class subfolders.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional override: UCM, AID, or NWPU. Inferred from path when possible.")
    parser.add_argument("--IPC", type=int, default=None, help="Optional image count per class. Inferred from IPC_<N> path when possible; otherwise uses all images.")
    parser.add_argument("--root_dir", type=str, default="/proj/cvl/users/x_xuyon/Data/VisionLanguage/")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--real_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=5, help="Deprecated compatibility argument; NN-SSIM always uses top-5.")
    parser.add_argument("--output_dir", type=str, default="./nn_ssim")
    parser.add_argument("--output_txt", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
