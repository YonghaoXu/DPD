import argparse
import os
import re
from glob import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import tools.model as models
from dataset.scene_dataset import scene_dataset


DATASET_META = {
    "UCM": (
        21,
        (
            "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
            "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
            "harbor", "intersection", "mediumresidential", "mobilehomepark",
            "overpass", "parkinglot", "river", "runway", "sparseresidential",
            "storagetanks", "tenniscourt",
        ),
    ),
    "AID": (
        30,
        (
            "airport", "bareland", "baseballfield", "beach", "bridge", "center",
            "church", "commercial", "denseresidential", "desert", "farmland",
            "forest", "industrial", "meadow", "mediumresidential", "mountain",
            "park", "parking", "playground", "pond", "port", "railwaystation",
            "resort", "river", "school", "sparseresidential", "square", "stadium",
            "storagetanks", "viaduct",
        ),
    ),
    "NWPU": (
        45,
        (
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
    ),
}


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def default_loader(path):
    return Image.open(path).convert("RGB")


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None, loader=default_loader):
        self.image_paths = list(image_paths)
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.loader(self.image_paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image


def build_eval_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def build_dataloader_kwargs(batch_size, shuffle, num_workers, prefetch_factor):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def read_stats_file(filepath):
    if not filepath.endswith(".npz"):
        raise ValueError(f"Stats file must be .npz: {filepath}")
    data = np.load(filepath)
    mu = data["mu"][:]
    sigma = data["sigma"][:]
    data.close()
    return mu, sigma


def save_stats_file(filepath, mu, sigma):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez(filepath, mu=mu, sigma=sigma)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean vectors have different lengths: {mu1.shape}, {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariances have different dimensions: {sigma1.shape}, {sigma2.shape}")

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            max_imag = np.max(np.abs(covmean.imag))
            raise ValueError(f"FID sqrtm has non-negligible imaginary component: {max_imag}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def clean_state_dict(state_dict):
    return {key[len("module."):] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return clean_state_dict(checkpoint[key]), checkpoint
    if isinstance(checkpoint, dict):
        return clean_state_dict(checkpoint), {}
    raise TypeError(f"Unsupported checkpoint format: {path}")


def build_inception(num_classes, checkpoint_path, device):
    model = models.inception_v3(pretrained=False, aux_logits=False, transform_input=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict, metadata = load_checkpoint_state(checkpoint_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata


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


def build_generated_setting_labels(setting_label, num_generated_sets, generated_set_id=None):
    if generated_set_id is not None:
        return [f"{setting_label}-gid{generated_set_id}"]
    if num_generated_sets <= 1:
        return [str(setting_label)]
    return [f"{setting_label}-gid{i}" for i in range(1, num_generated_sets + 1)]


def build_distilled_dirs(args):
    if args.distilled_dir is not None:
        distilled_dir = os.path.abspath(args.distilled_dir)
        dataset = (args.dataset or infer_dataset_from_path(distilled_dir) or "").upper()
        if dataset not in DATASET_META:
            raise ValueError("Could not infer dataset from --distilled_dir. Pass --dataset UCM, AID, or NWPU.")

        ipc = args.IPC if args.IPC is not None else infer_ipc_from_path(distilled_dir)
        return dataset, ipc, [(None, distilled_dir)]

    dataset = (args.dataset or "").upper()
    if dataset not in DATASET_META:
        raise ValueError("Pass --dataset UCM, AID, or NWPU when --distilled_dir is not used.")
    if args.sd_root is None:
        raise ValueError("Pass either --distilled_dir or --sd_root.")
    if args.setting_tag is None:
        raise ValueError("Pass --setting_tag when using --sd_root.")
    if args.IPC is None:
        raise ValueError("Pass --IPC when using --sd_root.")

    mode = args.mode.lower()
    setting_labels = (
        [str(args.setting_tag)]
        if mode == "full"
        else build_generated_setting_labels(args.setting_tag, args.num_generated_sets, args.generated_set_id)
    )
    distilled_dirs = []
    for setting_label in setting_labels:
        distilled_dir = os.path.abspath(
            os.path.join(args.sd_root, dataset, mode, setting_label, f"IPC_{args.IPC}")
        )
        distilled_dirs.append((setting_label, distilled_dir))
    return dataset, args.IPC, distilled_dirs


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


def collect_distilled_paths(distilled_dir, classnames, ipc=None, extensions=IMAGE_EXTENSIONS):
    if not os.path.isdir(distilled_dir):
        raise FileNotFoundError(f"Distilled directory not found: {distilled_dir}")

    image_paths = []
    for label, class_name in enumerate(classnames):
        class_dir = find_class_dir(distilled_dir, label, class_name)
        if class_dir is None:
            raise FileNotFoundError(
                f"Class folder not found for label={label}, class={class_name} under {distilled_dir}"
            )

        class_paths = []
        for ext in extensions:
            class_paths.extend(glob(os.path.join(class_dir, f"*{ext}")))
            class_paths.extend(glob(os.path.join(class_dir, f"*{ext.upper()}")))
        class_paths = sorted(set(class_paths))

        if ipc is not None and len(class_paths) < ipc:
            raise ValueError(f"Class {label}_{class_name} has {len(class_paths)} images, need IPC={ipc}")
        image_paths.extend(class_paths[:ipc] if ipc is not None else class_paths)

    if not image_paths:
        raise ValueError(f"No images found under class folders in: {distilled_dir}")
    return image_paths


def build_real_train_dataset(root_dir, dataset, transform):
    return scene_dataset(
        root_dir=root_dir,
        pathfile=os.path.join("dataset", f"{dataset}_train.txt"),
        transform=transform,
    )


def calc_stats(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


class FIDFeatureExtractor:
    def __init__(self, dataset, checkpoint_path, device):
        dataset = dataset.upper()
        if dataset not in DATASET_META:
            raise ValueError(f"Unknown dataset: {dataset}")
        num_classes, classnames = DATASET_META[dataset]
        self.dataset = dataset
        self.num_classes = num_classes
        self.classnames = classnames
        self.device = device
        self.model, self.metadata = build_inception(num_classes, checkpoint_path, device)
        self._features = None
        self._hook = self.model.fc.register_forward_hook(self._capture_fc_input)

    def close(self):
        self._hook.remove()

    def _capture_fc_input(self, module, inputs, output):
        self._features = inputs[0].detach()

    def forward_batch(self, images):
        self._features = None
        with torch.no_grad():
            self.model(images)
            if self._features is None:
                raise RuntimeError("Failed to capture Inception-v3 pool features from fc input.")
            return self._features.detach().cpu().numpy()

    def dataset_stats(self, dataset, batch_size=64, num_workers=4, prefetch_factor=4):
        loader = data.DataLoader(
            dataset,
            **build_dataloader_kwargs(batch_size, False, num_workers, prefetch_factor),
        )
        n_img = len(dataset)
        if n_img < 2:
            raise ValueError("Need at least two images to compute covariance for FID.")

        all_features = np.zeros((n_img, 2048), dtype=np.float64)
        cursor = 0
        for batch in tqdm(loader, desc="Extracting Inception features"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(self.device, non_blocking=True)
            features = self.forward_batch(images)
            batch_size_i = images.shape[0]
            all_features[cursor:cursor + batch_size_i] = features
            cursor += batch_size_i

        return calc_stats(all_features)


def compute_fid_for_dir(feature_extractor, real_mu, real_sigma, distilled_dir, classnames, ipc, transform, args):
    gen_paths = collect_distilled_paths(distilled_dir, classnames, ipc=ipc)
    gen_dataset = ImagePathDataset(gen_paths, transform=transform)
    print(f"[INFO] Calculating distilled stats from {distilled_dir} ({len(gen_dataset)} images)")
    gen_mu, gen_sigma = feature_extractor.dataset_stats(
        gen_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    return calculate_frechet_distance(real_mu, real_sigma, gen_mu, gen_sigma)


def resolve_log_file(args, dataset, ipc):
    if args.log_file is not None:
        return os.path.abspath(args.log_file)

    ipc_text = "all" if ipc is None else str(ipc)
    if args.distilled_dir is None:
        return os.path.abspath(
            os.path.join(
                args.save_path_prefix,
                dataset,
                args.mode.lower(),
                str(args.setting_tag),
                f"IPC_{ipc_text}",
                "fid_log.txt",
            )
        )

    safe_name = os.path.basename(os.path.normpath(args.distilled_dir)) or "distilled"
    return os.path.abspath(os.path.join(args.output_dir, f"{dataset}_{safe_name}_fid_log.txt"))


def format_fid_line(dataset, ipc, fid, generated_set=None, distilled_dir=None, mode=None, setting=None):
    ipc_text = "all" if ipc is None else str(ipc)
    parts = [f"Dataset={dataset}"]
    if mode is not None:
        parts.append(f"Mode={mode.lower()}")
    if setting is not None:
        parts.append(f"Setting={setting}")
    if generated_set is not None:
        parts.append(f"Generated_Set={generated_set}")
    parts.append(f"IPC={ipc_text}")
    parts.append(f"FID={fid:.6f}")
    if distilled_dir is not None:
        parts.append(f"Distilled_Dir={distilled_dir}")
    return ", ".join(parts)


def main(args):
    dataset, ipc, distilled_dirs = build_distilled_dirs(args)
    _, classnames = DATASET_META[dataset]
    checkpoint_path = args.inception_ckpt or os.path.join(args.pretrain_dir, dataset, "inception_v3.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Inception checkpoint not found: {checkpoint_path}. Run pretrain_inception.py first, "
            f"or pass --inception_ckpt."
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    transform = build_eval_transform(args.resolution)
    feature_extractor = FIDFeatureExtractor(dataset, checkpoint_path, device)

    real_stats_path = args.real_stats
    if real_stats_path is None and args.cache_real_stats:
        real_stats_path = os.path.join(args.stats_dir, dataset, f"train_inception_v3_res{args.resolution}.npz")

    if real_stats_path is not None and os.path.exists(real_stats_path) and not args.recompute_real_stats:
        print(f"[INFO] Loading cached real-train stats: {real_stats_path}")
        real_mu, real_sigma = read_stats_file(real_stats_path)
    else:
        real_dataset = build_real_train_dataset(args.root_dir, dataset, transform)
        print(f"[INFO] Calculating real-train stats from dataset/{dataset}_train.txt ({len(real_dataset)} images)")
        real_mu, real_sigma = feature_extractor.dataset_stats(
            real_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        if real_stats_path is not None:
            save_stats_file(real_stats_path, real_mu, real_sigma)
            print(f"[INFO] Saved real-train stats: {real_stats_path}")

    result_lines = []
    fid_values = []
    try:
        for generated_set, distilled_dir in distilled_dirs:
            if generated_set is not None:
                print(f"[INFO] Scoring distilled set: {generated_set}")
            fid = compute_fid_for_dir(
                feature_extractor,
                real_mu,
                real_sigma,
                distilled_dir,
                classnames,
                ipc,
                transform,
                args,
            )
            fid_values.append(fid)
            line = format_fid_line(
                dataset,
                ipc,
                fid,
                generated_set=generated_set,
                distilled_dir=distilled_dir if args.distilled_dir is not None else None,
                mode=args.mode if args.distilled_dir is None else None,
                setting=args.setting_tag if args.distilled_dir is None else None,
            )
            print(line)
            result_lines.append(line)

        if len(fid_values) > 1:
            fid_array = np.array(fid_values, dtype=np.float64)
            summary = (
                f"Dataset={dataset}, Mode={args.mode.lower()}, Setting={args.setting_tag}, "
                f"IPC={ipc}, Generated_Sets={len(fid_values)}, "
                f"Mean_FID={fid_array.mean():.6f}, Std_FID={fid_array.std():.6f}"
            )
            print(summary)
            result_lines.append(summary)

        log_file = resolve_log_file(args, dataset, ipc)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        write_mode = "w" if args.overwrite else "a"
        with open(log_file, write_mode, encoding="utf-8") as handle:
            for line in result_lines:
                handle.write(line + "\n")
        print(f"[INFO] Saved FID results to {log_file}")
    finally:
        feature_extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=str, default=None, help="Path to one IPC directory containing class subfolders.")
    parser.add_argument("--sd_root", type=str, default=None, help="Root of generated distilled data, used with --dataset/--mode/--setting_tag/--IPC.")
    parser.add_argument("--mode", type=str, default="template", help="Distilled data mode folder, e.g. template, prototype, or full.")
    parser.add_argument("--setting_tag", type=str, default=None, help="Base setting tag. With multiple generated sets, -gid<id> is appended.")
    parser.add_argument("--num_generated_sets", type=int, default=1)
    parser.add_argument("--generated_set_id", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="Optional override: UCM, AID, or NWPU. Inferred from path when possible.")
    parser.add_argument("--IPC", type=int, default=None, help="Optional image count per class. Inferred from IPC_<N> path when possible; otherwise uses all images.")
    parser.add_argument("--root_dir", type=str, default="/proj/cvl/users/x_xuyon/Data/VisionLanguage/")
    parser.add_argument("--pretrain_dir", type=str, default="./pretrain")
    parser.add_argument("--inception_ckpt", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--cache_real_stats", type=int, default=1)
    parser.add_argument("--recompute_real_stats", type=int, default=0)
    parser.add_argument("--real_stats", type=str, default=None)
    parser.add_argument("--stats_dir", type=str, default="./pretrain_stats")
    parser.add_argument("--output_dir", type=str, default="./results_fid", help="Default output folder for direct --distilled_dir evaluation.")
    parser.add_argument("--save_path_prefix", type=str, default="./results_fid", help="Default result root for --sd_root layout evaluation.")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the txt log instead of appending.")
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
