import argparse
import csv
import os
import re
from glob import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm
from dataset.scene_dataset import scene_dataset


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


class ImageClassDataset(Dataset):
    def __init__(self, samples, transform=None, loader=default_loader):
        self.samples = list(samples)
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name = self.samples[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path, label, class_name


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


def infer_method_from_path(path, dataset=None):
    parts = os.path.normpath(path).split(os.sep)
    upper_parts = [part.upper() for part in parts]
    dataset = (dataset or infer_dataset_from_path(path) or "").upper()

    if dataset in DATASET_META and dataset in upper_parts:
        dataset_index = upper_parts.index(dataset)
        if dataset_index > 0:
            return parts[dataset_index - 1]

    ipc_index = None
    for index in range(len(parts) - 1, -1, -1):
        if re.fullmatch(r"IPC_\d+", parts[index], flags=re.IGNORECASE):
            ipc_index = index
            break
    if ipc_index is not None:
        for index in range(ipc_index - 1, -1, -1):
            if upper_parts[index] in DATASET_META:
                continue
            if parts[index].lower() in {"template", "prototype", "full"}:
                continue
            return parts[index]

    return None


def build_generated_setting_labels(setting_label, num_generated_sets, generated_set_id=None):
    if generated_set_id is not None:
        return [f"{setting_label}-gid{generated_set_id}"]
    if num_generated_sets <= 1:
        return [str(setting_label)]
    return [f"{setting_label}-gid{i}" for i in range(1, num_generated_sets + 1)]


def build_distilled_dirs(args, dataset):
    if args.distilled_dir is not None:
        distilled_dir = os.path.abspath(args.distilled_dir)
        ipc = args.IPC if args.IPC is not None else infer_ipc_from_path(distilled_dir)
        return ipc, [(None, distilled_dir)]

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
    return args.IPC, distilled_dirs


def resolve_datasets(args):
    if args.distilled_dir is not None:
        distilled_dir = os.path.abspath(args.distilled_dir)
        dataset = (args.dataset or infer_dataset_from_path(distilled_dir) or "").upper()
        if dataset not in DATASET_META:
            raise ValueError("Could not infer dataset from --distilled_dir. Pass --dataset UCM, AID, or NWPU.")
        return [dataset]

    dataset = (args.dataset or "").upper()
    if dataset == "ALL":
        return list(DATASET_META.keys())
    if dataset not in DATASET_META:
        raise ValueError("Pass --dataset UCM, AID, NWPU, or ALL when --distilled_dir is not used.")
    return [dataset]


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


def collect_distilled_class_samples(distilled_dir, dataset, ipc=None):
    if not os.path.isdir(distilled_dir):
        raise FileNotFoundError(f"Distilled directory not found: {distilled_dir}")

    classnames = DATASET_META[dataset]
    samples_by_class = {}
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
        if not selected_paths:
            raise ValueError(f"Class {label}_{class_name} has no distilled images.")
        samples_by_class[label] = [(path, label, class_name) for path in selected_paths]

    return samples_by_class


def collect_real_train_class_samples(root_dir, dataset):
    classnames = DATASET_META[dataset]
    real_dataset = scene_dataset(
        root_dir=root_dir,
        pathfile=os.path.join("dataset", f"{dataset}_train.txt"),
        transform=None,
    )
    samples_by_class = {label: [] for label in range(len(classnames))}

    for image_path, label, _ in real_dataset.imgs:
        label = int(label)
        if label not in samples_by_class:
            raise ValueError(f"Unexpected label {label} in dataset/{dataset}_train.txt")
        samples_by_class[label].append((image_path, label, classnames[label]))

    empty_classes = [
        f"{label}_{classnames[label]}"
        for label, samples in samples_by_class.items()
        if not samples
    ]
    if empty_classes:
        raise ValueError(f"Real train split has empty classes: {', '.join(empty_classes)}")
    return samples_by_class


def flatten_samples(samples_by_class):
    samples = []
    for label in sorted(samples_by_class):
        samples.extend(samples_by_class[label])
    return samples


def normalize_features(features):
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class ResNet18FeatureExtractor:
    def __init__(self, weights_name, device):
        self.weights = getattr(ResNet18_Weights, weights_name)
        model = resnet18(weights=self.weights)
        model.fc = nn.Identity()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to(device)
        model.eval()

        self.model = model
        self.device = device
        self.transform = self.weights.transforms()

    def forward_batch(self, images):
        with torch.no_grad():
            features = self.model(images)
        return normalize_features(features)


def extract_features_by_class(feature_extractor, samples_by_class, args, desc):
    samples = flatten_samples(samples_by_class)
    loader = data.DataLoader(
        ImageClassDataset(samples, transform=feature_extractor.transform),
        **build_dataloader_kwargs(args.batch_size, False, args.num_workers, args.prefetch_factor),
    )
    features_by_class = {label: [] for label in samples_by_class}

    with torch.no_grad():
        for images, _, labels, _ in tqdm(loader, desc=desc):
            images = images.to(feature_extractor.device, non_blocking=True)
            features = feature_extractor.forward_batch(images).detach().cpu()
            for feature, label in zip(features, labels):
                features_by_class[int(label)].append(feature)

    return {
        label: normalize_features(torch.stack(features, dim=0))
        for label, features in features_by_class.items()
    }


def compute_class_coverage(real_features, distilled_features, chunk_size=2048):
    if real_features.shape[0] == 0:
        raise ValueError("Coverage needs at least one real image in each class.")
    if distilled_features.shape[0] == 0:
        raise ValueError("Coverage needs at least one distilled image in each class.")

    max_similarities = []
    distilled_features_t = distilled_features.t()
    for start in range(0, real_features.shape[0], chunk_size):
        real_chunk = real_features[start:start + chunk_size]
        similarity = real_chunk @ distilled_features_t
        max_similarities.append(similarity.max(dim=1).values)

    q_values = torch.cat(max_similarities, dim=0).clamp(-1.0, 1.0)
    return float(q_values.mean().item())


def append_summary_csv(path, row):
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "method", "dataset", "ipc", "generated_set", "coverage",
        "real_images", "distilled_images", "distilled_dir",
        "feature_extractor", "weights",
    ]
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def evaluate_distilled_dir(feature_extractor, real_features_by_class, real_samples_by_class,
                           distilled_dir, dataset, ipc, args):
    distilled_samples_by_class = collect_distilled_class_samples(distilled_dir, dataset, ipc=ipc)
    distilled_features_by_class = extract_features_by_class(
        feature_extractor,
        distilled_samples_by_class,
        args,
        desc="Extracting distilled ResNet18 features",
    )

    class_results = []
    classnames = DATASET_META[dataset]
    for label, class_name in enumerate(classnames):
        coverage = compute_class_coverage(
            real_features_by_class[label],
            distilled_features_by_class[label],
            chunk_size=args.similarity_chunk_size,
        )
        class_results.append({
            "label": label,
            "class_name": class_name,
            "real_images": len(real_samples_by_class[label]),
            "distilled_images": len(distilled_samples_by_class[label]),
            "coverage": coverage,
        })

    macro_coverage = float(np.mean([result["coverage"] for result in class_results]))
    return class_results, macro_coverage


def build_csv_row(args, dataset, ipc_text, generated_set, distilled_dir, class_results, macro_coverage):
    method = args.method or infer_method_from_path(distilled_dir, dataset) or ""
    return {
        "method": method,
        "dataset": dataset,
        "ipc": ipc_text,
        "generated_set": "" if generated_set is None else generated_set,
        "coverage": f"{macro_coverage:.6f}",
        "real_images": sum(result["real_images"] for result in class_results),
        "distilled_images": sum(result["distilled_images"] for result in class_results),
        "distilled_dir": distilled_dir,
        "feature_extractor": "ResNet18",
        "weights": args.weights,
    }

def main(args):
    datasets = resolve_datasets(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    feature_extractor = ResNet18FeatureExtractor(args.weights, device)
    all_coverages = []

    print(f"[INFO] Device={device}, FeatureExtractor=ResNet18, Weights={args.weights}")
    print("[INFO] Using torchvision official preprocessing from the selected ResNet18 weights.")

    for dataset in datasets:
        ipc, distilled_dirs = build_distilled_dirs(args, dataset)
        ipc_text = "all" if ipc is None else str(ipc)

        real_samples_by_class = collect_real_train_class_samples(args.root_dir, dataset)
        print(f"[INFO] Extracting real-train features from dataset/{dataset}_train.txt")
        real_features_by_class = extract_features_by_class(
            feature_extractor,
            real_samples_by_class,
            args,
            desc=f"Extracting {dataset} real ResNet18 features",
        )

        for generated_set, distilled_dir in distilled_dirs:
            method = args.method or infer_method_from_path(distilled_dir, dataset) or ""
            class_results, macro_coverage = evaluate_distilled_dir(
                feature_extractor,
                real_features_by_class,
                real_samples_by_class,
                distilled_dir,
                dataset,
                ipc,
                args,
            )
            all_coverages.append(macro_coverage)

            generated_set_text = "" if generated_set is None else f", Generated_Set={generated_set}"
            print(
                f"Method={method}, Dataset={dataset}{generated_set_text}, "
                f"IPC={ipc_text}, Distilled_Dir={distilled_dir}"
            )
            for result in class_results:
                print(
                    f"Class={result['label']}_{result['class_name']}, "
                    f"RealImages={result['real_images']}, "
                    f"DistilledImages={result['distilled_images']}, "
                    f"Coverage={result['coverage']:.6f}"
                )
            print(
                f"Method={method}, Dataset={dataset}{generated_set_text}, IPC={ipc_text}, "
                f"MacroCoverage={macro_coverage:.6f}, FeatureExtractor=ResNet18, "
                f"Weights={args.weights}"
            )

            if args.csv_file is not None:
                append_summary_csv(
                    args.csv_file,
                    build_csv_row(
                        args,
                        dataset,
                        ipc_text,
                        generated_set,
                        distilled_dir,
                        class_results,
                        macro_coverage,
                    ),
                )

    if len(all_coverages) > 1:
        print(
            f"Evaluated_Sets={len(all_coverages)}, "
            f"MeanMacroCoverage={np.mean(all_coverages):.6f}, "
            f"StdMacroCoverage={np.std(all_coverages):.6f}, "
            f"FeatureExtractor=ResNet18, Weights={args.weights}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=str, default=None, help="Path to one IPC directory containing class subfolders.")
    parser.add_argument("--sd_root", type=str, default=None, help="Root of generated distilled data, used with --dataset/--mode/--setting_tag/--IPC.")
    parser.add_argument("--mode", type=str, default="template", help="Distilled data mode folder, e.g. template, prototype, or full.")
    parser.add_argument("--setting_tag", type=str, default=None, help="Base setting tag. With multiple generated sets, -gid<id> is appended.")
    parser.add_argument("--num_generated_sets", type=int, default=1)
    parser.add_argument("--generated_set_id", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="UCM, AID, NWPU, or ALL. Inferred from --distilled_dir when possible.")
    parser.add_argument("--IPC", type=int, default=20, help="Optional image count per class. Inferred from IPC_<N> path when possible; otherwise uses all images.")
    parser.add_argument("--root_dir", type=str, default="/proj/cvl/users/x_xuyon/Data/VisionLanguage/")
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V1", choices=[weight.name for weight in ResNet18_Weights])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--similarity_chunk_size", type=int, default=2048)
    parser.add_argument("--csv_file", type=str, default=None, help="Optional CSV file for appending one macro coverage row per evaluated distilled directory.")
    parser.add_argument("--method", type=str, default=None, help="Optional method name written to --csv_file. If omitted, inferred from --distilled_dir when possible.")
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
