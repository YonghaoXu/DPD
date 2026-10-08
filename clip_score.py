import argparse
import os
import re
from glob import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DATASET_META = {
    "UCM": (
        (
            "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
            "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
            "harbor", "intersection", "mediumresidential", "mobilehomepark",
            "overpass", "parkinglot", "river", "runway", "sparseresidential",
            "storagetanks", "tenniscourt",
        ),
        {
            "agricultural": "agricultural area", "airplane": "airplane",
            "baseballdiamond": "baseball diamond", "beach": "beach",
            "buildings": "buildings", "chaparral": "chaparral",
            "denseresidential": "dense residential area", "forest": "forest",
            "freeway": "freeway", "golfcourse": "golf course",
            "harbor": "harbor", "intersection": "intersection",
            "mediumresidential": "medium residential area",
            "mobilehomepark": "mobile home park", "overpass": "overpass",
            "parkinglot": "parking lot", "river": "river", "runway": "runway",
            "sparseresidential": "sparse residential area",
            "storagetanks": "storage tanks", "tenniscourt": "tennis court",
        },
    ),
    "AID": (
        (
            "airport", "bareland", "baseballfield", "beach", "bridge", "center",
            "church", "commercial", "denseresidential", "desert", "farmland",
            "forest", "industrial", "meadow", "mediumresidential", "mountain",
            "park", "parking", "playground", "pond", "port", "railwaystation",
            "resort", "river", "school", "sparseresidential", "square", "stadium",
            "storagetanks", "viaduct",
        ),
        {
            "airport": "airport", "bareland": "bare land",
            "baseballfield": "baseball field", "beach": "beach",
            "bridge": "bridge", "center": "city center", "church": "church",
            "commercial": "commercial area", "denseresidential": "dense residential area",
            "desert": "desert", "farmland": "farmland", "forest": "forest",
            "industrial": "industrial area", "meadow": "meadow",
            "mediumresidential": "medium residential area", "mountain": "mountain",
            "park": "park", "parking": "parking lot", "playground": "playground",
            "pond": "pond", "port": "port", "railwaystation": "railway station",
            "resort": "resort", "river": "river", "school": "school",
            "sparseresidential": "sparse residential area", "square": "city square",
            "stadium": "stadium", "storagetanks": "storage tanks", "viaduct": "viaduct",
        },
    ),
    "NWPU": (
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
        {
            "airplane": "airplane", "airport": "airport",
            "baseball_diamond": "baseball diamond",
            "basketball_court": "basketball court", "beach": "beach",
            "bridge": "bridge", "chaparral": "chaparral", "church": "church",
            "circular_farmland": "circular farmland", "cloud": "cloud",
            "commercial_area": "commercial area",
            "dense_residential": "dense residential area", "desert": "desert",
            "forest": "forest", "freeway": "freeway", "golf_course": "golf course",
            "ground_track_field": "ground track field", "harbor": "harbor",
            "industrial_area": "industrial area", "intersection": "intersection",
            "island": "island", "lake": "lake", "meadow": "meadow",
            "medium_residential": "medium residential area",
            "mobile_home_park": "mobile home park", "mountain": "mountain",
            "overpass": "overpass", "palace": "palace", "parking_lot": "parking lot",
            "railway": "railway", "railway_station": "railway station",
            "rectangular_farmland": "rectangular farmland", "river": "river",
            "roundabout": "roundabout", "runway": "runway", "sea_ice": "sea ice",
            "ship": "ship", "snowberg": "snowberg",
            "sparse_residential": "sparse residential area", "stadium": "stadium",
            "storage_tank": "storage tank", "tennis_court": "tennis court",
            "terrace": "terrace", "thermal_power_station": "thermal power station",
            "wetland": "wetland",
        },
    ),
}


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def default_loader(path):
    return Image.open(path).convert("RGB")


class DistilledCLIPDataset(Dataset):
    def __init__(self, samples, loader=default_loader):
        self.samples = list(samples)
        self.loader = loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, text, label, class_name = self.samples[index]
        return self.loader(image_path), text, image_path, label, class_name


def collate_clip_batch(batch):
    images, texts, image_paths, labels, class_names = zip(*batch)
    return list(images), list(texts), list(image_paths), list(labels), list(class_names)


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


def build_prompt(readable_name, prompt_template):
    return prompt_template.format(readable_name=readable_name)


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


def collect_samples(distilled_dir, dataset, ipc=None, prompt_template="a remote sensing image of {readable_name}"):
    classnames, readable_mapping = DATASET_META[dataset]
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

        text = build_prompt(readable_mapping[class_name], prompt_template)
        selected_paths = image_paths[:ipc] if ipc is not None else image_paths
        for image_path in selected_paths:
            samples.append((image_path, text, label, class_name))

    if not samples:
        raise ValueError(f"No images found in distilled directory: {distilled_dir}")
    return samples


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


def make_loader(distilled_dir, dataset, ipc, prompt_template, batch_size, num_workers):
    samples = collect_samples(distilled_dir, dataset, ipc=ipc, prompt_template=prompt_template)
    clip_dataset = DistilledCLIPDataset(samples)
    return DataLoader(
        clip_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_clip_batch,
    )


def normalize_features(features):
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def encode_class_text_features(model, processor, class_prompts, device):
    text_inputs = processor(
        text=class_prompts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    text_features = model.get_text_features(text_inputs)
    return normalize_features(text_features)


def encode_image_features(model, processor, images, device):
    image_inputs = processor(images=images, return_tensors="pt").pixel_values.to(device)
    image_features = model.get_image_features(image_inputs)
    return normalize_features(image_features)


def compute_clip_metrics(model, processor, loader, device, class_prompts):
    clip_scores = []
    top1_hits = []

    with torch.no_grad():
        class_text_features = encode_class_text_features(model, processor, class_prompts, device)
        for images, _, _, labels, _ in tqdm(loader, desc="CLIP-RSICD-v2 score / zero-shot"):
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            image_features = encode_image_features(model, processor, images, device)
            similarity = image_features @ class_text_features.t()

            matched_scores = similarity.gather(1, labels.view(-1, 1)).squeeze(1)
            predictions = torch.argmax(similarity, dim=1)
            clip_scores.extend(matched_scores.detach().cpu().tolist())
            top1_hits.extend(predictions.eq(labels).float().detach().cpu().tolist())

    return {
        "clip_score": float(np.mean(clip_scores)),
        "top1": float(np.mean(top1_hits)),
        "num_images": len(top1_hits),
    }


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
                "clip_score_log.txt",
            )
        )

    safe_name = os.path.basename(os.path.normpath(args.distilled_dir)) or "distilled"
    return os.path.abspath(os.path.join(args.output_dir, f"{dataset}_{safe_name}_clip_score_log.txt"))


def main(args):
    dataset, ipc, distilled_dirs = build_distilled_dirs(args)
    classnames, readable_mapping = DATASET_META[dataset]
    class_prompts = [build_prompt(readable_mapping[class_name], args.prompt_template) for class_name in classnames]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    model.eval()

    result_lines = []
    clip_score_values = []
    top1_values = []
    ipc_text = "all" if ipc is None else str(ipc)

    for generated_set, distilled_dir in distilled_dirs:
        if not os.path.isdir(distilled_dir):
            raise FileNotFoundError(f"Distilled directory not found: {distilled_dir}")

        clip_loader = make_loader(
            distilled_dir,
            dataset,
            ipc,
            args.prompt_template,
            args.batch_size,
            args.num_workers,
        )
        metrics = compute_clip_metrics(model, processor, clip_loader, device, class_prompts)
        clip_score_values.append(metrics["clip_score"])
        top1_values.append(metrics["top1"])

        generated_set_text = "" if generated_set is None else f", Generated_Set={generated_set}"
        line = (
            f"Dataset={dataset}{generated_set_text}, IPC={ipc_text}, "
            f"Images={metrics['num_images']}, CLIPScore={metrics['clip_score']:.6f}, "
            f"Top1={metrics['top1'] * 100:.2f}%, "
            f"Distilled_Dir={distilled_dir}, CLIPModel={args.clip_model}, "
            f"PromptTemplate={args.prompt_template}"
        )
        print(line)
        result_lines.append(line)

    if len(top1_values) > 1:
        summary_line = (
            f"Dataset={dataset}, Mode={args.mode.upper()}, Setting={args.setting_tag}, IPC={ipc_text}, "
            f"Generated_Sets={len(top1_values)}, MeanCLIPScore={np.mean(clip_score_values):.6f}, "
            f"StdCLIPScore={np.std(clip_score_values):.6f}, MeanTop1={np.mean(top1_values) * 100:.2f}%, "
            f"StdTop1={np.std(top1_values) * 100:.2f}%, CLIPModel={args.clip_model}, "
            f"PromptTemplate={args.prompt_template}"
        )
        print(summary_line)
        result_lines.append(summary_line)

    log_file = resolve_log_file(args, dataset, ipc)
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    write_mode = "w" if args.overwrite else "a"
    with open(log_file, write_mode, encoding="utf-8") as handle:
        for line in result_lines:
            handle.write(line + "\n")
    print(f"[INFO] Saved CLIP-RSICD-v2 score and zero-shot top-1 results to {log_file}")


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
    parser.add_argument("--clip_model", type=str, default="flax-community/clip-rsicd-v2")
    parser.add_argument("--prompt_template", type=str, default="a remote sensing image of {readable_name}")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./clip_score", help="Default output folder for direct --distilled_dir evaluation.")
    parser.add_argument("--save_path_prefix", type=str, default="./clip_score", help="Default result root for --sd_root layout evaluation.")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the txt log instead of appending.")
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
