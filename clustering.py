import os
import re
import json
import argparse
from collections import defaultdict
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import AutoencoderKL


def match_class_from_filename(filename, class_list, dataset):
    filename = filename.lower()
    class_list = sorted(class_list, key=len, reverse=True)
    for class_name in class_list:
        if dataset == "UCM":
            if filename.startswith(class_name):
                return class_name
        else:
            pattern = re.compile(rf"^{re.escape(class_name)}[\W_]")
            if pattern.match(filename):
                return class_name
    return None


def build_filename_to_path_index(txt_file_path):
    filename_to_fullpath = {}
    with open(txt_file_path, "r") as f:
        for line in f:
            rel_path = line.strip().split()[0]
            file_name = os.path.basename(rel_path)
            filename_to_fullpath[file_name] = rel_path
    return filename_to_fullpath


def get_class_list(dataset):
    dataset_classes = {
        "UCM": (
            "agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral",
            "denseresidential", "forest", "freeway", "golfcourse", "harbor", "intersection",
            "mediumresidential", "mobilehomepark", "overpass", "parkinglot", "river", "runway",
            "sparseresidential", "storagetanks", "tenniscourt",
        ),
        "AID": (
            "airport", "bareland", "baseballfield", "beach", "bridge", "center", "church", "commercial",
            "denseresidential", "desert", "farmland", "forest", "industrial", "meadow", "mediumresidential",
            "mountain", "park", "parking", "playground", "pond", "port", "railwaystation", "resort", "river",
            "school", "sparseresidential", "square", "stadium", "storagetanks", "viaduct",
        ),
        "NWPU": (
            "airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge", "chaparral",
            "church", "circular_farmland", "cloud", "commercial_area", "dense_residential", "desert",
            "forest", "freeway", "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection",
            "island", "lake", "meadow", "medium_residential", "mobile_home_park", "mountain", "overpass", "palace",
            "parking_lot", "railway", "railway_station", "rectangular_farmland", "river", "roundabout", "runway",
            "sea_ice", "ship", "snowberg", "sparse_residential", "stadium", "storage_tank", "tennis_court",
            "terrace", "thermal_power_station", "wetland",
        ),
    }
    return dataset_classes[dataset]


def get_train_split_file(dataset):
    dataset = dataset.upper()
    if dataset == "UCM":
        return "./dataset/UCM_train.txt"
    if dataset == "AID":
        return "./dataset/AID_train.txt"
    if dataset == "NWPU":
        return "./dataset/NWPU_train.txt"
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_preprocess(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def load_image_tensor(image_path, preprocess):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image)


def encode_paths_with_vae(vae, image_paths, preprocess, batch_size, device, latent_pool_size):
    latents = []
    valid_paths = []
    valid_names = []

    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start:start + batch_size]
        batch_tensors = []
        batch_names = []
        batch_paths = []
        for fname, image_path in chunk:
            try:
                batch_tensors.append(load_image_tensor(image_path, preprocess))
                batch_names.append(fname)
                batch_paths.append(image_path)
            except Exception as exc:
                print(f"[Error] Failed to load {image_path}: {exc}")

        if not batch_tensors:
            continue

        images = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            latent = vae.encode(images).latent_dist.mean * 0.18215
            if latent_pool_size > 0:
                latent = F.adaptive_avg_pool2d(latent, (latent_pool_size, latent_pool_size))
        latent = latent.flatten(1).cpu().numpy()

        latents.append(latent)
        valid_paths.extend(batch_paths)
        valid_names.extend(batch_names)

    if not latents:
        return None, [], []

    return np.concatenate(latents, axis=0), valid_names, valid_paths


def build_class_entries(dataset, root_dir):
    class_list = get_class_list(dataset)
    file_to_path = build_filename_to_path_index(get_train_split_file(dataset))
    class_to_paths = defaultdict(list)

    for fname, rel_path in file_to_path.items():
        matched_class = match_class_from_filename(fname, class_list, dataset)
        if matched_class:
            full_path = os.path.join(root_dir, rel_path)
            class_to_paths[matched_class].append((fname, full_path))

    return class_list, class_to_paths


def select_representative(latents, centers, labels, cluster_id, alpha):
    indices = np.where(labels == cluster_id)[0]
    if len(indices) == 0:
        return None

    own_center = centers[cluster_id]
    other_centers = [centers[j] for j in range(len(centers)) if j != cluster_id]
    if not other_centers:
        other_centers = [own_center]

    best = None
    for idx in indices:
        z = latents[idx]
        d_intra = np.linalg.norm(z - own_center)
        d_inter_min = min(np.linalg.norm(z - c) for c in other_centers)
        score = d_inter_min - alpha * d_intra
        candidate = {
            "index": int(idx),
            "margin": float(score),
            "d_intra": float(d_intra),
            "d_inter_min": float(d_inter_min),
            "cluster_size": int(len(indices)),
        }
        if best is None or candidate["margin"] > best["margin"]:
            best = candidate
    return best


def extract_ipc_prototypes(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    vae = AutoencoderKL.from_pretrained(args.vae_model, subfolder="vae").to(device).eval()
    preprocess = build_preprocess(args.vae_resolution)

    class_list, class_to_paths = build_class_entries(args.dataset, args.root_dir)

    pairs_json = {}
    debug_info = {}

    for class_name in tqdm(class_list, desc="Processing classes"):
        entries = class_to_paths.get(class_name, [])
        if len(entries) < args.n_clusters:
            print(f"[Warning] Class '{class_name}' has too few samples ({len(entries)}) for K={args.n_clusters}, skipping.")
            continue

        latents, valid_names, valid_paths = encode_paths_with_vae(
            vae=vae,
            image_paths=entries,
            preprocess=preprocess,
            batch_size=args.batch_size,
            device=device,
            latent_pool_size=args.latent_pool_size,
        )
        if latents is None or len(latents) < args.n_clusters:
            print(f"[Warning] Class '{class_name}': not enough valid latents, skip.")
            continue

        latents_cluster = normalize(latents, norm="l2") if args.normalize_latents else latents

        kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=args.kmeans_n_init).fit(latents_cluster)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        class_pairs = []
        dbg_lines = []

        for cluster_id in range(args.n_clusters):
            best = select_representative(latents_cluster, centers, labels, cluster_id, args.alpha)
            if best is None:
                continue

            idx = best["index"]
            rep_img_path = valid_paths[idx]
            rep_name = valid_names[idx]
            class_pairs.append({
                "image": rep_img_path,
                "caption": "",
                "cluster_id": int(cluster_id),
                "file_name": rep_name,
                "margin": round(best["margin"], 6),
                "d_intra": round(best["d_intra"], 6),
                "d_inter_min": round(best["d_inter_min"], 6),
                "cluster_size": int(best["cluster_size"]),
            })

            dbg_lines.append(
                f"Cluster {cluster_id} | size={best['cluster_size']} | rep={rep_name} | "
                f"margin={best['margin']:.6f} | d_intra(rep)={best['d_intra']:.6f} | d_inter_min(rep)={best['d_inter_min']:.6f}"
            )

        pairs_json[class_name] = class_pairs
        debug_info[class_name] = dbg_lines

    return pairs_json, debug_info


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {path}")


def save_debug(debug_info, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cls, lines in debug_info.items():
            f.write(f"## {cls}\n")
            for ln in lines:
                f.write(ln + "\n")
            f.write("\n")
    print(f"[Saved] {path}")


def main(args):
    pairs_json, debug_info = extract_ipc_prototypes(args)

    k_tag = f"n_cluster_{args.n_clusters}"
    out_dir = args.output_root
    os.makedirs(out_dir, exist_ok=True)

    pairs_out = os.path.join(out_dir, f"prototype_pairs_{args.dataset}_{k_tag}.json")
    debug_out = os.path.join(out_dir, f"debug_proto_{args.dataset}_{k_tag}.txt")

    save_json(pairs_json, pairs_out)
    save_debug(debug_info, debug_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="UCM")
    parser.add_argument("--root_dir", type=str, default="/proj/cvl/users/x_xuyon/Data/VisionLanguage/")
    parser.add_argument("--output_root", type=str, default="./prototypes/")
    parser.add_argument("--n_clusters", type=int, default=5, help="IPC = number of clusters per class")
    parser.add_argument("--alpha", type=float, default=1.0, help="margin weight: score = min_inter - alpha * intra")
    parser.add_argument("--vae_model", type=str, default="lcybuaa/Text2Earth")
    parser.add_argument("--vae_resolution", type=int, default=256)
    parser.add_argument("--latent_pool_size", type=int, default=8, help="Spatially pool VAE latents to latent_pool_size x latent_pool_size before flattening; <=0 disables pooling.")
    parser.add_argument("--no_normalize_latents", action="store_false", dest="normalize_latents", help="Disable L2 normalization before KMeans.")
    parser.set_defaults(normalize_latents=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--kmeans_n_init", type=int, default=10)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--device", type=str, default=None)
    main(parser.parse_args())