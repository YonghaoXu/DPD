import argparse
import os
import random
import time

import numpy as np
import torch
from diffusers import AutoencoderKL
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR, SequentialLR
from torch.utils import data
from torchvision import transforms

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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_lines(pathfile):
    with open(pathfile, 'r') as handle:
        return [line.strip() for line in handle if line.strip()]


def build_internal_split(lines, val_ratio, split_seed):
    per_class = {}
    for line in lines:
        label = int(line.split()[-1])
        per_class.setdefault(label, []).append(line)

    train_lines = []
    val_lines = []
    rng = random.Random(split_seed)

    for label in sorted(per_class):
        class_lines = list(per_class[label])
        rng.shuffle(class_lines)
        val_count = max(1, int(round(len(class_lines) * val_ratio)))
        val_count = min(val_count, len(class_lines) - 1)
        val_lines.extend(class_lines[:val_count])
        train_lines.extend(class_lines[val_count:])

    rng.shuffle(train_lines)
    rng.shuffle(val_lines)
    return train_lines, val_lines


def save_split_manifest(save_dir, train_lines, val_lines):
    split_dir = os.path.join(save_dir, 'splits')
    os.makedirs(split_dir, exist_ok=True)
    train_path = os.path.join(split_dir, 'internal_train.txt')
    val_path = os.path.join(split_dir, 'internal_val.txt')
    with open(train_path, 'w') as handle:
        handle.write('\n'.join(train_lines) + '\n')
    with open(val_path, 'w') as handle:
        handle.write('\n'.join(val_lines) + '\n')
    print(f"[INFO] Saved internal split files to {split_dir}")


def build_dataloader_kwargs(batch_size, shuffle, num_workers, prefetch_factor):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def parse_scheduler_milestones(spec, total_epochs):
    milestones = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value < 1.0:
            milestone = int(round(total_epochs * value))
        else:
            milestone = int(round(value))
        if 0 < milestone < total_epochs:
            milestones.append(milestone)
    return sorted(set(milestones))


def build_optimizer(model, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=bool(args.nesterov),
        )
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None, []
    if args.scheduler == "multistep":
        milestones = parse_scheduler_milestones(args.lr_decay_milestones, args.num_epochs)
        if not milestones:
            return None, []
        return MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay_gamma), milestones
    if args.scheduler == "cosine":
        warmup_epochs = max(0, int(args.warmup_epochs))
        warmup_epochs = min(warmup_epochs, args.num_epochs - 1)

        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.num_epochs - warmup_epochs,
                eta_min=args.min_lr,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
            return scheduler, [f"warmup={warmup_epochs}", f"min_lr={args.min_lr}"]

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=args.min_lr,
        )
        return scheduler, [f"min_lr={args.min_lr}"]
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def build_train_image_transform(args):
    return transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def build_val_image_transform(args):
    return transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, latents):
        identity = self.shortcut(latents)
        out = self.conv1(latents)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.relu(out)


class LatentClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.2, base_width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(base_width, base_width, stride=1),
            ResidualBlock(base_width, base_width, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(base_width, base_width * 2, stride=2),
            ResidualBlock(base_width * 2, base_width * 2, stride=1),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(base_width * 2, base_width * 4, stride=2),
            ResidualBlock(base_width * 4, base_width * 4, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(base_width * 4, num_classes)

    def forward_features(self, latents):
        latents = self.stem(latents)
        latents = self.layer1(latents)
        latents = self.layer2(latents)
        latents = self.layer3(latents)
        latents = self.pool(latents).flatten(1)
        return latents

    def forward(self, latents, return_features=False):
        features = self.forward_features(latents)
        logits = self.fc(self.dropout(features))
        if return_features:
            return logits, features
        return logits


def encode_images_to_latent_stats(vae, images):
    posterior = vae.encode(images.to(dtype=torch.float32)).latent_dist
    scaling_factor = vae.config.scaling_factor
    mean = posterior.mean * scaling_factor
    if hasattr(posterior, "std"):
        std = posterior.std * scaling_factor
    elif hasattr(posterior, "logvar"):
        std = torch.exp(0.5 * posterior.logvar) * scaling_factor
    else:
        raise AttributeError("VAE posterior does not expose std or logvar.")
    return mean, std


def sample_latents(mean, std, use_sampling):
    if use_sampling:
        return mean + std * torch.randn_like(std)
    return mean


def augment_latents(latents, hflip_prob, vflip_prob):
    batch_size = latents.shape[0]
    augmented = latents

    if hflip_prob > 0:
        mask = torch.rand(batch_size, device=latents.device) < hflip_prob
        if mask.any():
            augmented = augmented.clone()
            augmented[mask] = torch.flip(augmented[mask], dims=[3])

    if vflip_prob > 0:
        mask = torch.rand(batch_size, device=latents.device) < vflip_prob
        if mask.any():
            if augmented is latents:
                augmented = augmented.clone()
            augmented[mask] = torch.flip(augmented[mask], dims=[2])

    return augmented


def build_latent_ckpt_name(args):
    if args.save_name:
        return args.save_name
    return "latent_classifier.pth"


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def encode_dataset_to_latent_stats(vae, dataset, batch_size, num_workers, prefetch_factor, split_name):
    loader = data.DataLoader(dataset, **build_dataloader_kwargs(batch_size, False, num_workers, prefetch_factor))
    mean_chunks = []
    std_chunks = []
    label_chunks = []
    start = time.time()

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.cuda(non_blocking=True)
            mean, std = encode_images_to_latent_stats(vae, images)
            mean_chunks.append(mean.cpu())
            std_chunks.append(std.cpu())
            label_chunks.append(labels.clone())

    all_mean = torch.cat(mean_chunks, dim=0).contiguous()
    all_std = torch.cat(std_chunks, dim=0).contiguous()
    all_labels = torch.cat(label_chunks, dim=0).long()
    print(
        f"[INFO] Encoded {split_name} split to latent stats: N={all_mean.shape[0]}, "
        f"Shape={tuple(all_mean.shape[1:])}, Time={time.time() - start:.2f}s"
    )
    return all_mean, all_std, all_labels


def iterate_cached_latent_stat_batches(mean, std, labels, batch_size, shuffle):
    num_samples = labels.shape[0]
    if shuffle:
        indices = torch.randperm(num_samples, device=mean.device)
    else:
        indices = torch.arange(num_samples, device=mean.device)

    for start in range(0, num_samples, batch_size):
        batch_indices = indices[start:start + batch_size]
        yield mean[batch_indices], std[batch_indices], labels[batch_indices]


def iterate_cached_latent_batches(latents, labels, batch_size, shuffle):
    num_samples = labels.shape[0]
    if shuffle:
        indices = torch.randperm(num_samples, device=latents.device)
    else:
        indices = torch.arange(num_samples, device=latents.device)

    for start in range(0, num_samples, batch_size):
        batch_indices = indices[start:start + batch_size]
        yield latents[batch_indices], labels[batch_indices]


def evaluate_model(model, val_mean, val_labels, batch_size, classnames, split_name, report_per_class=False):
    model.eval()
    num_classes = len(classnames)
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]
    correct = 0
    total = 0

    with torch.no_grad():
        for mean, labels in iterate_cached_latent_batches(val_mean, val_labels, batch_size, shuffle=False):
            logits = model(mean)
            preds = torch.argmax(logits, dim=1)

            correct_mask = preds.eq(labels)
            correct += correct_mask.sum().item()
            total += labels.size(0)

            for idx in range(labels.size(0)):
                label = labels[idx].item()
                class_total[label] += 1
                class_correct[label] += correct_mask[idx].item()

    class_acc = []
    for class_idx in range(num_classes):
        acc = 0.0 if class_total[class_idx] == 0 else class_correct[class_idx] / class_total[class_idx]
        class_acc.append(acc)
        if report_per_class:
            print(f"[{class_idx:2d}] Accuracy of {classnames[class_idx]:<25}: {100.0 * acc:.2f}%")

    oa = correct / max(total, 1)
    aa = float(np.mean(class_acc))
    print(f"[INFO] {split_name} OA={oa * 100:.2f}%, AA={aa * 100:.2f}%")
    return oa, aa, class_acc


def train_once(model, vae, args, classnames, train_data, val_data, save_path):
    optimizer = build_optimizer(model, args)
    scheduler, scheduler_milestones = build_scheduler(optimizer, args)
    criterion = nn.CrossEntropyLoss().cuda()

    train_mode = 'cached-latents' if args.cache_train_latents else 'on-the-fly-vae'
    print(
        f"[INFO] Optimizer={args.optimizer}, LR={args.lr}, Epochs={args.num_epochs}, "
        f"Scheduler={args.scheduler}, Milestones={scheduler_milestones}, TrainMode={train_mode}, "
        f"TrainLatentSampling={bool(args.train_latent_sampling)}, "
        f"LatentHFlip={args.latent_hflip_prob}, LatentVFlip={args.latent_vflip_prob}, ValLatents=mean"
    )

    best_oa = -1.0
    best_epoch = -1

    for epoch in range(args.num_epochs):
        model.train()
        start = time.time()
        losses = []
        accs = []

        if args.cache_train_latents:
            train_mean, train_std, train_labels = train_data
            train_batches = iterate_cached_latent_stat_batches(
                train_mean,
                train_std,
                train_labels,
                args.train_batch_size,
                shuffle=True,
            )
        else:
            train_batches = train_data

        for batch in train_batches:
            if args.cache_train_latents:
                mean, std, labels = batch
                latents = sample_latents(mean, std, bool(args.train_latent_sampling))
            else:
                images, labels, _ = batch
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                with torch.no_grad():
                    mean, std = encode_images_to_latent_stats(vae, images)
                    latents = sample_latents(mean, std, bool(args.train_latent_sampling))

            latents = augment_latents(latents, args.latent_hflip_prob, args.latent_vflip_prob)
            logits = model(latents)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = preds.eq(labels).float().mean().item()
            losses.append(loss.item())
            accs.append(acc)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch + 1}/{args.num_epochs}] Loss={np.mean(losses):.4f}, "
            f"Acc={np.mean(accs) * 100:.2f}%, LR={current_lr:.6f}, Time={time.time() - start:.2f}s"
        )

        should_eval = (epoch + 1) == args.num_epochs or (epoch + 1) % args.eval_interval == 0
        if not should_eval:
            continue

        val_mean, val_labels = val_data
        oa, aa, _ = evaluate_model(
            model,
            val_mean,
            val_labels,
            args.val_batch_size,
            classnames,
            split_name='Internal Val',
            report_per_class=False,
        )
        if oa >= best_oa:
            best_oa = oa
            best_epoch = epoch + 1
            torch.save(unwrap_model(model).state_dict(), save_path)
            print(f"[INFO] Saved new best checkpoint to {save_path}")

    state_dict = torch.load(save_path, map_location="cpu")
    unwrap_model(model).load_state_dict(state_dict)
    val_mean, val_labels = val_data
    final_oa, final_aa, _ = evaluate_model(
        model,
        val_mean,
        val_labels,
        args.val_batch_size,
        classnames,
        split_name='Internal Val',
        report_per_class=True,
    )
    print(f"[DONE] Best Epoch={best_epoch}, Final Best Internal-Val OA={final_oa * 100:.2f}%, AA={final_aa * 100:.2f}%")
    return final_oa


def main(args):
    torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)

    args.dataset = args.dataset.upper()
    if args.dataset not in DATASET_META:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    num_classes, classnames = DATASET_META[args.dataset]
    save_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, build_latent_ckpt_name(args))

    full_train_lines = load_lines(f"./dataset/{args.dataset}_train.txt")
    train_lines, val_lines = build_internal_split(full_train_lines, args.val_ratio, args.split_seed)
    save_split_manifest(save_dir, train_lines, val_lines)

    train_dataset = scene_dataset(root_dir=args.root_dir, lines=train_lines, transform=build_train_image_transform(args))
    val_dataset = scene_dataset(root_dir=args.root_dir, lines=val_lines, transform=build_val_image_transform(args))

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')
    vae.requires_grad_(False)
    vae.eval()
    vae = vae.cuda()

    if args.cache_train_latents:
        train_mean, train_std, train_labels = encode_dataset_to_latent_stats(
            vae, train_dataset, args.encode_batch_size, args.num_workers, args.prefetch_factor, 'internal-train'
        )
        train_mean = train_mean.cuda(non_blocking=True)
        train_std = train_std.cuda(non_blocking=True)
        train_labels = train_labels.cuda(non_blocking=True)
        train_data = (train_mean, train_std, train_labels)
        print(f"[INFO] Cached train latent stats on GPU: N={train_mean.shape[0]}, Batch={args.train_batch_size}")
    else:
        train_data = data.DataLoader(train_dataset, **build_dataloader_kwargs(args.train_batch_size, True, args.num_workers, args.prefetch_factor))
        print(f"[INFO] Train split uses on-the-fly VAE encoding with deterministic resize: N={len(train_dataset)}")

    val_mean, _, val_labels = encode_dataset_to_latent_stats(
        vae, val_dataset, args.encode_batch_size, args.num_workers, args.prefetch_factor, 'internal-val'
    )
    val_mean = val_mean.cuda(non_blocking=True)
    val_labels = val_labels.cuda(non_blocking=True)
    val_data = (val_mean, val_labels)
    print(f"[INFO] Cached val latent mean on GPU: N={val_mean.shape[0]}, Batch={args.val_batch_size} | Eval uses mean")

    if args.cache_train_latents:
        del vae
        torch.cuda.empty_cache()
        vae_for_train = None
    else:
        vae_for_train = vae

    print(f"[INFO] Dataset={args.dataset} | InternalTrain={len(train_dataset)} | InternalVal={len(val_dataset)} | Resolution={args.resolution} | Save={save_path}")
    model = LatentClassifier(num_classes, dropout=args.dropout).cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_once(model, vae_for_train, args, classnames, train_data, val_data, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='UCM')
    parser.add_argument('--root_dir', type=str, default='/proj/cvl/users/x_xuyon/Data/VisionLanguage/')
    parser.add_argument('--output_dir', type=str, default='./pretrain')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='lcybuaa/Text2Earth')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--encode_batch_size', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--nesterov', type=int, default=0)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'multistep', 'cosine'])
    parser.add_argument('--lr_decay_milestones', type=str, default='0.6667,0.8333')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--split_seed', type=int, default=666)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cache_train_latents', type=int, default=1)
    parser.add_argument('--train_latent_sampling', type=int, default=1)
    parser.add_argument('--latent_hflip_prob', type=float, default=0.5)
    parser.add_argument('--latent_vflip_prob', type=float, default=0.5)
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args()
    main(args)
