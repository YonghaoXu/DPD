import argparse
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
from torchvision import transforms
import tools.model as models
from dataset.scene_dataset import scene_dataset
from tools.utils import test_acc

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SUPPORTED_NETWORKS = ("vgg16", "inception_v3", "resnet18", "densenet121")

def build_model(name, num_classes, pretrained=False):
    name = name.lower()
    if name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(1024, num_classes)
    else:
        raise ValueError(f"[ERROR] Unsupported network: {name}. Choose from {', '.join(SUPPORTED_NETWORKS)}")

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


# ---------------------------------------------------------
# Generic SD distilled loader
# Path:
# {sd_root}/{DATASET}/{MODE}/{SETTING}/IPC_{IPC}/{label}_{classname}/*.png
# ---------------------------------------------------------
def load_sd_distilled(sd_root, mode, dataset_name, weight, IPC, classname, setting_tag=None):
    setting_label = setting_tag if setting_tag is not None else weight
    root = os.path.join(sd_root, dataset_name, mode, str(setting_label), f"IPC_{IPC}")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"[SD] Base directory not found: {root}")

    result = []
    for label, cls_name in enumerate(classname):
        cls_dir = os.path.join(root, f"{label}_{cls_name}")
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"[SD] Class dir missing: {cls_dir}")

        images = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith('.png')])
        if len(images) < IPC:
            raise ValueError(f"[SD] Class {cls_name} has {len(images)} images, need {IPC}")

        for img_name in images[:IPC]:
            img_path = os.path.abspath(os.path.join(cls_dir, img_name))
            result.append(f"{img_path} {label}")

    return result

def build_generated_setting_labels(setting_label, args):
    if args.generated_set_id is not None:
        return [f"{setting_label}-gid{args.generated_set_id}"]
    if args.num_generated_sets <= 1:
        return [str(setting_label)]
    return [f"{setting_label}-gid{i}" for i in range(1, args.num_generated_sets + 1)]

def build_base_train_transform(args):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        normalize,
    ])

def build_full_train_transform(args):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return transforms.Compose([
        transforms.RandomResizedCrop(
            args.resolution,
            scale=(args.train_crop_min, 1.0),
            ratio=(args.train_aspect_min, args.train_aspect_max),
        ),
        transforms.RandomHorizontalFlip(p=args.hflip_prob),
        transforms.RandomVerticalFlip(p=args.vflip_prob),
        transforms.ToTensor(),
        normalize,
    ])

def build_val_transform(args):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        normalize,
    ])

def build_optimizer(model, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=bool(args.nesterov),
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")

def parse_scheduler_milestones(spec, total_iterations):
    milestones = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value < 1.0:
            milestone = int(round(total_iterations * value))
        else:
            milestone = int(round(value))
        if 0 < milestone < total_iterations:
            milestones.append(milestone)
    return sorted(set(milestones))

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

def build_scheduler(optimizer, args, total_iterations):
    if args.scheduler == "none":
        return None, []
    if args.scheduler == "multistep":
        milestones = parse_scheduler_milestones(args.lr_decay_milestones, total_iterations)
        if not milestones:
            return None, []
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay_gamma)
        return scheduler, milestones
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")

def resolve_cache_device(cache_device):
    if cache_device == "gpu" and not torch.cuda.is_available():
        return "cpu"
    return cache_device

def preload_train_cache(lines, args):
    cache_device = resolve_cache_device(args.train_cache_device)
    dataset = scene_dataset(lines=lines, root_dir=args.root_dir, transform=build_base_train_transform(args))
    images = []
    labels = []
    for image, label, _ in dataset:
        images.append(image)
        labels.append(label)

    train_images = torch.stack(images, dim=0).contiguous()
    train_labels = torch.tensor(labels, dtype=torch.long)

    if cache_device == "gpu":
        train_images = train_images.cuda(non_blocking=True)
        train_labels = train_labels.cuda(non_blocking=True)

    print(
        f"[INFO] Cached distilled train set: {train_images.shape[0]} images on {cache_device.upper()} | "
        f"Resolution={args.resolution} | Batch={args.train_batch_size}"
    )
    return train_images, train_labels, cache_device

def sample_random_resized_crop_params(height, width, args):
    area = float(height * width)
    log_ratio_min = math.log(args.train_aspect_min)
    log_ratio_max = math.log(args.train_aspect_max)

    for _ in range(10):
        target_area = random.uniform(args.train_crop_min, 1.0) * area
        aspect_ratio = math.exp(random.uniform(log_ratio_min, log_ratio_max))
        crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            top = 0 if crop_h == height else random.randint(0, height - crop_h)
            left = 0 if crop_w == width else random.randint(0, width - crop_w)
            return top, left, crop_h, crop_w

    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return top, left, crop_size, crop_size

def apply_gpu_batch_augmentation(images, args):
    if args.aug_policy != "distill_eval" or not bool(args.use_gpu_batch_aug):
        return images

    batch_size, _, height, width = images.shape
    augmented = []
    for idx in range(batch_size):
        top, left, crop_h, crop_w = sample_random_resized_crop_params(height, width, args)
        cropped = images[idx:idx + 1, :, top:top + crop_h, left:left + crop_w]
        resized = F.interpolate(cropped, size=(args.resolution, args.resolution), mode="bilinear", align_corners=False)
        augmented.append(resized)
    images = torch.cat(augmented, dim=0)

    if args.hflip_prob > 0:
        mask = torch.rand(batch_size, device=images.device) < args.hflip_prob
        if mask.any():
            images[mask] = torch.flip(images[mask], dims=[3])
    if args.vflip_prob > 0:
        mask = torch.rand(batch_size, device=images.device) < args.vflip_prob
        if mask.any():
            images[mask] = torch.flip(images[mask], dims=[2])

    return images

def iterate_train_batches(train_images, train_labels, batch_size):
    num_samples = train_labels.shape[0]
    if train_labels.is_cuda:
        permutation = torch.randperm(num_samples, device=train_labels.device)
    else:
        permutation = torch.randperm(num_samples)

    for start in range(0, num_samples, batch_size):
        indices = permutation[start:start + batch_size]
        images = train_images[indices]
        labels = train_labels[indices]
        if not images.is_cuda:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        yield images, labels

def forward_logits(model, images):
    output = model(images)
    if isinstance(output, tuple):
        return output[1]
    return output

def train_once(model, args, num_classes, classname, train_images, train_labels, val_loader, generated_set_label, cache_device):
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()
    optimizer = build_optimizer(model, args)
    steps_per_epoch = math.ceil(train_labels.shape[0] / args.train_batch_size)
    total_iterations = steps_per_epoch * args.num_epochs
    scheduler, scheduler_milestones = build_scheduler(optimizer, args, total_iterations)

    print(
        f"[GeneratedSet {generated_set_label}] Optimizer={args.optimizer}, LR={args.lr}, Epochs={args.num_epochs}, "
        f"Scheduler={args.scheduler}, Milestones={scheduler_milestones}, "
        f"Pretrained={bool(args.pretrained)}, TrainCache={cache_device.upper()}, GPUBatchAug={bool(args.use_gpu_batch_aug)}"
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        start = time.time()
        losses = []
        accs = []
        model.train()

        for images, targets in iterate_train_batches(train_images, train_labels, args.train_batch_size):
            images = apply_gpu_batch_augmentation(images, args)

            output = forward_logits(model, images)
            loss = criterion(output, targets)
            preds = torch.argmax(output, dim=1)
            acc = (preds == targets).float().mean().item()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected at epoch {epoch + 1}, step {global_step + 1}. "
                    "This usually means the classifier protocol is too aggressive for the current setting."
                )

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            losses.append(loss.item())
            accs.append(acc)
            global_step += 1

        current_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) == 1 or (epoch + 1) % args.log_interval == 0 or (epoch + 1) == args.num_epochs:
            print(
                f"[GeneratedSet {generated_set_label}][Epoch {epoch + 1}/{args.num_epochs}] "
                f"Loss={np.mean(losses):.4f}, Acc={np.mean(accs) * 100:.2f}%, "
                f"LR={current_lr:.6f}, Time={time.time() - start:.2f}s, Iter={global_step}/{total_iterations}"
            )

    model.eval()
    oa, _ = test_acc(model, classname, val_loader, args.num_epochs, num_classes)
    print(f"[GeneratedSet {generated_set_label}] Final OA = {oa * 100:.2f}%")
    return oa


def train_once_loader(model, args, num_classes, classname, train_loader, val_loader, train_set_label):
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()
    optimizer = build_optimizer(model, args)
    steps_per_epoch = len(train_loader)
    total_iterations = steps_per_epoch * args.num_epochs
    scheduler, scheduler_milestones = build_scheduler(optimizer, args, total_iterations)

    print(
        f"[TrainSet {train_set_label}] Optimizer={args.optimizer}, LR={args.lr}, Epochs={args.num_epochs}, "
        f"Scheduler={args.scheduler}, Milestones={scheduler_milestones}, "
        f"Pretrained={bool(args.pretrained)}, TrainMode=real-full-dataloader"
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        start = time.time()
        losses = []
        accs = []
        model.train()

        for images, targets, _ in train_loader:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            output = forward_logits(model, images)
            loss = criterion(output, targets)
            preds = torch.argmax(output, dim=1)
            acc = (preds == targets).float().mean().item()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected at epoch {epoch + 1}, step {global_step + 1}. "
                    "This usually means the classifier protocol is too aggressive for the current setting."
                )

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            losses.append(loss.item())
            accs.append(acc)
            global_step += 1

        current_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) == 1 or (epoch + 1) % args.log_interval == 0 or (epoch + 1) == args.num_epochs:
            print(
                f"[TrainSet {train_set_label}][Epoch {epoch + 1}/{args.num_epochs}] "
                f"Loss={np.mean(losses):.4f}, Acc={np.mean(accs) * 100:.2f}%, "
                f"LR={current_lr:.6f}, Time={time.time() - start:.2f}s, Iter={global_step}/{total_iterations}"
            )

    model.eval()
    oa, _ = test_acc(model, classname, val_loader, args.num_epochs, num_classes)
    print(f"[TrainSet {train_set_label}] Final OA = {oa * 100:.2f}%")
    return oa


def main(args):
    torch.backends.cudnn.benchmark = True
    args.dataset = args.dataset.upper()
    mode = args.mode.lower()
    if mode not in ("template", "prototype", "full"):
        raise ValueError(f"Unsupported mode: {args.mode}. Choose from template, prototype, full.")
    setting_label = args.setting_tag if args.setting_tag is not None else ("full" if mode == "full" else args.weight)

    dataset_meta = {
        "UCM": (
            21,
            (
                "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
                "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
                "harbor", "intersection", "mediumresidential", "mobilehomepark",
                "overpass", "parkinglot", "river", "runway", "sparseresidential",
                "storagetanks", "tenniscourt"
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
                "storagetanks", "viaduct"
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
                "terrace", "thermal_power_station", "wetland"
            ),
        ),
    }

    if args.dataset not in dataset_meta:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    num_classes, classname = dataset_meta[args.dataset]

    generation_labels = [str(setting_label)] if mode == "full" else build_generated_setting_labels(setting_label, args)

    val_dataset = scene_dataset(
        root_dir=args.root_dir,
        pathfile=f"./dataset/{args.dataset}_test.txt",
        transform=build_val_transform(args),
    )
    val_loader = data.DataLoader(
        val_dataset,
        **build_dataloader_kwargs(
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        ),
    )

    if mode == "full":
        result_dir = os.path.join(args.save_path_prefix, args.dataset, mode, str(setting_label))
    else:
        result_dir = os.path.join(args.save_path_prefix, args.dataset, mode, str(setting_label), f"IPC_{args.IPC}")
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, "test_log.txt")

    protocol_msg = (
        f"Protocol: optimizer={args.optimizer}, lr={args.lr}, scheduler={args.scheduler}, "
        f"milestones={args.lr_decay_milestones}, gamma={args.lr_decay_gamma}, epochs={args.num_epochs}, "
        f"pretrained={bool(args.pretrained)}, aug_policy={args.aug_policy}, resolution={args.resolution}, "
        f"mode={mode}, generated_sets={len(generation_labels)}, train_bs={args.train_batch_size}, val_bs={args.val_batch_size}, "
        f"num_workers={args.num_workers}, prefetch_factor={args.prefetch_factor}, gpu_batch_aug={bool(args.use_gpu_batch_aug)}"
    )
    print(protocol_msg)
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(protocol_msg + "\n")

    full_train_loader = None
    if mode == "full":
        full_train_dataset = scene_dataset(
            root_dir=args.root_dir,
            pathfile=f"./dataset/{args.dataset}_train.txt",
            transform=build_full_train_transform(args),
        )
        full_train_loader = data.DataLoader(
            full_train_dataset,
            **build_dataloader_kwargs(
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
            ),
        )
        full_msg = (
            f"FullTrain: dataset={args.dataset}, train_samples={len(full_train_dataset)}, "
            f"test_samples={len(val_dataset)}"
        )
        print(full_msg)
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(full_msg + "\n")

    nets = [n.strip() for n in args.network.split(",") if n.strip()]
    for net in nets:
        oa_list = []
        ipc_msg = f" | IPC={args.IPC}" if mode != "full" else ""
        print(f"\n========== Training {net} on {args.dataset} | Mode={mode.upper()}{ipc_msg} ==========")

        for gen_idx, generation_label in enumerate(generation_labels):
            set_seed = args.seed + gen_idx
            seed_everything(set_seed)
            model = build_model(net, num_classes, pretrained=bool(args.pretrained))

            if mode == "full":
                print(f"\n[INFO] Loading real full train split | Mode={mode.upper()} | Setting={generation_label}")
                oa = train_once_loader(
                    model,
                    args,
                    num_classes,
                    classname,
                    full_train_loader,
                    val_loader,
                    train_set_label=generation_label,
                )
            else:
                print(f"\n[INFO] Loading distilled dataset | Mode={mode.upper()} | Setting={generation_label} | IPC={args.IPC}")
                train_lines = load_sd_distilled(
                    sd_root=args.sd_root,
                    mode=mode,
                    dataset_name=args.dataset,
                    weight=args.weight,
                    setting_tag=generation_label,
                    IPC=args.IPC,
                    classname=classname,
                )
                train_images, train_labels, cache_device = preload_train_cache(train_lines, args)
                oa = train_once(model, args, num_classes, classname, train_images, train_labels, val_loader, generation_label, cache_device)
                del train_images
                del train_labels

            oa_list.append((generation_label, oa))
            del model
            torch.cuda.empty_cache()

        mean_oa = np.mean([oa for _, oa in oa_list]) * 100
        std_oa = np.std([oa for _, oa in oa_list]) * 100
        for generation_label, oa in oa_list:
            if mode == "full":
                detail_msg = (
                    f"Dataset={args.dataset}, Mode={mode.upper()}, Setting={setting_label}, Train_Set=real_train, "
                    f"Network={net}, OA={oa * 100:.2f}%"
                )
            else:
                detail_msg = (
                    f"Dataset={args.dataset}, Mode={mode.upper()}, Setting={setting_label}, Generated_Set={generation_label}, IPC={args.IPC}, "
                    f"Network={net}, OA={oa * 100:.2f}%"
                )
            print(detail_msg)
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(detail_msg + "\n")

        if mode == "full":
            msg = (
                f"Dataset={args.dataset}, Mode={mode.upper()}, Setting={setting_label}, "
                f"Network={net}, Runs={len(generation_labels)}, Mean OA={mean_oa:.2f}%, Std={std_oa:.2f}%"
            )
        else:
            msg = (
                f"Dataset={args.dataset}, Mode={mode.upper()}, Setting={setting_label}, IPC={args.IPC}, "
                f"Network={net}, Generated_Sets={len(generation_labels)}, Mean OA={mean_oa:.2f}%, Std={std_oa:.2f}%"
            )
        print(msg)
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(msg + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="UCM")
    parser.add_argument("--mode", type=str, default="template")
    parser.add_argument("--weight", type=str, default="default")
    parser.add_argument("--setting_tag", type=str, default=None)
    parser.add_argument("--network", type=str, default="resnet18", help=f"Comma-separated networks from: {','.join(SUPPORTED_NETWORKS)}")
    parser.add_argument("--IPC", type=int, default=3)
    parser.add_argument("--root_dir", type=str, default="/proj/cvl/users/x_xuyon/Data/VisionLanguage/")
    parser.add_argument("--sd_root", type=str, default="./generated_data")
    parser.add_argument("--save_path_prefix", type=str, default="./results/")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_generated_sets", type=int, default=10)
    parser.add_argument("--generated_set_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="multistep", choices=["none", "multistep"])
    parser.add_argument("--lr_decay_milestones", type=str, default="0.6667,0.8333")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--aug_policy", type=str, default="distill_eval", choices=["basic", "distill_eval"])
    parser.add_argument("--train_crop_min", type=float, default=0.67)
    parser.add_argument("--train_aspect_min", type=float, default=0.75)
    parser.add_argument("--train_aspect_max", type=float, default=1.3333)
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--vflip_prob", type=float, default=0.5)
    parser.add_argument("--train_cache_device", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--use_gpu_batch_aug", type=int, default=1)
    main(parser.parse_args())