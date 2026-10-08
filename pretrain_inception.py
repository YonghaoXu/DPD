import argparse
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
from torchvision import transforms

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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_logits(model, images):
    output = model(images)
    if isinstance(output, tuple):
        return output[1]
    return output


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def build_inception(num_classes, pretrained=True):
    model = models.inception_v3(pretrained=pretrained, aux_logits=False, transform_input=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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


def parse_scheduler_milestones(spec, total_epochs):
    milestones = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        milestone = int(round(total_epochs * value)) if value < 1.0 else int(round(value))
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
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def build_train_transform(args):
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


def build_eval_transform(args):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        normalize,
    ])


def evaluate(model, loader, device, classnames, split_name, report_per_class=False):
    model.eval()
    num_classes = len(classnames)
    correct = 0
    total = 0
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = forward_logits(model, images)
            preds = torch.argmax(logits, dim=1)
            mask = preds.eq(labels)
            correct += mask.sum().item()
            total += labels.numel()
            for idx in range(labels.numel()):
                label = int(labels[idx].item())
                class_total[label] += 1
                class_correct[label] += int(mask[idx].item())

    class_acc = []
    for idx, class_name in enumerate(classnames):
        acc = 0.0 if class_total[idx] == 0 else class_correct[idx] / class_total[idx]
        class_acc.append(acc)
        if report_per_class:
            print(f"[{idx:2d}] Accuracy of {class_name:<25}: {acc * 100:.2f}%")

    oa = correct / max(total, 1)
    aa = float(np.mean(class_acc))
    print(f"[INFO] {split_name} OA={oa * 100:.2f}%, AA={aa * 100:.2f}%")
    return oa, aa


def save_checkpoint(model, path, args, dataset, classnames, epoch, oa=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": unwrap_model(model).state_dict(),
        "dataset": dataset,
        "classnames": tuple(classnames),
        "num_classes": len(classnames),
        "epoch": epoch,
        "resolution": args.resolution,
        "oa": oa,
        "arch": "inception_v3",
    }
    torch.save(payload, path)
    print(f"[INFO] Saved checkpoint: {path}")


def main(args):
    args.dataset = args.dataset.upper()
    if args.dataset not in DATASET_META:
        raise ValueError(f"Unknown dataset: {args.dataset}. Choose from {sorted(DATASET_META)}")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    num_classes, classnames = DATASET_META[args.dataset]
    save_dir = os.path.join(args.output_dir, args.dataset)
    final_ckpt = args.checkpoint_name or "inception_v3.pth"
    final_path = os.path.join(save_dir, final_ckpt)
    best_path = os.path.join(save_dir, "inception_v3_best.pth")

    train_dataset = scene_dataset(
        root_dir=args.root_dir,
        pathfile=os.path.join("dataset", f"{args.dataset}_train.txt"),
        transform=build_train_transform(args),
    )
    train_eval_dataset = scene_dataset(
        root_dir=args.root_dir,
        pathfile=os.path.join("dataset", f"{args.dataset}_train.txt"),
        transform=build_eval_transform(args),
    )
    train_loader = data.DataLoader(
        train_dataset,
        **build_dataloader_kwargs(args.train_batch_size, True, args.num_workers, args.prefetch_factor),
    )
    train_eval_loader = data.DataLoader(
        train_eval_dataset,
        **build_dataloader_kwargs(args.val_batch_size, False, args.num_workers, args.prefetch_factor),
    )

    test_loader = None
    if args.eval_test:
        test_dataset = scene_dataset(
            root_dir=args.root_dir,
            pathfile=os.path.join("dataset", f"{args.dataset}_test.txt"),
            transform=build_eval_transform(args),
        )
        test_loader = data.DataLoader(
            test_dataset,
            **build_dataloader_kwargs(args.val_batch_size, False, args.num_workers, args.prefetch_factor),
        )

    print(
        f"[INFO] Dataset={args.dataset} | Train={len(train_dataset)} | Classes={num_classes} | "
        f"Device={device} | SaveDir={save_dir}"
    )

    model = build_inception(num_classes, pretrained=bool(args.pretrained)).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda" and args.data_parallel:
        model = nn.DataParallel(model)

    optimizer = build_optimizer(model, args)
    scheduler, milestones = build_scheduler(optimizer, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    print(
        f"[INFO] Optimizer={args.optimizer}, LR={args.lr}, Epochs={args.num_epochs}, "
        f"Scheduler={args.scheduler}, Milestones={milestones}, Pretrained={bool(args.pretrained)}"
    )

    best_metric = -1.0
    best_epoch = -1
    for epoch in range(args.num_epochs):
        model.train()
        start = time.time()
        losses = []
        accs = []
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = forward_logits(model, images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            losses.append(loss.item())
            accs.append(preds.eq(labels).float().mean().item())

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch + 1}/{args.num_epochs}] Loss={np.mean(losses):.4f}, "
            f"Acc={np.mean(accs) * 100:.2f}%, LR={lr:.6f}, Time={time.time() - start:.2f}s"
        )

        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.num_epochs:
            train_oa, _ = evaluate(model, train_eval_loader, device, classnames, "Train", report_per_class=False)
            metric = train_oa
            if test_loader is not None:
                test_oa, _ = evaluate(model, test_loader, device, classnames, "Test", report_per_class=False)
                metric = test_oa
            if metric >= best_metric:
                best_metric = metric
                best_epoch = epoch + 1
                save_checkpoint(model, best_path, args, args.dataset, classnames, epoch + 1, oa=metric)

    save_checkpoint(model, final_path, args, args.dataset, classnames, args.num_epochs, oa=best_metric)
    print(f"[DONE] Best checkpoint epoch={best_epoch}, metric={best_metric * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="UCM", help="UCM, AID, or NWPU")
    parser.add_argument("--root_dir", type=str, default="/proj/cvl/users/x_xuyon/Data/VisionLanguage/")
    parser.add_argument("--output_dir", type=str, default="./pretrain")
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default="multistep", choices=["none", "multistep"])
    parser.add_argument("--lr_decay_milestones", type=str, default="0.6667,0.8333")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--eval_test", type=int, default=0, help="Also report fixed test split accuracy during pretraining.")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--train_crop_min", type=float, default=0.67)
    parser.add_argument("--train_aspect_min", type=float, default=0.75)
    parser.add_argument("--train_aspect_max", type=float, default=1.3333)
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--vflip_prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--data_parallel", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())