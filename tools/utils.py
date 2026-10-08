import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms


def test_acc(model, classname, data_loader, epoch, num_classes, print_per_batches=10, log_file=None):
    """
    Evaluate model accuracy on validation/test set.
    Returns:
        OA (float): overall accuracy
        class_acc (np.array): per-class accuracy
    """
    model.eval()
    total = 0
    correct = 0
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]
    class_acc = np.zeros((num_classes, 1))
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            _, outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_batch = (predicted == labels).squeeze()

            total += labels.size(0)
            correct += correct_batch.sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += correct_batch[i].item()
                class_total[label] += 1

            if (batch_idx + 1) % print_per_batches == 0:
                acc_batch = 100.0 * correct_batch.sum().item() / labels.size(0)
                print(f"Epoch[{epoch}] - Val Batch [{batch_idx+1}/{num_batches}] OA: {acc_batch:.2f}%")

    logs = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc[i] = class_correct[i] / class_total[i]
        else:
            class_acc[i] = 0.0
        line = f"[{i:2d}] Accuracy of {classname[i]:<25}: {100 * class_acc[i][0]:.2f}%"
        print(line)
        logs.append(line)

    OA = correct / total
    AA = class_acc.mean()
    summary = [
        f"=== Epoch[{epoch}] Validation OA: {100.0 * OA:.2f}%",
        f"=== Epoch[{epoch}] Validation AA: {100.0 * AA:.2f}%"
    ]
    print("\n".join(summary))

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write('\n'.join(logs + summary) + '\n')

    return OA, class_acc

def test_acc_latent(model, vae, classname, data_loader, epoch, num_classes, print_per_batches=10, log_file=None):
    """
    Evaluate latent classifier accuracy on validation/test set.

    Args:
        model: latent classifier
        vae: frozen AutoencoderKL
        classname: tuple of class names
        data_loader: dataloader yielding (image, label, path)
        epoch: current epoch
        num_classes: number of classes
    Returns:
        OA (float): overall accuracy
        class_acc (np.array): per-class accuracy
    """

    model.eval()
    vae.eval()

    total = 0
    correct = 0
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]
    class_acc = np.zeros((num_classes, 1))
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(data_loader):
            images = images.cuda()
            labels = labels.cuda()

            # ----------------------------------
            # Encode images to latent
            # ----------------------------------
            latents = vae.encode(
                images.to(dtype=torch.float32)
            ).latent_dist.mean
            latents = latents * vae.config.scaling_factor

            # ----------------------------------
            # Forward latent classifier
            # ----------------------------------
            outputs = model(latents)
            _, predicted = torch.max(outputs, 1)
            correct_batch = (predicted == labels)

            total += labels.size(0)
            correct += correct_batch.sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += correct_batch[i].item()
                class_total[label] += 1

            if (batch_idx + 1) % print_per_batches == 0:
                acc_batch = 100.0 * correct_batch.sum().item() / labels.size(0)
                print(
                    f"Epoch[{epoch}] - Val Batch "
                    f"[{batch_idx+1}/{num_batches}] "
                    f"OA: {acc_batch:.2f}%"
                )

    logs = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc[i] = class_correct[i] / class_total[i]
        else:
            class_acc[i] = 0.0

        line = f"[{i:2d}] Accuracy of {classname[i]:<25}: {100 * class_acc[i][0]:.2f}%"
        print(line)
        logs.append(line)

    OA = correct / total
    AA = class_acc.mean()

    summary = [
        f"=== Epoch[{epoch}] Validation OA: {100.0 * OA:.2f}%",
        f"=== Epoch[{epoch}] Validation AA: {100.0 * AA:.2f}%"
    ]
    print("\n".join(summary))

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write('\n'.join(logs + summary) + '\n')

    return OA, class_acc


def preprocess_image(img, args):
    """
    Apply standard preprocessing (resize, normalize) for input image.
    Args:
        img (PIL.Image): input image
        args: must have .crop_size
    Returns:
        torch.Tensor: preprocessed and normalized image tensor (1,C,H,W)
    """
    transform = transforms.Compose([
        transforms.Resize(size=(args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    tensor = transform(img).unsqueeze(0).cuda().requires_grad_()
    return tensor


def recreate_image(im_as_var):
    """
    Recreate a numpy image (H,W,C) from a normalized torch variable.
    Args:
        im_as_var (Variable or Tensor): input image tensor with shape (1,C,H,W)
    Returns:
        np.uint8 ndarray: image in (H,W,C), range [0,255]
    """
    if isinstance(im_as_var, torch.autograd.Variable):
        im_as_var = im_as_var.data

    im_np = im_as_var.cpu().numpy()[0].copy()
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]

    for c in range(3):
        im_np[c] = im_np[c] * reverse_std[c] + reverse_mean[c]

    im_np = np.clip(im_np, 0, 1)
    im_np = np.round(im_np * 255).astype(np.uint8)
    im_np = im_np.transpose(1, 2, 0)  # CHW -> HWC
    return im_np
