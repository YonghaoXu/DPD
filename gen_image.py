
import os
import argparse
import json
import torch
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_gen_latents import (
    StableDiffusionGenLatentsPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_latents2img import (
    StableDiffusionLatents2ImgPipeline,
)


def load_prototype_pairs(dataset, prototypes_dir, ipc):
    json_path = os.path.join(prototypes_dir, f"prototype_pairs_{dataset}_n_cluster_{ipc}.json")
    assert os.path.exists(json_path), f"[prototype] File not found: {json_path}"
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def get_classname_mapping(dataset):
    dataset = dataset.upper()
    if dataset == "UCM":
        return {
            "agricultural": "agricultural area",
            "airplane": "airplane",
            "baseballdiamond": "baseball diamond",
            "beach": "beach",
            "buildings": "buildings",
            "chaparral": "chaparral",
            "denseresidential": "dense residential area",
            "forest": "forest",
            "freeway": "freeway",
            "golfcourse": "golf course",
            "harbor": "harbor",
            "intersection": "intersection",
            "mediumresidential": "medium residential area",
            "mobilehomepark": "mobile home park",
            "overpass": "overpass",
            "parkinglot": "parking lot",
            "river": "river",
            "runway": "runway",
            "sparseresidential": "sparse residential area",
            "storagetanks": "storage tanks",
            "tenniscourt": "tennis court",
        }
    if dataset == "AID":
        return {
            "airport": "airport",
            "bareland": "bare land",
            "baseballfield": "baseball field",
            "beach": "beach",
            "bridge": "bridge",
            "center": "city center",
            "church": "church",
            "commercial": "commercial area",
            "denseresidential": "dense residential area",
            "desert": "desert",
            "farmland": "farmland",
            "forest": "forest",
            "industrial": "industrial area",
            "meadow": "meadow",
            "mediumresidential": "medium residential area",
            "mountain": "mountain",
            "park": "park",
            "parking": "parking lot",
            "playground": "playground",
            "pond": "pond",
            "port": "port",
            "railwaystation": "railway station",
            "resort": "resort",
            "river": "river",
            "school": "school",
            "sparseresidential": "sparse residential area",
            "square": "city square",
            "stadium": "stadium",
            "storagetanks": "storage tanks",
            "viaduct": "viaduct",
        }
    if dataset == "NWPU":
        return {
            "airplane": "airplane",
            "airport": "airport",
            "baseball_diamond": "baseball diamond",
            "basketball_court": "basketball court",
            "beach": "beach",
            "bridge": "bridge",
            "chaparral": "chaparral",
            "church": "church",
            "circular_farmland": "circular farmland",
            "cloud": "cloud",
            "commercial_area": "commercial area",
            "dense_residential": "dense residential area",
            "desert": "desert",
            "forest": "forest",
            "freeway": "freeway",
            "golf_course": "golf course",
            "ground_track_field": "ground track field",
            "harbor": "harbor",
            "industrial_area": "industrial area",
            "intersection": "intersection",
            "island": "island",
            "lake": "lake",
            "meadow": "meadow",
            "medium_residential": "medium residential area",
            "mobile_home_park": "mobile home park",
            "mountain": "mountain",
            "overpass": "overpass",
            "palace": "palace",
            "parking_lot": "parking lot",
            "railway": "railway",
            "railway_station": "railway station",
            "rectangular_farmland": "rectangular farmland",
            "river": "river",
            "roundabout": "roundabout",
            "runway": "runway",
            "sea_ice": "sea ice",
            "ship": "ship",
            "snowberg": "snowberg",
            "sparse_residential": "sparse residential area",
            "stadium": "stadium",
            "storage_tank": "storage tank",
            "tennis_court": "tennis court",
            "terrace": "terrace",
            "thermal_power_station": "thermal power station",
            "wetland": "wetland",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_target_resolution(dataset):
    dataset = dataset.upper()
    if dataset == "AID":
        return (600, 600)
    return (256, 256)


def sanitize_decimal_tag(value):
    return str(value).replace(".", "p")


def build_generation_setting_labels(setting_label, args):
    if args.generation_repeat_id is not None:
        return [f"{setting_label}-gid{args.generation_repeat_id}"]
    if args.num_generation_repeats <= 1:
        return [str(setting_label)]
    return [f"{setting_label}-gid{i}" for i in range(1, args.num_generation_repeats + 1)]


def build_template_prompt(readable_name, prompt_tail=None):
    base = f"A satellite image of {readable_name}"
    prompt_tail = (prompt_tail or "").strip()
    if prompt_tail:
        return f"{base}. {prompt_tail}"
    return base


def resolve_pair_image_path(image_path, image_root):
    if not os.path.isabs(image_path):
        return os.path.join(image_root, image_path)

    if os.path.exists(image_path):
        return image_path

    normalized = image_path.replace('\\', '/')
    marker = '/Data/VisionLanguage/'
    if image_root and marker in normalized:
        suffix = normalized.split(marker, 1)[1].lstrip('/')
        candidate = os.path.join(image_root, suffix.replace('/', os.sep))
        if os.path.exists(candidate):
            return candidate

    for dataset_dir in ['UCMerced_LandUse', 'AID', 'NWPU-RESISC45']:
        token = f'/{dataset_dir}/'
        if image_root and token in normalized:
            suffix = normalized.split(token, 1)[1].lstrip('/')
            candidate = os.path.join(image_root, dataset_dir, suffix.replace('/', os.sep))
            if os.path.exists(candidate):
                return candidate

    return image_path


def build_prototype_prompt(readable_name, pair, args):
    return build_template_prompt(readable_name)


def load_init_image(image_path, size):
    return Image.open(image_path).convert("RGB").resize(size, Image.BICUBIC)


def check_latent_classifier_class_order(dataset, classnames):
    from train_cls_latent import DATASET_META

    if dataset not in DATASET_META:
        raise ValueError(f"Unknown dataset for latent classifier: {dataset}")
    _, latent_classifier_classnames = DATASET_META[dataset]
    if tuple(classnames) != tuple(latent_classifier_classnames):
        raise ValueError(
            "Class order mismatch between gen_image.py and train_cls_latent.py. "
            f"gen_image.py={tuple(classnames)}, train_cls_latent.py={tuple(latent_classifier_classnames)}"
        )


def load_latent_classifier(args, dataset, classnames):
    from train_cls_latent import DATASET_META, LatentClassifier

    check_latent_classifier_class_order(dataset, classnames)
    num_classes, _ = DATASET_META[dataset]

    ckpt_path = args.latent_classifier_ckpt
    if isinstance(ckpt_path, str) and ckpt_path.strip().lower() in ("", "none", "null"):
        ckpt_path = None
    if ckpt_path is None:
        ckpt_path = os.path.join(".", "pretrain", dataset, "latent_classifier.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Latent classifier checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    state_dict = {}
    for key, value in checkpoint.items():
        clean_key = key[len("module."):] if key.startswith("module.") else key
        state_dict[clean_key] = value

    model = LatentClassifier(num_classes).to("cuda")
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model, ckpt_path


def score_candidate_latents(latent_classifier, candidate_latents, target_class_id, metric):
    if metric != "target_logit_margin":
        raise ValueError(f"Unsupported selection_metric: {metric}")

    with torch.no_grad():
        logits = latent_classifier(candidate_latents.float())
        target_logits = logits[:, target_class_id]
        other_logits = logits.clone()
        other_logits[:, target_class_id] = -torch.inf
        max_other_logits = other_logits.max(dim=1).values
        return target_logits - max_other_logits


def decode_latents_to_pil(pipe, latents):
    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        return pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)


def model_requires_class_labels(base_model):
    unet_config = UNet2DConditionModel.load_config(base_model, subfolder="unet")
    num_class_embeds = unet_config.get("num_class_embeds", None)
    return num_class_embeds is not None and num_class_embeds > 0


def unet_requires_class_labels(unet):
    num_class_embeds = getattr(unet.config, "num_class_embeds", None)
    return (
        (num_class_embeds is not None and num_class_embeds > 0)
        or getattr(unet, "class_embedding", None) is not None
    )


def patch_unet_default_class_labels(pipe):
    if not unet_requires_class_labels(pipe.unet):
        return pipe
    if getattr(pipe.unet, "_r1_dpd_default_class_labels", False):
        return pipe

    original_forward = pipe.unet.forward

    def forward_with_default_class_labels(
        sample,
        timestep,
        encoder_hidden_states,
        *args,
        class_labels=None,
        **kwargs,
    ):
        if class_labels is None:
            class_labels = torch.zeros(sample.shape[0], device=sample.device, dtype=torch.long)
        return original_forward(
            sample,
            timestep,
            encoder_hidden_states,
            *args,
            class_labels=class_labels,
            **kwargs,
        )

    pipe.unet.forward = forward_with_default_class_labels
    pipe.unet._r1_dpd_default_class_labels = True
    print("[INFO] Patched UNet forward to use default class_labels=0 when class_labels is missing.")
    return pipe


def build_latents2img_pipe_from_gen_pipe(gen_latents_pipe):
    return StableDiffusionLatents2ImgPipeline(
        vae=gen_latents_pipe.vae,
        text_encoder=gen_latents_pipe.text_encoder,
        tokenizer=gen_latents_pipe.tokenizer,
        unet=gen_latents_pipe.unet,
        scheduler=gen_latents_pipe.scheduler,
        safety_checker=gen_latents_pipe.safety_checker,
        feature_extractor=gen_latents_pipe.feature_extractor,
        requires_safety_checker=getattr(gen_latents_pipe.config, "requires_safety_checker", True),
    )


def load_generation_pipeline(args, dtype):
    requires_class_labels = model_requires_class_labels(args.base_model)
    if requires_class_labels:
        print("[INFO] Base UNet requires class_labels. Missing class_labels will default to 0.")
    else:
        print("[INFO] Base UNet does not require class_labels.")

    if args.mode == "prototype":
        gen_latents_pipe = StableDiffusionGenLatentsPipeline.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
        )
        latents2img_pipe = build_latents2img_pipe_from_gen_pipe(gen_latents_pipe)

        gen_latents_pipe = gen_latents_pipe.to("cuda")
        latents2img_pipe = latents2img_pipe.to("cuda")

        gen_latents_pipe = patch_unet_default_class_labels(gen_latents_pipe)
        latents2img_pipe = patch_unet_default_class_labels(latents2img_pipe)

        return gen_latents_pipe, latents2img_pipe

    if requires_class_labels:
        pipe = DiffusionPipeline.from_pretrained(
            args.base_model,
            custom_pipeline="pipeline_text2earth_diffusion",
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=dtype)

    pipe = pipe.to("cuda")
    return patch_unet_default_class_labels(pipe)


def main(args):
    dataset = args.dataset.upper()
    class_map = get_classname_mapping(dataset)
    classnames = list(class_map.keys())
    readable_names = list(class_map.values())
    target_size = get_target_resolution(dataset)

    valid_modes = ["template", "prototype"]
    if args.mode not in valid_modes:
        raise ValueError(f"Invalid mode. Should be one of {valid_modes}.")

    if args.use_prototype_guidance and args.mode != "prototype":
        raise ValueError("Prototype guidance is only supported for prototype mode.")
    if args.num_candidates_per_prototype > 1 and args.mode != "prototype":
        raise ValueError("Discriminative selection is only supported for prototype mode.")
    if args.num_candidates_per_prototype > 1 and not args.use_prototype_guidance:
        raise ValueError("Discriminative selection requires --use_prototype_guidance.")
    if args.num_candidates_per_prototype < 1:
        raise ValueError("--num_candidates_per_prototype must be >= 1.")
    if args.candidate_batch_size is not None and args.candidate_batch_size < 1:
        raise ValueError("--candidate_batch_size must be >= 1 when set.")
    if args.use_spherical_prototype_target_sampling and not args.use_prototype_guidance:
        raise ValueError("Spherical prototype target sampling requires --use_prototype_guidance.")

    if args.setting_tag is not None:
        setting_label = args.setting_tag
    elif args.use_prototype_guidance:
        setting_label = (
            f"prototype-guidance-s{sanitize_decimal_tag(args.prototype_guidance_scale)}-"
            f"g{sanitize_decimal_tag(args.prototype_guidance_start)}-"
            f"{sanitize_decimal_tag(args.prototype_guidance_end)}"
        )
        if args.num_candidates_per_prototype > 1:
            setting_label = (
                f"{setting_label}-select-{args.selection_metric}-"
                f"r{args.num_candidates_per_prototype}"
            )
        if args.use_spherical_prototype_target_sampling:
            setting_label = f"{setting_label}-spherical-target"
    else:
        setting_label = args.mode

    lora_weights = args.lora_dir
    if isinstance(lora_weights, str) and lora_weights.strip().lower() in ("", "none", "null"):
        lora_weights = None
    if lora_weights is not None and not os.path.exists(lora_weights):
        raise FileNotFoundError(f"LoRA weights not found: {lora_weights}")

    generation_pipeline = load_generation_pipeline(args, torch.float16)

    if lora_weights is None:
        print(f"[INFO] No LoRA directory provided. Using base model only: {args.base_model}")
    else:
        print(f"[INFO] Loading LoRA weights from: {lora_weights}")

    if args.mode == "prototype":
        gen_latents_pipe, latents2img_pipe = generation_pipeline

        # The two pipes share modules such as UNet, text_encoder, and VAE.
        # Load LoRA once to avoid injecting the same adapter twice.
        if lora_weights is not None:
            gen_latents_pipe.load_lora_weights(lora_weights)
        if args.use_prototype_guidance:
            print(f"[INFO] prototype_guidance_scale: {args.prototype_guidance_scale}")
            print(
                "[INFO] prototype_guidance_start / prototype_guidance_end: "
                f"{args.prototype_guidance_start} / {args.prototype_guidance_end}"
            )
            if args.use_spherical_prototype_target_sampling:
                print("[INFO] Spherical prototype target sampling enabled.")
                print("[INFO] spherical_prototype_target_radius: 1 / sqrt(num_latent_dims)")
            print("[INFO] Prototype guidance enabled. latents2img starts from random Gaussian noise.")
            if args.num_candidates_per_prototype > 1:
                latent_classifier, latent_classifier_ckpt_path = load_latent_classifier(args, dataset, classnames)
                print(f"[INFO] Latent classifier checkpoint: {latent_classifier_ckpt_path}")
                print(f"[INFO] num_candidates_per_prototype: {args.num_candidates_per_prototype}")
                candidate_batch_size = args.candidate_batch_size or args.num_candidates_per_prototype
                print(f"[INFO] candidate_batch_size: {candidate_batch_size}")
                print(f"[INFO] selection_metric: {args.selection_metric}")
                print("[INFO] Discriminative selection enabled.")
            else:
                latent_classifier = None
        else:
            latent_classifier = None
            print("[INFO] Prototype guidance disabled. Using original prototype generation.")
    else:
        pipe = generation_pipeline
        if lora_weights is not None:
            pipe.load_lora_weights(lora_weights)

    generation_setting_labels = build_generation_setting_labels(setting_label, args)

    prototypes_dir = args.prototypes_dir

    if args.mode == "prototype":
        proto_pairs = load_prototype_pairs(dataset, prototypes_dir, args.IPC)

    for generation_label in generation_setting_labels:
        save_root = os.path.join(args.output_dir, dataset, args.mode, str(generation_label), f"IPC_{args.IPC}")
        os.makedirs(save_root, exist_ok=True)
        print(f"[INFO] Generating distilled dataset: {generation_label}")

        for class_id, (cls_key, readable_name) in enumerate(zip(classnames, readable_names)):
            save_dir = os.path.join(save_root, f"{class_id}_{cls_key}")
            os.makedirs(save_dir, exist_ok=True)

            if args.mode == "prototype":
                pairs = proto_pairs.get(cls_key, [])
                if len(pairs) < args.IPC:
                    print(f"[Warning] Class '{cls_key}' has {len(pairs)} prototype pairs, expected {args.IPC}")
                for i, pair in enumerate(pairs[:args.IPC]):
                    prompt = build_prototype_prompt(readable_name, pair, args)
                    image_path = resolve_pair_image_path(pair["image"], args.image_root)
                    if not os.path.exists(image_path):
                        print(f"[Warning] [Prototype] Image not found: {image_path}")
                        continue
                    init_image = load_init_image(image_path, (256, 256))
                    prototype_latents, noisy_latents = gen_latents_pipe(
                        prompt=prompt,
                        num_inference_steps=args.num_inference_steps,
                        image=init_image,
                        strength=args.prototype_strength,
                        guidance_scale=7.5,
                    )

                    if args.use_prototype_guidance:
                        if args.num_candidates_per_prototype > 1:
                            candidate_chunks = []
                            candidate_batch_size = args.candidate_batch_size or args.num_candidates_per_prototype
                            for candidate_start in range(0, args.num_candidates_per_prototype, candidate_batch_size):
                                current_batch_size = min(
                                    candidate_batch_size,
                                    args.num_candidates_per_prototype - candidate_start,
                                )
                                random_latents = torch.randn(
                                    (current_batch_size, *prototype_latents.shape[1:]),
                                    device=prototype_latents.device,
                                    dtype=prototype_latents.dtype,
                                )
                                candidate_latents = latents2img_pipe(
                                    prompt=[prompt] * current_batch_size,
                                    num_inference_steps=args.num_inference_steps,
                                    latents=random_latents,
                                    is_init=False,
                                    strength=1.0,
                                    guidance_scale=7.5,
                                    output_type="latent",
                                    prototype_latents=prototype_latents,
                                    prototype_guidance_scale=args.prototype_guidance_scale,
                                    prototype_guidance_start=args.prototype_guidance_start,
                                    prototype_guidance_end=args.prototype_guidance_end,
                                    spherical_prototype_target_sampling=args.use_spherical_prototype_target_sampling,
                                ).images
                                if not torch.is_tensor(candidate_latents):
                                    raise TypeError("Expected latents2img_pipe output_type='latent' to return a tensor.")
                                candidate_chunks.append(candidate_latents)

                            candidate_latents = torch.cat(candidate_chunks, dim=0)
                            scores = score_candidate_latents(
                                latent_classifier,
                                candidate_latents,
                                class_id,
                                args.selection_metric,
                            )
                            best_idx = int(torch.argmax(scores).item())
                            best_latents = candidate_latents[best_idx:best_idx + 1]
                            image = decode_latents_to_pil(latents2img_pipe, best_latents)[0]
                            print(
                                f"[INFO] Selected candidate {best_idx + 1}/{args.num_candidates_per_prototype} "
                                f"with {args.selection_metric}={scores[best_idx].item():.4f}"
                            )
                        else:
                            random_latents = torch.randn_like(prototype_latents)
                            image = latents2img_pipe(
                                prompt=prompt,
                                num_inference_steps=args.num_inference_steps,
                                latents=random_latents,
                                is_init=False,
                                strength=1.0,
                                guidance_scale=7.5,
                                prototype_latents=prototype_latents,
                                prototype_guidance_scale=args.prototype_guidance_scale,
                                prototype_guidance_start=args.prototype_guidance_start,
                                prototype_guidance_end=args.prototype_guidance_end,
                                spherical_prototype_target_sampling=args.use_spherical_prototype_target_sampling,
                            ).images[0]
                    else:
                        # noisy_latents has already been noised by gen_latents_pipe, so
                        # is_init=False prevents latents2img_pipe from adding noise again.
                        image = latents2img_pipe(
                            prompt=prompt,
                            num_inference_steps=args.num_inference_steps,
                            latents=noisy_latents,
                            is_init=False,
                            strength=args.prototype_strength,
                            guidance_scale=7.5,
                        ).images[0]
                    image = image.resize(target_size, Image.BICUBIC)
                    out_path = os.path.join(save_dir, f"{cls_key}_{i:03d}.png")
                    image.save(out_path)
                    print(f"[{dataset}] [Prototype IPC={args.IPC}] Saved: {out_path}")

            else:
                prompt = build_template_prompt(readable_name)
                for i in range(args.IPC):
                    image = pipe(prompt, num_inference_steps=args.num_inference_steps, guidance_scale=7.5).images[0]
                    image = image.resize(target_size, Image.BICUBIC)
                    image.save(os.path.join(save_dir, f"{cls_key}_{i:03d}.png"))
                    print(f"[{dataset}] [{args.mode}] Saved: {os.path.join(save_dir, f'{cls_key}_{i:03d}.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="UCM", help="UCM, AID, or NWPU")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--IPC", type=int, default=3)
    parser.add_argument("--setting_tag", type=str, default=None, help="Output tag for hyperparameter sweeps.")
    parser.add_argument("--num_generation_repeats", type=int, default=5, help="How many independently generated distilled datasets to create for the same model.")
    parser.add_argument("--generation_repeat_id", type=int, default=None, help="Optional 1-based generation id. If set, generate only this specific repeat as <setting>-gid<ID>.")
    parser.add_argument("--base_model", type=str, default="lcybuaa/Text2Earth")
    parser.add_argument("--base_lora_path", type=str, default="/proj/cvl/users/x_xuyon/Code/DPD/")
    parser.add_argument("--lora_dir", type=str, default=None, help="Direct path to a LoRA directory or checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./generated_data")
    parser.add_argument("--mode", type=str, default="template", choices=["template", "prototype"], help="template or prototype")
    parser.add_argument("--image_root", type=str, default="/proj/cvl/users/x_xuyon/Data/VisionLanguage/")
    parser.add_argument("--prototypes_dir", type=str, default="./prototypes")
    parser.add_argument("--prototype_strength", type=float, default=0.75, help="strength when using prototype mode")
    parser.add_argument("--prototype_prompt_mode", type=str, default="template", choices=["template", "stored_caption"])
    parser.add_argument("--use_prototype_guidance", action="store_true")
    parser.add_argument("--prototype_guidance_scale", type=float, default=1)
    parser.add_argument("--prototype_guidance_start", type=float, default=0.0)
    parser.add_argument("--prototype_guidance_end", type=float, default=0.8)
    parser.add_argument("--use_spherical_prototype_target_sampling", action="store_true")
    parser.add_argument("--num_candidates_per_prototype", type=int, default=5)
    parser.add_argument("--candidate_batch_size", type=int, default=None, help="Candidate generation batch size for discriminative selection. Default: num_candidates_per_prototype.")
    parser.add_argument("--selection_metric", type=str, default="target_logit_margin", choices=["target_logit_margin"])
    parser.add_argument("--latent_classifier_ckpt", type=str, default=None, help="Optional override. Default: ./pretrain/<DATASET>/latent_classifier.pth")
    main(parser.parse_args())
