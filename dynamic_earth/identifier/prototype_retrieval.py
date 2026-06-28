import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from skimage.io import imread


def _natural_sort_key(file_name):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", file_name)
    ]


def extract_features(image, model, processor, model_config, device):
    """Extract a dense feature map from an image."""
    model_type = model_config["model_type"]
    feature_dim = model_config["feature_dim"]
    patch_size = model_config["patch_size"]

    if model_type == "DINO":
        processed_image = processor(image).unsqueeze(0).to(device)
        height, width = processed_image.shape[-2:]
        features = model.get_intermediate_layers(processed_image)[0]
        features = (
            features[:, 1:]
            .permute(0, 2, 1)
            .reshape(1, feature_dim, height // patch_size, width // patch_size)
        )
    elif model_type == "DINOv2":
        processed_image = processor(images=image, return_tensors="pt").to(device)
        height, width = processed_image["pixel_values"].shape[-2:]
        features = model(**processed_image).last_hidden_state
        features = (
            features[:, 1:]
            .permute(0, 2, 1)
            .reshape(1, feature_dim, height // patch_size, width // patch_size)
        )
    else:
        raise ValueError("Unsupported model type. Use 'DINO' or 'DINOv2'.")

    return features


def _extract_encoder_features(image, encoder_model, encoder_processor, encoder_config, device, output_size=None):
    img_embed = extract_features(
        image,
        encoder_model,
        encoder_processor,
        encoder_config,
        device,
    )
    if output_size is not None:
        img_embed = F.interpolate(
            img_embed,
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )
    return img_embed


def _resolve_dataset_config(num_classes):
    if num_classes not in {2, 6, 8}:
        raise ValueError(
            f"Unsupported num_classes: {num_classes}. Expected one of: 2, 6, 8."
        )
    return list(range(num_classes))


@torch.no_grad()
def _pool_mask_feature_map(feature_map, mask):
    masked_feat = feature_map[:, mask]
    pooled_feature = masked_feat.mean(dim=1, keepdim=True).permute(1, 0)
    return F.normalize(pooled_feature, dim=-1)


@torch.no_grad()
def _collect_class_features(
    encoder_model,
    encoder_processor,
    encoder_config,
    class_name,
    support_set_root,
    device,
):
    support_mask_root = f"{support_set_root}_mask"
    img_dir = os.path.join(support_set_root, class_name)
    mask_dir = os.path.join(support_mask_root, class_name)

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        return []

    feature_list = []

    for img_file in sorted(os.listdir(img_dir), key=_natural_sort_key):
        if not (
            img_file.endswith(".png")
            or img_file.endswith(".JPEG")
            or img_file.endswith(".jpg")
        ):
            continue

        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, base_name + ".png")
        if not os.path.exists(mask_path):
            continue

        try:
            img = imread(img_path)

            img_embed = _extract_encoder_features(
                img,
                encoder_model,
                encoder_processor,
                encoder_config,
                device,
            )
            height, width = img.shape[:2]
            img_embed = F.interpolate(
                img_embed,
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)

            mask = imread(mask_path).astype(np.uint8)
            mask_tensor = torch.from_numpy(mask > 128).to(device)

            if mask_tensor.sum() == 0:
                continue

            masked_feat = img_embed[:, mask_tensor]
            masked_mean = masked_feat.mean(dim=1, keepdim=True).permute(1, 0)
            masked_mean = masked_mean / masked_mean.norm(dim=-1, keepdim=True)

            feature_list.append(masked_mean)

        except Exception:
            continue

    return feature_list


def _cluster_prototypes(feature_list, num_prototypes, device):
    all_feats = torch.cat(feature_list, dim=0)
    kmeans = KMeans(n_clusters=num_prototypes, random_state=42).fit(
        all_feats.cpu().numpy()
    )
    centers = torch.tensor(
        kmeans.cluster_centers_,
        device=device,
        dtype=all_feats.dtype,
    )
    return centers / centers.norm(dim=-1, keepdim=True)


@torch.no_grad()
def build_prototype_set(
    encoder_model,
    encoder_processor,
    encoder_config,
    prototype_classes,
    support_set_root,
    device,
):
    all_prototypes = []
    all_labels = []
    built_classes = []
    max_num_prototypes = encoder_config.get("max_num_prototypes", 10)

    for class_idx, class_name in enumerate(prototype_classes):
        feature_list = _collect_class_features(
            encoder_model,
            encoder_processor,
            encoder_config,
            class_name,
            support_set_root,
            device,
        )
        if not feature_list:
            continue

        num_samples = len(feature_list)
        num_prototypes = min(max(num_samples, 3), max_num_prototypes)
        centers = _cluster_prototypes(
            feature_list,
            num_prototypes,
            device,
        )

        all_prototypes.append(centers)
        all_labels.extend([class_idx] * len(centers))
        built_classes.append(class_name)
        print(f"[Prototype] {class_name}: {len(centers)} prototypes")

    if not all_prototypes:
        raise RuntimeError("No valid support prototypes were built.")

    print(f"[Prototype] Built prototypes for classes: {', '.join(built_classes)}")
    
    print(torch.cat(all_prototypes, dim=0))
    # exit()
    return {
        "class_prototypes_dino": torch.cat(all_prototypes, dim=0),
        "prototype_labels": torch.tensor(all_labels, dtype=torch.long, device=device),
    }


@torch.no_grad()
def build_binary_prototype_set(
    encoder_model,
    encoder_processor,
    encoder_config,
    prototype_classes,
    support_set_root,
    foreground_class,
    device,
):
    background_features = []
    for class_name in prototype_classes:
        if class_name == foreground_class:
            continue
        background_features.extend(
            _collect_class_features(
                encoder_model,
                encoder_processor,
                encoder_config,
                class_name,
                support_set_root,
                device,
            )
        )

    foreground_features = _collect_class_features(
        encoder_model,
        encoder_processor,
        encoder_config,
        foreground_class,
        support_set_root,
        device,
    )

    if not background_features or not foreground_features:
        raise RuntimeError("No valid support prototypes were built.")

    max_num_prototypes = encoder_config.get("max_num_prototypes", 5)
    num_background_prototypes = encoder_config.get(
        "num_background_prototypes", max_num_prototypes
    )
    num_foreground_prototypes = encoder_config.get(
        "num_foreground_prototypes", max_num_prototypes
    )

    background_centers = _cluster_prototypes(
        background_features,
        num_background_prototypes,
        device,
    )
    foreground_centers = _cluster_prototypes(
        foreground_features,
        num_foreground_prototypes,
        device,
    )

    print(f"[Prototype] Built prototypes for classes: background, {foreground_class}")
    print(f"[Prototype] background: {len(background_centers)} prototypes")
    print(f"[Prototype] {foreground_class}: {len(foreground_centers)} prototypes")

    return {
        "class_prototypes_dino": torch.cat(
            [background_centers, foreground_centers],
            dim=0,
        ),
        "prototype_labels": torch.tensor(
            [0] * len(background_centers) + [1] * len(foreground_centers),
            dtype=torch.long,
            device=device,
        ),
    }


@torch.no_grad()
def classify_candidate_masks(
    img_origin,
    proposal_masks,
    encoder_model,
    encoder_processor,
    encoder_config,
    prototype_state,
    device,
    output_size=None,
):
    if proposal_masks is None:
        return []

    features_dino = _extract_encoder_features(
        img_origin,
        encoder_model,
        encoder_processor,
        encoder_config,
        device,
        output_size=output_size or tuple(img_origin.shape[:2]),
    )

    mask_classes = []
    height, width = features_dino.shape[-2:]
    labels = prototype_state["prototype_labels"]

    for proposal_mask in proposal_masks:
        proposal_mask = torch.as_tensor(proposal_mask, device=features_dino.device, dtype=torch.uint8)
        proposal_mask = F.interpolate(
            proposal_mask[None, None],
            size=(height, width),
            mode="nearest",
        ).squeeze(0, 1).bool()

        if not proposal_mask.any():
            mask_classes.append(0)
            continue

        mask_feature = _pool_mask_feature_map(features_dino[0], proposal_mask).squeeze(0)
        similarity = F.cosine_similarity(
            mask_feature.unsqueeze(0),
            prototype_state["class_prototypes_dino"],
            dim=-1,
        )

        # Mode 1: global-max
        # Find the single most similar prototype and use its label.
        max_similarity_idx = similarity.argmax()
        mask_pred_sim_max = int(labels[max_similarity_idx].item())

        # Mode 2: category-mean
        # Average similarities within each class, then choose the class
        # with the highest mean similarity.
        #
        # num_classes = int(labels.max().item() + 1)
        # similarity_sum = torch.zeros(num_classes, device=similarity.device)
        # prototype_count = torch.zeros(num_classes, device=similarity.device)
        # similarity_sum.scatter_add_(0, labels, similarity)
        # prototype_count.scatter_add_(0, labels, torch.ones_like(similarity))
        # similarity_mean = similarity_sum / prototype_count.clamp_min(1)
        # mask_pred_sim_mean = int(similarity_mean.argmax().item())

        mask_classes.append(mask_pred_sim_max)

    return mask_classes


@torch.no_grad()
def identify(
    img1: np.array,
    img2: np.array,
    mask_data: np.array,
    img1_mask_num: int,
    model,
    processor,
    prototype_state,
    model_config,
    device: str = "cpu",
    is_instance_class: bool = True,
    num_classes: int = 6,
    identifier_output_size=(448, 448),
) -> np.array:
    img1_mask_classes = classify_candidate_masks(
        img1,
        proposal_masks=mask_data,
        encoder_model=model,
        encoder_processor=processor,
        encoder_config=model_config,
        prototype_state=prototype_state,
        device=device,
        output_size=identifier_output_size,
    )

    img2_mask_classes = classify_candidate_masks(
        img2,
        proposal_masks=mask_data,
        encoder_model=model,
        encoder_processor=processor,
        encoder_config=model_config,
        prototype_state=prototype_state,
        device=device,
        output_size=identifier_output_size,
    )

    if is_instance_class:
        change_instance_match = img1_mask_classes[:img1_mask_num] + img2_mask_classes[img1_mask_num:]
        change_idx = np.where(
            (np.array(img1_mask_classes) != np.array(img2_mask_classes))
            & np.array(change_instance_match).astype(bool)
        )
    else:
        change_idx = np.where(np.array(img1_mask_classes) != np.array(img2_mask_classes))

    all_classes = _resolve_dataset_config(num_classes)

    h, w = img1.shape[:2]
    label_map_t1 = np.full((h, w), num_classes, dtype=np.uint8)
    label_map_t2 = np.full((h, w), num_classes, dtype=np.uint8)
    for idx in change_idx[0]:
        mask = mask_data[idx]
        cls1 = img1_mask_classes[idx]
        cls2 = img2_mask_classes[idx]
        label_map_t1[mask] = cls1
        label_map_t2[mask] = cls2

    cmasks = np.array(mask_data)[change_idx]
    if num_classes <= 2:
        img1_mask_num = np.sum(change_idx[0] < img1_mask_num)
        keep_mask = [
            (img1_mask_classes[idx] == 1 or img2_mask_classes[idx] == 1)
            for idx in change_idx[0]
        ]
        cmasks_binary = np.array([
            mask for keep, mask in zip(keep_mask, cmasks) if keep
        ]) if len(cmasks) > 0 else np.empty((0, *cmasks.shape[1:]))
        return cmasks_binary, img1_mask_num, label_map_t1, label_map_t2

    cmasks_cls_dict = {}
    for cls in all_classes:
        keep_mask = [
            (img1_mask_classes[idx] == cls or img2_mask_classes[idx] == cls)
            for idx in change_idx[0]
        ]
        cmasks_cls = np.array([
            mask for keep, mask in zip(keep_mask, cmasks) if keep
        ]) if len(cmasks) > 0 else np.empty((0, *cmasks.shape[1:]))
        cmasks_cls_dict[cls] = cmasks_cls

    return cmasks_cls_dict, cmasks, label_map_t1, label_map_t2
