import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
from tqdm import tqdm
from skimage.io import imread, imsave
from segment_anything import sam_model_registry
from segment_anything.utils.amg import rle_to_mask

torch.set_num_threads(4)

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(parent_directory)

from dynamic_earth.mask_generator import MaskProposal
from dynamic_earth.identifier.prototype_retrieval import build_prototype_set, identify
from dynamic_earth.comparator.bi_match import bitemporal_match
from dynamic_earth.utils import get_model_and_processor


def merge_masks(change_masks, shape):
    """Merge multiple instance masks into a single binary mask."""
    if len(change_masks) == 0:
        return np.zeros((shape[0], shape[1]), dtype=np.uint8)

    change_mask = np.sum(change_masks, axis=0).astype(np.uint8)
    change_mask[change_mask > 0] = 255
    return change_mask


def colorize_label_map(label_map, palette):
    color_map = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        color_map[label_map == class_idx] = color
    return color_map


NAME_LIST = [
    "water",
    "grass",
    "building",
    "greenhouse",
    "road",
    "bridge",
    "bareland",
    "woodland",
]

INPUT_DIR = "<HIUCD_ROOT>/test/image/"
OUTPUT_DIR = "output/HIUCD/infer_all"
LABEL_DIR = "<HIUCD_ROOT>/test/mask/"
NUM_CLASSES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM_VERSION = "vit_h"
SAM_CHECKPOINT = "<SAM_CHECKPOINT>"
COMPARATOR_MODEL_TYPE = "DINOv2"
COMPARATOR_FEATURE_DIM = 768
COMPARATOR_PATCH_SIZE = 14
IDENTIFIER_MODEL_TYPE = "DINOv2"
IDENTIFIER_FEATURE_DIM = 768
IDENTIFIER_PATCH_SIZE = 14
MAX_NUM_PROTOTYPES = 5
SUPPORT_SET_ROOT = "support_set/Hi-UCD"

PALETTE = np.array(
    [
        [0, 152, 254],
        [202, 254, 122],
        [254, 0, 26],
        [228, 0, 254],
        [254, 230, 0],
        [254, 180, 196],
        [174, 122, 254],
        [26, 254, 0],
        [0, 0, 0],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

sam = sam_model_registry[SAM_VERSION](checkpoint=SAM_CHECKPOINT).to(DEVICE)

mp = MaskProposal()
mp.make_mask_generator(
    model=sam,
    points_per_side=32,
    points_per_batch=64,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.95,
    stability_score_offset=0.9,
    box_nms_thresh=0.7,
    min_mask_region_area=0,
)
mp.set_hyperparameters()

comparator_model, comparator_processor = get_model_and_processor(COMPARATOR_MODEL_TYPE, DEVICE)
comparator_config = {
    "model_type": COMPARATOR_MODEL_TYPE,
    "feature_dim": COMPARATOR_FEATURE_DIM,
    "patch_size": COMPARATOR_PATCH_SIZE,
}

identifier_model, identifier_processor = get_model_and_processor(IDENTIFIER_MODEL_TYPE, DEVICE)
identifier_config = {
    "model_type": IDENTIFIER_MODEL_TYPE,
    "feature_dim": IDENTIFIER_FEATURE_DIM,
    "patch_size": IDENTIFIER_PATCH_SIZE,
    "max_num_prototypes": MAX_NUM_PROTOTYPES,
}
prototype_state = build_prototype_set(
    encoder_model=identifier_model,
    encoder_processor=identifier_processor,
    encoder_config=identifier_config,
    prototype_classes=NAME_LIST,
    support_set_root=SUPPORT_SET_ROOT,
    device=DEVICE,
)

target_files = sorted(os.listdir(os.path.join(INPUT_DIR, "A")))

for file_name in tqdm(target_files, desc="Processing", unit="iteration"):
    img1_path = os.path.join(INPUT_DIR, "A", file_name)
    img2_path = os.path.join(INPUT_DIR, "B", file_name)

    img1 = imread(img1_path)
    img2 = imread(img2_path)

    masks, img1_mask_num = mp.forward(img1, img2)
    masks = np.array([rle_to_mask(rle).astype(bool) for rle in masks["rles"]])

    cmasks, img1_mask_num = bitemporal_match(
        img1,
        img2,
        masks,
        comparator_model,
        comparator_processor,
        img1_mask_num,
        change_confidence_threshold=145,
        device=DEVICE,
        model_config=comparator_config,
    )

    cmasks_cls_dict, cmasks, label_map_t1, label_map_t2 = identify(
        img1,
        img2,
        cmasks,
        img1_mask_num,
        identifier_model,
        identifier_processor,
        prototype_state=prototype_state,
        device=DEVICE,
        is_instance_class=False,
        num_classes=NUM_CLASSES,
        model_config=identifier_config,
    )

    label1_path = os.path.join(LABEL_DIR, "A_scd_gt", file_name)
    label2_path = os.path.join(LABEL_DIR, "B_scd_gt", file_name)
    label1 = imread(label1_path)
    label2 = imread(label2_path)

    for class_id, class_masks in cmasks_cls_dict.items():
        class_change_mask = merge_masks(class_masks, img1.shape[:2])
        class_change_mask[label1 == 10] = 128

        class_output_dir = os.path.join(OUTPUT_DIR, f"class_{class_id}")
        os.makedirs(class_output_dir, exist_ok=True)

        save_path = os.path.join(class_output_dir, file_name)
        imsave(save_path, class_change_mask)

    change_mask = merge_masks(cmasks, img1.shape[:2])
    all_output_dir = os.path.join(OUTPUT_DIR, "all")

    t1_dir_color = os.path.join(all_output_dir, "T1_color")
    t2_dir_color = os.path.join(all_output_dir, "T2_color")
    change_dir_color = os.path.join(all_output_dir, "Change_color")
    t1_dir = os.path.join(all_output_dir, "T1")
    t2_dir = os.path.join(all_output_dir, "T2")

    os.makedirs(t1_dir_color, exist_ok=True)
    os.makedirs(t2_dir_color, exist_ok=True)
    os.makedirs(change_dir_color, exist_ok=True)
    os.makedirs(t1_dir, exist_ok=True)
    os.makedirs(t2_dir, exist_ok=True)

    label_map_t1[label1 == 10] = 9
    label_map_t2[label1 == 10] = 9
    change_mask[label1 == 10] = 128

    imsave(os.path.join(t1_dir, file_name), label_map_t1.astype(np.uint8))
    imsave(os.path.join(t2_dir, file_name), label_map_t2.astype(np.uint8))
    imsave(os.path.join(t1_dir_color, file_name), colorize_label_map(label_map_t1, PALETTE))
    imsave(os.path.join(t2_dir_color, file_name), colorize_label_map(label_map_t2, PALETTE))
    imsave(os.path.join(change_dir_color, file_name), change_mask.astype(np.uint8))
