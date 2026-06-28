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
    change_mask[change_mask > 0] = 1
    return change_mask


NAME_LIST = [
    "background",
    "building",
]

INPUT_DIR = "<WHU_CD_ROOT>"
TXT_DIR = "<WHU_CD_ROOT>/list/test.txt"
OUTPUT_DIR = "output/WHU/infer_building"
NUM_CLASSES = 2
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
SUPPORT_SET_ROOT = "support_set/WHU-CD"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sam = sam_model_registry[SAM_VERSION](checkpoint=SAM_CHECKPOINT).to(DEVICE)

mp = MaskProposal()
mp.make_mask_generator(
    model=sam,
    points_per_side=8,
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

with open(TXT_DIR, "r") as file:
    target_files = [line.strip() for line in file.readlines()]

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
        change_confidence_threshold=125,
        device=DEVICE,
        model_config=comparator_config,
    )

    cmasks, _, _, _ = identify(
        img1,
        img2,
        cmasks,
        img1_mask_num,
        identifier_model,
        identifier_processor,
        prototype_state=prototype_state,
        device=DEVICE,
        is_instance_class=True,
        num_classes=NUM_CLASSES,
        model_config=identifier_config,
    )

    change_mask = merge_masks(cmasks, img1.shape[:2])
    imsave(os.path.join(OUTPUT_DIR, file_name), (change_mask*255).astype(np.uint8))
