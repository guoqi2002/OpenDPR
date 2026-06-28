import os
import sys

import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))


@torch.no_grad()
def get_model_and_processor(
    model_type,
    device,
    model_config=None,
    processor_config=None,
):
    if model_type == "DINO":
        processor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        # Hugging Face / online loading option:
        # model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(device)

        # Default: load the local ViT-B/16 structure and local checkpoint.
        from DINO.vision_transformer import vit_base
        model = vit_base(patch_size=16)
        state_dict = torch.load(
            "<DINO_V1_CHECKPOINT>",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)

    elif model_type == "DINOv2":
        # Hugging Face / online loading option:
        # processor = AutoImageProcessor.from_pretrained(
        #     "facebook/dinov2-base",
        #     do_resize=False,
        #     do_center_crop=False,
        # )
        # model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

        # Default: load the processor and model from a local checkpoint.
        local_path = "<DINOV2_MODEL_DIR>"
        processor = AutoImageProcessor.from_pretrained(
            local_path,
            do_resize=False,
            do_center_crop=False,
        )
        model = AutoModel.from_pretrained(local_path).to(device)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model, processor
