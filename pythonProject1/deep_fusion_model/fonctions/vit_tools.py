from platform import processor

from PIL import ImageOps
import torch
from transformers import ViTModel, ViTImageProcessor, ViTForImageClassification


def pad_to_224_width(img):
    # img est une image PIL
    w, h = img.size          # attention: PIL = (width, height)
    assert h == 224, f"Hauteur attendue = 224, mais trouvé {h}"
    if w == 224:
        return img

    diff = 224 - w
    pad_left = diff // 2
    pad_right = diff - pad_left

    # border = (left, top, right, bottom)
    img_padded = ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=0)
    return img_padded


def model_vit():
    device = "cuda"
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    vit = ViTModel.from_pretrained(model_name)
    vit.to(device)
    return vit, processor

def model_vit_for_finetune():
    num_labels=23
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    id2label = {i: f"classe_{i}" for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


    model.to("cuda")
    model.config.problem_type = "multi_label_classification"
    for param in model.vit.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    return model, processor

def pixel_value_method(exemple):
    image = exemple['image'][0]

    enc = processor(images=image, return_tensors="pt")
    pixel_values = enc["pixel_values"]

    exemple['pixel_values'] = pixel_values.squeeze(0).numpy()

    return exemple


def viT_cls(image, entry):
    model, processor = entry

    inputs = processor(images=image, return_tensors="pt")
    # >>> MOVE TO SAME DEVICE AS MODEL <<<
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding