import re
from math import ceil
from PIL import Image
import torch.nn.functional as F
from datasets import Features, Sequence, Value
from datasets import Dataset as HFDataset, DatasetDict as HFDatasetDict
from torch.utils.data import DataLoader
import torch
from torchvision import transforms

#Supprime les photos et conserve les textes.
def flatten_text_only(example):

    import json
    msgs = example.get("messages", []) or []

    user_msgs = [m.get("content", "") for m in msgs if m.get("role") == "user"]
    assistant_msgs = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]

    prompt = "\n\n".join([u for u in user_msgs if isinstance(u, str)]) if user_msgs else ""
    answer = assistant_msgs[-1] if assistant_msgs else ""

    return {
        "prompt": prompt,
        "answer": answer,
    }

def extract_plot(text: str) -> str:
    PATTERN = re.compile(r'(?is)\bplot\s*:\s*(.*?)(?=\bnote\s+that\b|answer\s*:|$)')
    if text is None:
        return None
    m = PATTERN.search(text)
    if not m:
        return ""  # ou text.strip() si tu préfères conserver l'original quand Pas de "Plot:"
    plot = m.group(1).strip()
    # petites normalisations
    plot = re.sub(r'\s+\n', '\n', plot)
    plot = re.sub(r'\s{2,}', ' ', plot)
    return plot

#Remplace de les one-hots
def data_loader(ds_flat):
    LABELS = [
        "drama", "comedy", "romance", "thriller", "crime", "action", "adventure", "horror",
        "documentary", "mystery", "sci-fi", "fantasy", "family", "biography", "war", "history",
        "music", "animation", "musical", "western", "sport", "short", "film-noir"
    ]
    label2id = {lab: i for i, lab in enumerate(LABELS)}

    ALIASES = {
        "documentry": "documentary",
        "science fiction": "sci-fi",
        "sci fi": "sci-fi",
        "film noir": "film-noir",
        "westerns": "western",
        "sports": "sport",
    }

    def norm_token(t: str) -> str:
        t = t.lower()
        t = re.sub(r"[\s\-]+", " ", t).strip()
        t = ALIASES.get(t, t)
        t = t.replace("sci fi", "sci-fi")
        return t

    def parse_answer(ans: str):
        if ans is None:
            return set()
        toks = re.split(r"[;,]", ans)
        toks = [norm_token(x) for x in toks if x.strip()]
        return set(t for t in toks if t in label2id)

    def to_multi_hot_float(labels_set):
        vec = [0.0] * len(LABELS)
        for lab in labels_set:
            vec[label2id[lab]] = 1.0
        return vec

    def mapper(batch):
        answers = batch["answer"]
        return {"labels": [to_multi_hot_float(parse_answer(a)) for a in answers]}

    new_features = ds_flat.features.copy()
    new_features["labels"] = Sequence(feature=Value("float32"), length=len(LABELS))

    ds_flat = ds_flat.map(mapper, batched=True, features=new_features)

    print(type(ds_flat[0]["labels"][0]), ds_flat.features["labels"])

    return ds_flat

#Creation du one-hot
def data_loader(exemple):
    LABELS = [
        "drama", "comedy", "romance", "thriller", "crime", "action", "adventure", "horror",
        "documentary", "mystery", "sci-fi", "fantasy", "family", "biography", "war", "history",
        "music", "animation", "musical", "western", "sport", "short", "film-noir"
    ]
    label2id = {lab: i for i, lab in enumerate(LABELS)}

    ALIASES = {
        "documentry": "documentary",
        "science fiction": "sci-fi",
        "sci fi": "sci-fi",
        "film noir": "film-noir",
        "westerns": "western",
        "sports": "sport",
    }

    def norm_token(t: str) -> str:
        t = t.lower()
        t = re.sub(r"[\s\-]+", " ", t).strip()
        t = ALIASES.get(t, t)
        t = t.replace("sci fi", "sci-fi")
        return t

    def parse_answer(ans: str):
        if ans is None:
            return set()
        toks = re.split(r"[;,]", ans)
        toks = [norm_token(x) for x in toks if x.strip()]
        return set(t for t in toks if t in label2id)

    def to_multi_hot_float(labels_set):
        vec = [0.0] * len(LABELS)
        for lab in labels_set:
            vec[label2id[lab]] = 1.0
        return vec

    def mapper(batch):
        answers = batch["answer"]
        return {"labels": [to_multi_hot_float(parse_answer(a)) for a in answers]}
    labels = mapper(exemple)
    return labels


def hot_to_labels(hot):
    LABELS = [
        "drama", "comedy", "romance", "thriller", "crime", "action", "adventure", "horror",
        "documentary", "mystery", "sci-fi", "fantasy", "family", "biography", "war", "history",
        "music", "animation", "musical", "western", "sport", "short", "film-noir"
    ]
    idx = (hot == 1).nonzero(as_tuple=True)[0]
    actifs = [LABELS[i] for i in (hot == 1).nonzero(as_tuple=True)[0].tolist()]
    print(actifs)


def data_cnn_treatment (ds_all):
    import os, re

    LABELS = [
        "drama", "comedy", "romance", "thriller", "crime", "action", "adventure", "horror",
        "documentary", "mystery", "sci-fi", "fantasy", "family", "biography", "war", "history",
        "music", "animation", "musical", "western", "sport", "short", "film-noir"
    ]
    label2id = {lab: i for i, lab in enumerate(LABELS)}

    ALIASES = {
        "documentry": "documentary",
        "science fiction": "sci-fi",
        "sci fi": "sci-fi",
        "film noir": "film-noir",
        "westerns": "western",
        "sports": "sport",
    }

    _SPLIT = re.compile(r"\s*(,|/|\||;|&|\band\b|\bet\b)\s*", re.IGNORECASE)

    def _norm(tok: str) -> str:
        t = tok.strip().lower()
        t = re.sub(r"\s+", " ", t).replace("–", "-").replace("—", "-")
        return ALIASES.get(t, t)

    def _encode_one(answer: str):
        if not isinstance(answer, str):
            answer = "" if answer is None else str(answer)
        parts = [p for p in _SPLIT.split(answer) if p and not _SPLIT.fullmatch(p)]
        found = set()
        for p in (parts if parts else [answer]):
            n = _norm(p)
            if n in label2id:
                found.add(n)
        idxs = sorted(label2id[n] for n in found)
        one_hot = [0.0] * len(LABELS)
        for i in idxs:
            one_hot[i] = 1.0
        return idxs, one_hot

    def remap_batch(batch):
        new_idx, new_oh = [], []
        for ans in batch["answer"]:
            idxs, oh = _encode_one(ans)
            new_idx.append(idxs)
            new_oh.append(oh)
        return {"labels_idx": new_idx, "labels": new_oh}

    # --- Remap en forçant l'écriture du cache dans un dossier écrivable ---
    cache_dir = "/kaggle/working/hf-indices"
    os.makedirs(cache_dir, exist_ok=True)

    ds_all_23 = ds_all.map(
        remap_batch,
        batched=True,
        desc="Remap to 23 labels",
        load_from_cache_file=False,  # évite de réutiliser un cache ancien
        cache_file_name=os.path.join(cache_dir, "remap_23.arrow"),  # <-- important
    )

    # Vérif
    ex = ds_all_23[0]
    print("answer:", ex["answer"])
    print("labels_idx:", ex["labels_idx"])
    print("one-hot length:", len(ex["labels"]))
    print("active labels:", [LABELS[i] for i in ex["labels_idx"]])

    return ds_all_23


# <<<<<<<<<------------------cnn-------------------->>>>>>>>>
def _to_multiple(x, m=32):
    return int(ceil(x / m) * m)


def collate_resize(batch, target_h=224, multiple=32):
    imgs, labs = [], []

    # 1) resize en gardant le ratio (hauteur fixe)
    widths = []
    for b in batch:
        x = b["pixel_values"]
        C, H, W = x.shape
        scale = target_h / float(H)
        new_w = max(1, int(round(W * scale)))
        x_res = F.interpolate(
            x.unsqueeze(0), size=(target_h, new_w),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        imgs.append(x_res)
        labs.append(b["labels"])
        widths.append(new_w)

    # 2) pad à droite pour aligner toutes les largeurs du batch
    max_w = _to_multiple(max(widths), multiple)
    imgs_pad = []
    for x_res, w in zip(imgs, widths):
        pad_right = max_w - w
        # pad format: (left, right, top, bottom)
        x_pad = F.pad(x_res, (0, pad_right, 0, 0), mode="constant", value=0.0)
        imgs_pad.append(x_pad)

    imgs_pad = torch.stack(imgs_pad, dim=0)      # (B,C,target_h,max_w)

    # 3) labels -> tensor
    if isinstance(labs[0], torch.Tensor):
        labs = torch.stack(labs, dim=0)
    else:
        labs = torch.tensor(labs)

    return {"pixel_values": imgs_pad, "labels": labs}


def flatten_text_only(example):

    import json
    Image.MAX_IMAGE_PIXELS = 200_000_000
    msgs = example.get("messages", []) or []

    user_msgs = [m.get("content", "") for m in msgs if m.get("role") == "user"]
    assistant_msgs = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]

    prompt = "\n\n".join([u for u in user_msgs if isinstance(u, str)]) if user_msgs else ""
    answer = assistant_msgs[-1] if assistant_msgs else ""

    return {
        "prompt": prompt,
        "answer": answer,
    }

def general_dataset_transform(example):
    import json
    msgs = example.get("messages", []) or []
    image = example.get("images")
    user_msgs = [m.get("content", "") for m in msgs if m.get("role") == "user"]
    assistant_msgs = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]

    prompt = "\n\n".join([u for u in user_msgs if isinstance(u, str)]) if user_msgs else ""
    answer = assistant_msgs[-1] if assistant_msgs else ""
    return {
        "prompt": prompt,
        "answer": answer,
         "image":image,
    }


def hf_transform(batch):
    tfm = transforms.Compose([
        transforms.ToTensor(),  # -> float32, [0,1], (C,H,W)
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    imgs = batch["images_json"]
    xs = [tfm(img.convert("RGB")) for img in imgs]
    ys = [torch.tensor(y, dtype=torch.float32) for y in batch["labels"]]
    return {"pixel_values": xs, "labels": ys}


def data_cnn_spliter(ds_all):
    cache_dir = "../data/CNN_model/hf-indices"
    if isinstance(ds_all, HFDatasetDict):
        base = ds_all["train"]
    else:
        base = ds_all

    # 2) Split 80/20 en forçant les chemins des indices
    tmp = base.train_test_split(
        test_size=0.2, seed=42,
        train_indices_cache_file_name=f"{cache_dir}/s1-train.arrow",
        test_indices_cache_file_name=f"{cache_dir}/s1-temp.arrow",
    )

    # 3) Re-split du 20% en 10% val / 10% test (toujours avec fichiers d’indices)
    tmp2 = tmp["test"].train_test_split(
        test_size=0.5, seed=42,
        train_indices_cache_file_name=f"{cache_dir}/s2-val.arrow",
        test_indices_cache_file_name=f"{cache_dir}/s2-test.arrow",
    )

    # 4) Regroupe

    ds = HFDatasetDict({
        "train": tmp["train"],
        "val": tmp2["train"],
        "test": tmp2["test"],
    })

    return ds
