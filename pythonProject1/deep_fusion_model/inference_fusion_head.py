
"""
inference_fusion_head.py
------------------------
Script d'inférence pour ton FusionHead (CLS ViT + CLS BERT).

Pré-requis:
- dataset CLS déjà créé et sauvé: ../data/dataset_cls_bert_vit
  contenant les colonnes: cls_vit, cls_bert, labels
- checkpoint sauvé par savemodel / save_best_checkpoint:
  models_no_finetune/fusion_head.pt (ou autre chemin)

Utilisation:
python inference_fusion_head.py
"""

import os
from pathlib import Path
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from deep_fusion_model.model.FusionHead import FusionHead
from deep_fusion_model.fonctions.tools_for_main import dataset_spliter, dataset_loader
from deep_fusion_model.fonctions.bert_tools import model_bert, bert_features
from deep_fusion_model.fonctions.vit_tools import model_vit, viT_cls
import random


# ====== Labels (dans l'ordre de ton one-hot) ======
LABELS = [
    "drama","comedy","romance","thriller","crime","action","adventure","horror",
    "documentary","mystery","sci-fi","fantasy","family","biography","war","history",
    "music","animation","musical","western","sport","short","film-noir"
]


# ====== Chemin checkpoint par défaut ======
CKPT_PATH = "models_no_finetune/fusion_head.pt"


# ====== Utils affichage ======
def decode_predictions(probs, preds, labels=LABELS, topk=None):
    """
    probs: Tensor [num_labels]
    preds: Tensor [num_labels] (0/1)
    topk: si tu veux afficher top-k probas même si preds=0
    """
    probs_list = probs.tolist()
    preds_list = preds.tolist()

    predicted = []
    for i, lab in enumerate(labels):
        if preds_list[i] == 1:
            predicted.append((lab, probs_list[i]))

    if topk is not None:
        top = sorted([(labels[i], probs_list[i]) for i in range(len(labels))],
                     key=lambda x: x[1], reverse=True)[:topk]
        return predicted, top

    return predicted


def print_predictions(probs, preds, thr, labels=LABELS, topk_if_empty=3):
    predicted, top = decode_predictions(probs, preds, labels=labels, topk=topk_if_empty)

    print(f"\nSeuil utilisé thr={thr}")
    if len(predicted) > 0:
        print("✅ Labels prédits :")
        for lab, p in predicted:
            print(f" - {lab:12s} : {p:.3f}")
    else:
        print("⚠️ Aucun label au-dessus du seuil.")
        print(f"Top-{topk_if_empty} probas :")
        for lab, p in top:
            print(f" - {lab:12s} : {p:.3f}")


# ====== Load checkpoint ======
def load_fusion_model(ckpt_path=CKPT_PATH, device=None):
    """
    Recharge FusionHead + meilleur thr.
    Le checkpoint doit contenir les clés:
      model_state_dict, best_thr, d_in_v, d_in_t, d, num_labels, dropout
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)

    model = FusionHead(
        d_in_v=ckpt["d_in_v"],
        d_in_t=ckpt["d_in_t"],
        d=ckpt["d"],
        num_labels=ckpt["num_labels"],
        p=ckpt["dropout"]
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    best_thr = float(ckpt.get("best_thr", 0.3))

    return model, best_thr


# ====== Prediction avec CLS déjà calculés ======
@torch.no_grad()
def predict_one_from_cls(model, cls_vit, cls_bert, thr, device):
    """
    cls_vit, cls_bert: Tensor [768]
    return probs, preds: Tensor [num_labels]
    """
    model.eval()
    cls_vit  = cls_vit.unsqueeze(0).to(device).float()   # [1,768]
    cls_bert = cls_bert.unsqueeze(0).to(device).float() # [1,768]

    logits = model(cls_vit, cls_bert)   # [1,23]
    probs  = torch.sigmoid(logits)[0]   # [23]
    preds  = (probs >= thr).int()

    return probs.cpu(), preds.cpu()


@torch.no_grad()
def predict_loader(model, loader, thr, device):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []

    for batch in loader:
        cls_vit  = batch["cls_vit"].to(device).float()
        cls_bert = batch["cls_bert"].to(device).float()
        labels   = batch["labels"].cpu()   # vrais labels

        logits = model(cls_vit, cls_bert)
        probs  = torch.sigmoid(logits)
        preds  = (probs >= thr).int()

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_targets.append(labels)

    return (torch.cat(all_probs),
            torch.cat(all_preds),
            torch.cat(all_targets))


# ====== Prediction sur nouveau texte+image brut ======
@torch.no_grad()
def predict_raw(text_ids, image_pil, fusion_model, thr, device):
    """
    text_ids : liste d'input_ids (déjà tokenisés avec bert-base-uncased)
    image_pil : PIL.Image
    """
    # Modèles freezes pour extraire CLS
    bert_model = model_bert().to(device).eval()
    vit_model  = model_vit().to(device).eval()

    cls_bert = bert_features(text_ids, bert_model)  # [1,768]
    cls_vit  = viT_cls(image_pil.convert("RGB"), vit_model)  # [1,768]

    logits = fusion_model(cls_vit.to(device), cls_bert.to(device))
    probs  = torch.sigmoid(logits)[0]
    preds  = (probs >= thr).int()

    return probs.cpu(), preds.cpu()

def decode_one_hot(y, labels=LABELS):
    """
    y: Tensor [23] ou liste length 23 (one-hot / multi-hot)
    retourne la liste des labels actifs
    """
    if torch.is_tensor(y):
        y = y.tolist()
    return [labels[i] for i, v in enumerate(y) if v == 1.0 or v == 1]

# ====== Demo inference sur split test ======
def inference_on_test(n_show=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Checkpoint introuvable: {CKPT_PATH}\n"
            "→ Vérifie que tu as bien sauvegardé le modèle avec savemodel(...) "
            "et que le dossier models_no_finetune/ existe."
        )

    model, best_thr = load_fusion_model(CKPT_PATH, device)
    print("✅ Modèle chargé. thr =", best_thr)

    ds = load_from_disk("../data/dataset_cls_bert_vit")
    splits = dataset_spliter(ds)
    _, _, test_loader = dataset_loader(splits)

    # 🔹 inference sur tout le test
    probs_all, preds_all, targets_all = predict_loader(model, test_loader, best_thr, device)

    print("✅ Inference finie sur test.")
    print("probs shape:", probs_all.shape)
    print("preds shape:", preds_all.shape)

    # 🔹 On choisit n_show indices aléatoires
    N = probs_all.shape[0]
    n_show = min(n_show, N)
    indices = random.sample(range(N), k=n_show)

    for idx in indices:
        print(f"\n== Exemple test index global {idx} ==")

        # labels vrais (one-hot -> noms de genres)
        gt_labels = decode_one_hot(targets_all[idx])
        print("🎯 GT :", gt_labels)

        # prédictions du modèle
        print_predictions(probs_all[idx], preds_all[idx], best_thr)



if __name__ == "__main__":
    # Demo par défaut: inference sur test
    inference_on_test(n_show=5)

    # Exemple pour un raw text+image:
    # from PIL import Image
    # text_ids = [...]  # tes input_ids tokenisés
    # image = Image.open("mon_image.png")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # fusion_model, best_thr = load_fusion_model(CKPT_PATH, device)
    # probs, preds = predict_raw(text_ids, image, fusion_model, best_thr, device)
    # print_predictions(probs, preds, best_thr)
