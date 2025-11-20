# main_late_fusion.py

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from late_fusion_model.fonctions.model_loader import Bert_loader, cnn_loader
from late_fusion_model.fonctions.inferance_settings import inf_cnn  # (import non utilisé, tu peux le retirer si tu veux)
from datasets import load_dataset, load_from_disk  # ou load_from_disk selon ton cas


def prepare_image(image_input, device):
    """
    Charge une image et applique les normalisations CNN.
    Accepte soit :
    - un chemin (str) vers une image
    - un objet PIL.Image.Image
    - un dict avec une clé 'path'
    """

    # Récupérer une PIL image
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, dict) and "path" in image_input:
        img = Image.open(image_input["path"]).convert("RGB")
    else:
        raise ValueError(
            f"Type d'entrée image non géré : {type(image_input)} "
            "(attendu: str, PIL.Image.Image, ou dict avec 'path')."
        )

    tfm = transforms.Compose([
        transforms.ToTensor(),  # -> float32, [0,1], (C,H,W)
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    x = tfm(img)              # (C,H,W)
    x = x.unsqueeze(0).to(device)  # (1,C,H,W)
    return x


def load_labels():
    """
    Charge la liste des labels dans le même ordre que pour BERT et le CNN.
    """
    with open("../data/unique_labels.txt") as f:
        unique_labels = [line.strip() for line in f]
    return unique_labels


def late_fusion_predict(text, image_input, alpha=0.5, thr=0.5, top_k=6):
    """
    Combine les prédictions BERT (texte) et CNN (image).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Labels
    unique_labels = load_labels()
    num_labels = len(unique_labels)

    # 2) Modèles
    bert_model = Bert_loader()
    bert_model.to(device)
    bert_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("../data/BertModel/model")

    cnn_model = cnn_loader(device)
    cnn_model.eval()

    # 3) Texte -> BERT
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs_bert = bert_model(**inputs)
        logits_bert = outputs_bert.logits              # (1, num_labels)
        probs_bert = torch.sigmoid(logits_bert)        # (1, num_labels)

    # 4) Image -> CNN
    x_img = prepare_image(image_input, device)

    with torch.no_grad():
        logits_cnn = cnn_model(x_img)                  # (1, num_labels)
        probs_cnn = torch.sigmoid(logits_cnn)          # (1, num_labels)

    # 5) Fusion
    probs_fused = alpha * probs_bert + (1.0 - alpha) * probs_cnn   # (1, num_labels)
    probs_fused_flat = probs_fused.squeeze(0)                      # (num_labels,)

    # 6) Top-k
    top_probs, top_idx = torch.topk(probs_fused_flat, k=top_k)
    print("\n=== Top prédictions fusionnées (BERT + CNN) ===")
    for p, idx in zip(top_probs, top_idx):
        label = unique_labels[idx.item()]
        p_fused = p.item()
        p_bert = probs_bert[0, idx].item()
        p_cnn = probs_cnn[0, idx].item()
        print(f"{label:12s} | fused={p_fused:.3f} | bert={p_bert:.3f} | cnn={p_cnn:.3f}")

    # 7) Labels actifs au-dessus du seuil
    active_fused = [
        unique_labels[i]
        for i in range(num_labels)
        if probs_fused_flat[i].item() >= thr
    ]
    probs_fused_dict = {
        unique_labels[i]: float(probs_fused_flat[i].item())
        for i in range(num_labels)
    }

    print("\nLabels actifs (p_fused >= {:.2f}) :".format(thr))
    print(active_fused)

    return active_fused, probs_fused_dict


def main():
    print("=== Fusion BERT (texte) + CNN (image) ===")

    # TODO : adapte selon comment tu as sauvegardé ton dataset
    # main_CNN.py
    ds = load_from_disk("../data/ds_flat")

    # Si tu as utilisé save_to_disk :
    # from datasets import load_from_disk
    # ds = load_from_disk("../data/ds_flat")

    sample = ds[2]
    print(sample)
    text = sample["plot"]
    # text = 'Mild mannered businessman Anthony Wongs life is shattered when his pregnant wife is run over by a busy taxi driver. This and another incident with a sleazy cab driver causes Wong to go on a mission to kill bad taxi drivers.'
    image = sample["image"]

    late_fusion_predict(text, image, alpha=0.5, thr=0.5, top_k=6)


if __name__ == "__main__":
    main()
