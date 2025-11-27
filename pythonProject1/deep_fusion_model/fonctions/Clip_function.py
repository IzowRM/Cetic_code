
def unfreeze_clip_blocks(model, n_last_vision=3, n_last_text=3, unfreeze_projections=True):
    # Tout geler
    for p in model.clip.parameters():
        p.requires_grad = False

    n_vision = len(model.clip.vision_model.encoder.layers)
    n_text   = len(model.clip.text_model.encoder.layers)

    # Sécurité : clamp si tu demandes plus de couches qu'il n'y en a
    n_last_vision = min(n_last_vision, n_vision)
    n_last_text   = min(n_last_text, n_text)

    # Dégeler les derniers blocs vision
    for i in range(n_vision - n_last_vision, n_vision):
        for p in model.clip.vision_model.encoder.layers[i].parameters():
            p.requires_grad = True

    # Dégeler les derniers blocs texte
    for i in range(n_text - n_last_text, n_text):
        for p in model.clip.text_model.encoder.layers[i].parameters():
            p.requires_grad = True

    # Projections finales
    if unfreeze_projections:
        for p in model.clip.visual_projection.parameters():
            p.requires_grad = True
        for p in model.clip.text_projection.parameters():
            p.requires_grad = True

    # (optionnel) affichage
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Params entraînables : {n_trainable:,}/{n_total:,}")