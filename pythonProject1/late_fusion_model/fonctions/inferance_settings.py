import torch
import numpy as np
from transformers import AutoTokenizer
from late_fusion_model.fonctions.model_loader import ask_int

def inference_loop(eval_dataset, tokenizer, model, unique_labels):
    """
    Performs inference for a selected example and displays top predictions.

    Input:
        - User prompt for an integer `n` (example index) via `ask_int()`.
        - `eval_dataset`: A dataset containing text and true labels.
        - `tokenizer`: Tokenizer object for text processing.
        - `model`: Trained PyTorch model for inference.
        - `unique_labels.txt`: List of all unique integer labels.

    Output:
        - Prints the selected example index `n`.
        - Prints the "top_indice update" message.
        - Prints the top `k` predictions (label and probability) for the example.
        - Prints the "True labels:" message followed by the actual labels for the example.
    """
    n = ask_int()
    print(n)
    text = eval_dataset[n]['plot']

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    device = model.device if hasattr(model, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    top_k = 6
    top_indices = np.argsort(probs)[-top_k:][::-1]
    print('top_indice update')
    print(f"Top {top_k} predictions:")
    for i in top_indices:
        print(f"Original Label {unique_labels[i]}: probability={probs[i]:.3f}")
    print('True labels:')
    print(eval_dataset[n]['answer'])

def inf(eval_dataset, model, tokenizer):
    """
    Manages an interactive inference session.

    Output:
        - Calls `inference_loop` multiple times based on user input.
        - Prints "Do you want to continue the test?" prompt.
        - Prints "Invalid response." for invalid user input.
        - Prints "Goodbye!" upon exit (via 'n' or interruption).

    Returns:
        - `True` if the session ends normally (user types 'n').
        - `False` if the session is interrupted (e.g., Ctrl+C).
    """
    with open("../data/unique_labels.txt") as f:
        unique_labels = [line.strip() for line in f]

    inference_loop(eval_dataset, tokenizer, model, unique_labels)

    while True:
        try:
            print("\nDo you want to continue the test?")
            choice = input("Type 'y' for yes or 'n' for no: ").lower().strip()

            if choice in ['y', 'yes', 'o', 'oui', '1']:
                inference_loop(eval_dataset, tokenizer, model, unique_labels)
            elif choice in ['n', 'no', 'non', '0']:
                return True
            else:
                print("  Invalid response. Use 'y' for yes or 'n' for no.")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return False


def inference_loop_taxonomy_dataset(eval_dataset, model,tokenizer):
    """
    Performs inference for a selected example and displays top predictions.

    Input:
        - User prompt for an integer `n` (example index) via `ask_int()`.
        - `eval_dataset`: A dataset containing text and true labels.
        - `tokenizer`: Tokenizer object for text processing.
        - `model`: Trained PyTorch model for inference.
        - `unique_labels.txt`: List of all unique integer labels.

    Output:
        - Prints the selected example index `n`.
        - Prints the "top_indice update" message.
        - Prints the top `k` predictions (label and probability) for the example.
        - Prints the "True labels:" message followed by the actual labels for the example.
    """
    n = ask_int()
    print(n)
    text = eval_dataset[n]['text']
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    device = model.device if hasattr(model, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    top_k = 10
    top_indices = np.argsort(probs)[-top_k:][::-1]
    print('top_indice update')
    print(f"Top {top_k} predictions:")

    for i in top_indices:
        print(f"Predicted Label {i}: probability={probs[i]:.3f}")

    print('True labels:')
    print(eval_dataset[n]['numerical_classification_labels'])

def inf_taxonomy_dataset(eval_dataset, model,tokenizer):
    """
    Manages an interactive inference session.

    Output:
        - Calls `inference_loop` multiple times based on user input.
        - Prints "Do you want to continue the test?" prompt.
        - Prints "Invalid response." for invalid user input.
        - Prints "Goodbye!" upon exit (via 'n' or interruption).

    Returns:
        - `True` if the session ends normally (user types 'n').
        - `False` if the session is interrupted (e.g., Ctrl+C).
    """
    with open("../data/unique_labels.txt") as f:
        unique_labels = [line.strip() for line in f]

    inference_loop_taxonomy_dataset(eval_dataset, model,tokenizer)

    while True:
        try:
            print("\nDo you want to continue the test?")
            choice = input("Type 'y' for yes or 'n' for no: ").lower().strip()

            if choice in ['y', 'yes', 'o', 'oui', '1']:
                inference_loop_taxonomy_dataset(eval_dataset, model,tokenizer)
            elif choice in ['n', 'no', 'non', '0']:
                return True
            else:
                print("  Invalid response. Use 'y' for yes or 'n' for no.")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return False
#<<<<<<<<<<<<<<<<<<<<<<-----------------------INF CNN------------------------->>>>>>>>>>>>>>>

def inf_cnn(logits, thr=0.5):
    """
    logits : tensor de shape (1, num_labels)
    retourne : (labels_actifs, probs_dict)
    """
    with open("../data/unique_labels.txt") as f:
        unique_labels = [line.strip() for line in f]

    probs = torch.sigmoid(logits)[0]  # (num_labels,)
    probs_np = probs.cpu().numpy()

    active = [
        unique_labels[i]
        for i, p in enumerate(probs_np)
        if p >= thr
    ]

    probs_dict = {unique_labels[i]: float(p) for i, p in enumerate(probs_np)}
    return active, probs_dict