from datasets import load_from_disk
from Bert_features import bert_features
from ViT_CLS import viT_cls
import torch
def main_bert_vit_no_finetune():
    print('Chargement du ds')
    ds = load_from_disk("../data/dataset_tokenized")
    # Get the input_ids list
    text_list = ds[0]['input_ids']
    print(text_list)

    # 1. Convert the list to a PyTorch tensor.
    # 2. Add a batch dimension using .unsqueeze(0) (since we are only processing one example).
    text_tensor = torch.tensor(text_list).unsqueeze(0)

    # 3. Create the expected dictionary input for the model.
    text_tokenizer = {'input_ids': text_tensor}

    sortie = bert_features(text_tokenizer) # Pass the dictionary
    print('sortie')
    print(sortie.size)

if __name__ == "__main__":
    main_bert_vit_no_finetune()

