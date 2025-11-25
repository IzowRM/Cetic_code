from datasets import load_dataset, load_from_disk
from late_fusion_model.fonctions.Data_loader import flatten_text_only , extract_plot, data_loader,general_dataset_transform
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import re
import torch
import torch  # Make sure torch is imported in this file/scope
from late_fusion_model.fonctions.model_loader import Bert_loader

def bert_features(text_tokenizer):  # text_tokenizer is expected to be {'input_ids': tensor, ...}
    print('creation du model')
    # Use from_pretrained for AutoModel
    model = AutoModel.from_pretrained('bert-base-cased')

    # This unpacking now works because text_tokenizer is a mapping (a dictionary)
    outputs = model(**text_tokenizer)

    last_hidden_state = outputs.last_hidden_state
    # Extracts the [CLS] token embedding from the first position (index 0)
    cls_embedding = last_hidden_state[:, 0, :]

    return cls_embedding

def bert_features_fintune(text_tokenizer):
    print('creation du model')
    model = Bert_loader()
    # This unpacking now works because text_tokenizer is a mapping (a dictionary)
    outputs = model(**text_tokenizer)

    last_hidden_state = outputs.last_hidden_state
    # Extracts the [CLS] token embedding from the first position (index 0)
    cls_embedding = last_hidden_state[:, 0, :]

    return cls_embedding