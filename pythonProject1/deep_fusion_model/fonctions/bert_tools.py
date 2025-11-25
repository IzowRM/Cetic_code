from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
import torch
def model_bert():
    model = AutoModel.from_pretrained('bert-base-uncased')
    device = "cuda"
    model.to(device)
    return model.eval()

def prepare_bert_inputs(text):
    text_tensor = torch.tensor(text).unsqueeze(0)
    attention_mask = (text_tensor != 0).long()
    return {'input_ids': text_tensor,
                      'attention_mask': attention_mask
                      }

def bert_loader():

    model = BertForSequenceClassification.from_pretrained("../data/BertModel/model")
    model.config.problem_type = "multi_label_classification"
    device = "cuda"
    model.to(device)
    return model.eval()

def bert_features(text, model):
    inputs = prepare_bert_inputs(text)
    # >>> MOVE TO SAME DEVICE AS MODEL <<<
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    last_hidden_state = model(**inputs).last_hidden_state
    cls_embedding = last_hidden_state[:, 0, :]
    return cls_embedding


def bert_finetune_features(text, model):
    inputs = prepare_bert_inputs(text)

    # même device que le modèle (plus robuste que model.device)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
            # on passe par le backbone BERT interne
        outputs = model.bert(**inputs, return_dict=True)
        last_hidden_state = outputs.last_hidden_state  # [1, T, H]
        cls_embedding = last_hidden_state[:, 0, :]  # [1, H]

    return cls_embedding
