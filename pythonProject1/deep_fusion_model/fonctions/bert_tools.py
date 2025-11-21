from transformers import AutoTokenizer, AutoModel
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



def bert_features(text, model):
    inputs = prepare_bert_inputs(text)
    # >>> MOVE TO SAME DEVICE AS MODEL <<<
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    last_hidden_state = model(**inputs).last_hidden_state
    cls_embedding = last_hidden_state[:, 0, :]
    return cls_embedding
