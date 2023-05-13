import torch
import json
import os
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, GPT2LMHeadModel



os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def load_text(file):
    hypothesis_list = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        for line in data:
            item = json.loads(line)
            text = item["text"]
            hypothesis_list.append(text)
    return hypothesis_list


def load_text_for_sty(file):
    lx_list = []
    jy_list = []
    gs_list = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        for line in data:
            item = json.loads(line)
            text = "".join(item["text"])
            sty = item["style"]
            if sty == "<GS>":
                gs_list.append(text)
            elif sty == "<LX>":
                lx_list.append(text)
            elif sty == "<JY>":
                jy_list.append(text)
            # hypothesis_list.append(text)
    # return hypothesis_list
    return {"<LX>":lx_list, "<JY>":jy_list, "<GS>":gs_list}


def ppl(file, style):
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    ce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").to(device)
    # model.load_state_dict(torch.load("../baselines/Fine-tune-GPT/model/fine-tune-gpt2-chinese/epoch-1-step-3728-loss-2.8955578804016113.pth"), strict=True)
    model.load_state_dict(torch.load("../baselines/Fine-tune-GPT/model/fine-tune-gpt2-chinese/epoch-26-step-50328-loss-0.4100896120071411.pth"), strict=True)
    ppl_list = []
    with torch.no_grad():
        model.eval()
        for i in tqdm(file):
            # label = tokenizer(j, return_tensors='pt').input_ids
            # encoded_input = tokenizer(i, return_tensors='pt')
            encoded_input = tokenizer(i, return_tensors='pt').input_ids.to(device)
            labels = encoded_input.clone()
            # input = [tokenizer.convert_ids_to_tokens(i) for i in a]
            output = model(input_ids=encoded_input)
            shift_logits = output.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = ce_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ppl_list.append(torch.mean(loss).cpu().detach().numpy())
    print(style)
    print("perplexity:", np.exp(np.mean(ppl_list)))

def com_ppl(file):
    # data = load_text(file)
    data = load_text_for_sty(file)
    keys = data.keys()
    for k in keys:
        ppl(data[k], k)


if __name__ == '__main__':
    # file = "../data_ours/auxiliary_data/train.sen.add_index.mask"
    file = "../data_ours/final_data_v1/test.json"
    com_ppl(file)