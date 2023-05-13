import traceback
import sys
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
import os
import numpy as np
from transformers import (
        T5Tokenizer,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        TopKLogitsWarper,
        TemperatureLogitsWarper,
        BeamSearchScorer,
    )
import random
def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    values, _ = torch.topk(logits, k=k)
    min_values = torch.unsqueeze(torch.min(values, -1).values, 1)# values[:, -1, tf.newaxis]
    return torch.where(
        logits < min_values,
        torch.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits,
    )

def gather_nd(x, indices):
    newshape = list(indices.shape[:-1] + x.shape[indices.shape[-1]:]) + [1]
    indices = indices.view(-1, indices.shape[-1]).tolist()
    out = torch.cat([torch.tensor([x.__getitem__(tuple(i))]) for i in indices]).reshape(tuple(newshape))
    return out

def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.size()
    sorted_logits, _ = torch.sort(logits, descending=True, axis=-1)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits), axis=-1)
    cumulative_position = torch.sum((cumulative_probs <= p).to(torch.int32), axis=-1) - 1
    indices = torch.stack([
        torch.arange(0, batch).to(device),
        # number of indices to include
        torch.max(cumulative_position, torch.zeros([batch], dtype=cumulative_position.dtype).to(device)),
    ], axis=-1)
    min_values = gather_nd(sorted_logits, indices).to(device)
    return torch.where(
        logits < min_values,
        torch.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(input_ids, model, max_length, temperature=0.7, top_p=0.9, top_k=40, no_sample=False):
    batch_size = input_ids.size()[0]
    decoder_input_ids = torch.tensor([1 for _ in range(batch_size)])[:, None].to(device)
    # tokens_embed = model.transformer.get_input_embeddings()
    for i in range(max_length):
        logits = model(input_ids, decoder_input_ids=decoder_input_ids)["logits"]
        logits = logits[:, -1, :] / temperature

        if no_sample:
            prev = torch.topk(logits, 1)[1]
        else:
            # logits = top_p_logits(logits, p=top_p)
            logits = top_k_logits(logits, k=top_k)
            probs = torch.nn.functional.softmax(logits)
            prev = torch.multinomial(probs, 1)
        decoder_input_ids = torch.cat([decoder_input_ids, prev], 1)
    return decoder_input_ids

# print(torch.cuda.device_count())
task_name = sys.argv[4]
device = "cuda:%s"%sys.argv[3]
print("using %s"%device)
model_name_path = "./%s/checkpoint-%s"%(sys.argv[1], sys.argv[2])
print(model_name_path)
ckpt_list = False
# ckpt_list = True
PPL, generation = False, True
# PPL, generation = True, False
name = "data"
with open("./%s/%s.source"%(name, task_name), "r") as fin:
    ipt = [line.strip() for line in fin]
with open("./%s/%s.target"%(name, task_name), "r") as fin:
    opt = [line.strip() for line in fin]
# ipt = ['''Empty solutions are of no worth.  # There was a grocery shop in a town. # mouse stood # mice wanted # big fat cat # nice time hunting # mice lived # move freely # mouse slowly stood # cat moves softly''' for _ in range(1000)]
import sys
from unicodedata import category
chrs = (chr(i) for i in range(sys.maxunicode + 1))
punctuation = set(c for c in chrs if category(c).startswith("P"))

def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring.replace("...", "…"):
        inside_code=ord(uchar)
        if uchar in punctuation:
            if inside_code == 32:
                inside_code = 12288
            elif inside_code >= 32 and inside_code <= 126:
                inside_code += 65248
        rstring += chr(inside_code)
    return rstring

def pro(token_list, tokenizer):
    # string = tokenizer.convert_ids_to_tokens(token_list, skip_special_tokens=False)
    string = tokenizer.decode(token_list)
    string = string[:string.find("</s>")].replace("</s>", "").replace("<s>", "").replace("<pad>", "").strip()
    for i in range(100):
        string = string.replace("<extra_id_%d>"%i, "")
    string = "".join(string.strip().split())
    string = strB2Q(string)
    return string



if generation:
    tokenizer = T5Tokenizer.from_pretrained(model_name_path)
    pad_token_id = tokenizer.pad_token_id

    tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_id_%d>"%k for k in range(100)]})

    # model = BartModel.from_pretrained('./bart-base', return_dict=True)
    model = T5ForConditionalGeneration.from_pretrained(model_name_path).to(device)
    # model.eval()
    # print(tokenizer.encode("<extra_id_0>"))
    # print(tokenizer.decode([32099]))
    # print(tokenizer.decode([2]))
    # exit()
    file_out = "./result/%s_%s.txt"%(model_name_path.replace("/", "_").replace(".", ""), task_name)
    print("write to %s"%file_out)
    with open(file_out, "w") as fout:
        batch_size = 16
        st, ed = 0, 0
        all_loss = []
        with torch.no_grad():
            while ed < len(ipt):
                st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
                
                # for i in range(st, ed):
                #     if "<eod>" in ipt[i]:
                #         ipt[i] = "玄幻仙侠<eod>"+ipt[i].strip().split("<eod>")[1]
                #     else:
                #         ipt[i] = "玄幻仙侠<eod>"+ipt[i].strip()
                input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
                try:
                    gen = model.generate(input_ids, do_sample=True, max_length=512, top_k=40, temperature=0.7, decoder_start_token_id=1)# num_beams=4 do_sample=True, top_p=0.9)#, temperature=0.7) decoder_start_token_id=0
                    # gen = model.generate(input_ids, do_sample=False, num_beams=1, max_length=512, decoder_start_token_id=19, early_stopping=True)# num_beams=4 do_sample=True, top_p=0.9)#, temperature=0.7) decoder_start_token_id=0
                    # gen = sample_sequence(input_ids=input_ids, model=model, max_length=512, temperature=0.7, top_k=40)
                except:
                    traceback.print_exc()
                    print("error")
                    gen = []
                    for iipt in ipt[st:ed]:
                        try:
                            input_ids = tokenizer([iipt], return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
                            gen += model.generate(input_ids, do_sample=True, max_length=512, top_k=40, temperature=0.7, decoder_start_token_id=1)# num_beams=4 do_sample=True, top_p=0.9)#, temperature=0.7) decoder_start_token_id=0
                        except:
                            gen.append([0]*10)
                            continue

                    # gen = [[0]*10 for _ in range(batch_size)]
                for ip, op, truth in zip(ipt[st:ed], gen, opt[st:ed]):
                    print(st, ed)
                    print(ip)
                    print(op)
                    print(tokenizer.decode(op))
                    print(truth)
                    fout.write(pro(op, tokenizer)+"\n")
                    print("="*10)
                    # fout.write(pro(ip, tokenizer) + "|||" + pro(op, tokenizer)+"\n")
                    # print(pro(ip, tokenizer))
                    # print(pro(op, tokenizer))
                    # print(truth)
                    # print("="*10)
                    # print(tokenizer.tokenize(truth))
                    # print(tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(truth))))
                    # print("="*5)

if PPL:
    batch_size = 10

    if not ckpt_list:
        model_name_path_dict = {"best": model_name_path}
    else:
        model_name_path_dict = {}
        for _, dd, _ in os.walk(model_name_path):
            for d in dd:
                if d.startswith("val_avg_loss="):
                    model_name_path_dict[d.split("=")[-1].split(".")[0]] = "%s/%s"%(model_name_path, d)
            break
    for name in sorted(model_name_path_dict.keys()):
        try:
            if ckpt_list and int(name) < 35:
                continue
        except:
            continue
        print("loading model %s"%name)
        tmp_model_name_path = model_name_path_dict[name]
        tokenizer = BartTokenizer.from_pretrained(tmp_model_name_path)
        pad_token_id = tokenizer.pad_token_id
        mask_token_id = tokenizer.mask_token_id

        # model = BartModel.from_pretrained('./bart-base', return_dict=True)
        model = BartForConditionalGeneration.from_pretrained(tmp_model_name_path, return_dict=True).to(device)
        model.eval()
        st, ed = 0, 0
        all_loss = []
        while ed < len(ipt):
            st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
            input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=1000).input_ids.to(device)
            with torch.no_grad():
                src_ids = input_ids
                tgt_ids = tokenizer(opt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=1000).input_ids.to(device)
                # tgt_ids = torch.cat([torch.zeros([batch_size, 1], dtype=tgt_ids.dtype), tgt_ids], 1)
                decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
                # print(src_ids, tokenizer.decode(src_ids[0], skip_special_tokens=False))
                # print(tgt_ids, tokenizer.decode(tgt_ids[0], skip_special_tokens=False))
                outputs = model(src_ids, decoder_input_ids=decoder_input_ids, use_cache=False)
                lm_logits = outputs["logits"]
                # print(src_ids.size(), lm_logits.size(), decoder_input_ids.size())


                tmp_batch_size = lm_logits.size()[0]
                pad_pos = torch.eq(tgt_ids, pad_token_id).to(torch.float)
                sen_pos = torch.eq(tgt_ids, mask_token_id).to(torch.float)
                dis_pos = torch.cat([torch.zeros([tmp_batch_size, 1]).to(sen_pos.device), sen_pos[:, :-1]], 1)
                loss_mask = 1 - (pad_pos + sen_pos + dis_pos)
                # Same behavior as modeling_bart.py, besides ignoring pad_token_id
                ce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

                # assert lm_logits.shape[-1] == self.vocab_size
                loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
                loss = torch.sum(loss * loss_mask.view(-1)) / (torch.sum(loss_mask) + 1e-20)
                all_loss.append(loss.cpu().numpy())


                # # Same behavior as modeling_bart.py, besides ignoring pad_token_id
                # ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

                # # assert lm_logits.shape[-1] == self.vocab_size
                # loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
                # all_loss.append(loss.cpu().numpy())
        print(name, "perplexity:", np.exp(np.mean(all_loss)))

    # all_loss_dict = {}
    # for i, l in enumerate(all_loss):
    #     all_loss_dict[i] = l
    # idx_list = sorted(all_loss_dict, key=all_loss_dict.get, reverse=True)
    # with open("./ppl.txt", "w") as fout:
    #     for idx in idx_list:
    #         fout.write("%d|||%.4f|||%s|||%s\n"%(idx, all_loss_dict[idx], ipt[idx], opt[idx]))