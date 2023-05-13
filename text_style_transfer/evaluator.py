from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch
from yaml import load
from data_set import Data_Encoder_eval
import json
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import jieba
from tqdm import tqdm
import numpy as np
from transformers import T5Tokenizer
import torch.nn.functional as F
from bert_score import score, BERTScorer
from transformers import BertTokenizer, GPT2LMHeadModel
from modeling_t5 import Style_Classifier
from nltk import ngrams
from zhon.hanzi import punctuation

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class StyleClassifier_Config(object):
    pre_trained_t5 = "pretrained_model/LongLM-base"
    pretrained_sc = "pretrained_model/pre-trained_style_classifier/epoch-9-step-4660-loss-1.873672408692073e-05.pth"
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    max_length = 512
    in_size = 768
    style_num = 3


def style_classify(eval_file):
    config = StyleClassifier_Config()
    dataset = Data_Encoder_eval(eval_file)
    data_generator = DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        # create model
        tokenizer = T5Tokenizer.from_pretrained("pretrained_model/LongLM-base")
        model = Style_Classifier(config).to(config.device)
        model.load_state_dict(torch.load(config.pretrained_sc), strict=True)
        model.eval()

        # if not os.path.exists(result_dir):
        #     os.mkdir(result_dir)
        result = eval_file + ".style_pred"
        print('begin predicting')

        y_pred = []
        y_true = []

        with open(result, 'w') as f:
            for i, (batch_text, label) in enumerate(tqdm(data_generator)):

                ids = tokenizer(batch_text,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=config.max_length).input_ids.to(config.device)

                labels = label.cpu().numpy()

                logits = model(ids)

                p = F.softmax(logits, dim=-1).cpu().numpy()
                pre_label = np.argmax(p, axis=-1)


                for res, i_label in zip(pre_label, labels):
                    y_pred.append(int(res))
                    y_true.append(int(i_label))
                    f.write(str(res) + '\n')
    acc = accuracy_score(y_true, y_pred)
    # save_path = result.replace('txt', 'eval')
    save_path = result + ".eval"
    with open(save_path, 'w') as f:
        f.write("acc : {}".format(acc))
    print("acc : {}".format(acc))

def proline(line):
    return " ".join([w for w in jieba.cut("".join(line.strip().split()))])

class Bleu_Metric(object):
    def __init__(self, hypothesis_file, reference_file):
        self.hypothesis_list = []
        self.refernce_list = []
        with open(hypothesis_file, 'r') as fin:
            data = fin.readlines()
            for line in data:
                item = json.loads(line)
                text = item["text"]
                self.hypothesis_list.append(text)
        with open(reference_file, 'r') as fin:
            data = fin.readlines()
            for line in data:
                item = json.loads(line)
                text = item["text"]
                self.refernce_list.append(text)

    def bleu(self):
        """
        compute rouge score
        Args:
            data (list of dict including reference and candidate):
        Returns:
                res (dict of list of scores): rouge score
        """

        res = {}
        for i in range(1, 5):
            res["bleu-%d" % i] = []

        for origin_reference, origin_candidate in tqdm(zip(self.refernce_list, self.hypothesis_list)):
            # origin_candidate = tmp_data['candidate']
            # origin_reference = tmp_data['reference']
            origin_reference = proline(origin_reference)
            origin_candidate = proline(origin_candidate)
            assert isinstance(origin_candidate, str)
            if not isinstance(origin_reference, list):
                origin_reference = [origin_reference]

            for i in range(1, 5):
                res["bleu-%d" % i].append(sentence_bleu(references=[r.strip().split() for r in origin_reference],
                                                        hypothesis=origin_candidate.strip().split(),
                                                        weights=tuple([1. / i for j in range(i)])))

        for key in res:
            res[key] = np.mean(res[key])

        print(res)

        return res

def load_text(file):
    hypothesis_list = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        for line in data:
            item = json.loads(line)
            text = item["text"]
            hypothesis_list.append(text)
    return hypothesis_list

def bert_sco(cands, refs):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # scorer = BERTScorer(lang="zh", rescale_with_baseline=True)
    cand_list = load_text(cands)
    refs_list = load_text(refs)
    (P, R, F), hashname = score(cand_list, refs_list, lang="zh", return_hash=True)
    print(f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}")


def com_ppl(file):
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    ce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").to(device)
    model.load_state_dict(torch.load("baselines/Fine-tune-GPT/model/fine-tune-gpt2-chinese/epoch-1-step-3728-loss-2.8955578804016113.pth"), strict=True)
    data = load_text(file)
    ppl_list = []
    with torch.no_grad():
        model.eval()
        for i in tqdm(data):
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
    print(file)
    print("perplexity:", np.exp(np.mean(ppl_list)))


def repetition_distinct(eval_data):
    result = {}
    data = load_text(eval_data)
    for i in range(1, 5):
        all_ngram, all_ngram_num = {}, 0.
        for k, tmp_data in enumerate(data):
            ngs = ["_".join(c) for c in ngrams(tmp_data, i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
        result["distinct-%d"%i] = len(all_ngram) / float(all_ngram_num)

    print(result)
    # return result


def target_score(eval_file):
    config = StyleClassifier_Config()
    dataset = Data_Encoder_eval(eval_file)
    data_generator = DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        # create model
        tokenizer = T5Tokenizer.from_pretrained("pretrained_model/LongLM-base")
        model = Style_Classifier(config).to(config.device)
        model.load_state_dict(torch.load(config.pretrained_sc), strict=True)
        model.eval()

        # if not os.path.exists(result_dir):
        #     os.mkdir(result_dir)
        result = eval_file + ".target_score"
        print('begin predicting')

        y_pred = []
        y_true = []

        with open(result, 'w') as f:
            for i, (batch_text, label) in enumerate(tqdm(data_generator)):

                ids = tokenizer(batch_text,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=config.max_length).input_ids.to(config.device)

                labels = label.cpu().numpy()

                logits = model(ids)

                p = F.softmax(logits, dim=-1).cpu().numpy()
                # pre_label = np.argmax(p, axis=-1)


                for sco in p:
                    f.write(str(sco[0]) + "  " + str(sco[1]) + "  " + str(sco[2]) + '\n')
    # acc = accuracy_score(y_true, y_pred)
    # save_path = result.replace('txt', 'eval')
    # save_path = result + ".eval"
    # with open(save_path, 'w') as f:
    #     f.write("acc : {}".format(acc))
    # print("acc : {}".format(acc))

    
    
    

def r_acc_and_a_acc(eval_file, ref, target_style_index):
    # target_style_index = 0
    score_file = eval_file + ".target_score"
    ref_file = ref + ".target_score"
    
    if not os.path.exists(score_file):
        target_score(eval_file)
    if not os.path.exists(ref_file):
        target_score(ref)
    
    
    def load_score(file, target_style_index):
        with open(file, 'r') as f:
            data = f.readlines()
            score = [float(i.split()[target_style_index]) for i in data]

        return np.array(score)
    
    scores = load_score(score_file, target_style_index)
    ref_score = load_score(ref_file, target_style_index)
    
    r_acc = np.mean(scores>ref_score)
    # a_acc = np.mean(scores>0.5)
    
    res = score_file + ".r_acc"
    with open(res, 'w') as f:
        f.write("r_acc:{}".format(r_acc))
    print("r_acc:{}".format(r_acc))

def copy_func(x, y):
    punctuation_str = punctuation
    x_texts = load_text(x)
    y_texts = load_text(y)
    copy_index = []
    for i, j in zip(x_texts, y_texts):
        # res = i.rstrip(punctuation_str)
        # res_index = i.rstrip(punctuation_str) == j.rstrip(punctuation_str)
        res_index = i == j
        copy_index.append(res_index)
    
    index = np.mean(copy_index)
    res = x + ".copy_index"
    with open(res, 'w') as f:
        f.write("copy:{}".format(index))
    print("copy:{}".format(index))
    # return index
    
    

if __name__ == '__main__':
    
    # dir_list = ["Keywords2story", "reverse_attention", "Stage_2", "Style_LongLM", 
    #             "ablation/multi_sen", "ablation/one_stage", "ablation/sen_loss", "ablation/style_classifier"]
    dir_list = ["pred_add_sentype/one_stage_transfer_1119_e_41/stage_1"]
    # file = "test_sen_mse_add_mask_in_input"
    file = "test_one_stage_transfer"
    
    for dir in dir_list:
        hypothesis_file_0 = "./{}/{}.0".format(dir, file)
        hypothesis_file_1 = "./{}/{}.1".format(dir, file)
        hypothesis_file_list = [hypothesis_file_0, hypothesis_file_1]
        reference_file = "./data_ours/final_data/test.json"
        for i in hypothesis_file_list:
            print(i)
            # style_classify(i)
            # target_style_index = i.split(".")[-1]
            # r_acc_and_a_acc(i, reference_file, int(target_style_index))
            # # # com_ppl(i)# need activate 4_ppl
            # # # # repetition_distinct(i)

            # # copy_func(i, reference_file)
            
            bleu_score = Bleu_Metric(hypothesis_file=i, reference_file=reference_file).bleu()
            bert_sco(i, reference_file)

            print("--------------------------")



    # file = "test_sen_mse_add_mask_in_input"
    # # file = "test_sen_mse_add_mask_in_input_ablation_token_mean"
    # dir = "train_sen_mse_add_mask_in_input_ablation_2_after_49_1122-s-89472-s-LongLM-7456"
    # hypothesis_file_0 = "./pred_mask_stage_2/{}/{}.0".format(dir, file)
    # hypothesis_file_1 = "./pred_mask_stage_2/{}/{}.1".format(dir, file)
    # # hypothesis_file_0 = "./Output/{}/{}.0".format(dir, file)
    # # hypothesis_file_1 = "./Output/{}/{}.1".format(dir, file)
    # # hypothesis_file_0 = "./predict/{}/{}.0".format(dir, file)
    # # hypothesis_file_1 = "./predict/{}/{}.1".format(dir, file)
    # hypothesis_file_list = [hypothesis_file_0, hypothesis_file_1]
    # reference_file = "./data_ours/final_data/test.json"
    # for i in hypothesis_file_list:
    #     # bleu_score = Bleu_Metric(hypothesis_file=i, reference_file=reference_file).bleu()
    #     # com_ppl(i) # need activate 4_ppl
    #     # bert_sco(i, reference_file)
    #     # style_classify(i)
    #     # repetition_distinct(i)
        



