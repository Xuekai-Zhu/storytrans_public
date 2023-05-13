from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch
from data_set import Data_Encoder_eval, Data_Encoder_eval_en
import json
import os
from sklearn.metrics import accuracy_score
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
from nltk import word_tokenize


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


class StyleClassifier_Config(object):
    pre_trained_t5 = "pretrained_model/t5-base"
    pretrained_sc = "pretrained_model/pre-trained_style_classifier_2_style/epoch-2-step-580-loss-2.6538571546552703e-05.pth"
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    max_length = 512
    in_size = 768
    style_num = 2


def style_classify_en(eval_file):
    config = StyleClassifier_Config()
    dataset = Data_Encoder_eval_en(eval_file)
    data_generator = DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        # create model
        tokenizer = T5Tokenizer.from_pretrained(config.pre_trained_t5)
        # tokenizer.add_special_tokens(
        #     {"bos_token": "<s>", "additional_special_tokens": ['<MT>', '<JK>', '<St>', '<SEN>', '<mask>']})
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": ['<Sp>', '<St>', '<SEN>', '<mask>']})
        model = Style_Classifier(config).to(config.device)
        model.encoder.resize_token_embeddings(len(tokenizer))

        # tokenizer = T5Tokenizer.from_pretrained("pretrained_model/t5-base")
        # model = Style_Classifier(config).to(config.device)
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
                if isinstance(text, list):
                    text = " ".join(text)
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


    def bleu_en(self):
        res = {}
        for i in range(1, 5):
            res["bleu-%d" % i] = []

        for origin_reference, origin_candidate in tqdm(zip(self.refernce_list, self.hypothesis_list)):
            origin_candidate = word_tokenize(origin_candidate)
            origin_reference = word_tokenize(origin_reference)
            # origin_reference = proline(origin_reference)
            # origin_candidate = proline(origin_candidate)
            # assert isinstance(origin_candidate, str)
            # if not isinstance(origin_reference, list):
            #     origin_reference = [origin_reference]

            for i in range(1, 5):
                res["bleu-%d" % i].append(sentence_bleu(references=[origin_reference],
                                                        hypothesis=origin_candidate,
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
    (P, R, F), hashname = score(cand_list, refs_list, lang="en", return_hash=True)
    print(f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}")


def com_ppl(file):
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    ce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").to(device)
    model.load_state_dict(torch.load(
        "baselines/Fine-tune-GPT/model/fine-tune-gpt2-chinese/epoch-1-step-3728-loss-2.8955578804016113.pth"),
                          strict=True)
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
        result["distinct-%d" % i] = len(all_ngram) / float(all_ngram_num)

    print(result)
    # return result


def target_score(eval_file):
    
    
    config = StyleClassifier_Config()
    dataset = Data_Encoder_eval_en(eval_file)
    data_generator = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        # create model
        tokenizer = T5Tokenizer.from_pretrained(config.pre_trained_t5)
        # tokenizer.add_special_tokens(
        #     {"bos_token": "<s>", "additional_special_tokens": ['<MT>', '<JK>', '<St>', '<SEN>', '<mask>']})
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": ['<Sp>', '<St>', '<SEN>', '<mask>']})
        model = Style_Classifier(config).to(config.device)
        model.encoder.resize_token_embeddings(len(tokenizer))

        # tokenizer = T5Tokenizer.from_pretrained("pretrained_model/t5-base")
        # model = Style_Classifier(config).to(config.device)
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
                if eval_file.split("/")[-1] == "test.mask":
                    batch_text = [i[0] for i in batch_text]
                    batch_text = " ".join(list(batch_text))
                ids = tokenizer(batch_text,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=config.max_length).input_ids.to(config.device)

                labels = label.cpu().numpy()

                logits = model(ids)

                p = F.softmax(logits, dim=-1).cpu().numpy()
                # pre_label = np.argmax(p, axis=-1)
                # target_score = p[labels]
                for sco in p:
                    f.write(str(sco[0]) + "  " + str(sco[1]) + '\n')


def r_acc_and_a_acc(eval_file, ref):
    target_style_index = 0
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
    # print("a_acc:{}".format(a_acc))
    
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # files = ["Output_en/2_stage_0.5/test_transfer_stage_1.0", "Output_en/Keywords2Story/test_en.0", "Output_en/reverse_attention/test_reverse_style_attention.0", "Output_en/Style_Transformer/test_st_en.0"]
    # ablation_files = ["pred_mask_stage_2/tran_transfer_stage_1_0.5_abalation_sen_loss_0103-s-18560-t5-base-2320/test_transfer_stage_1.0", "pred_mask_stage_2/tran_transfer_stage_1_0.5_abalation_style_classifier_loss_0103-s-18560-t5-base-2320/test_transfer_stage_1.0"]
    reference_file_en = 'data_ours/Longtext_en/sp+story/style_transfer_data/test.mask'
    # singel_text = ["pred_mask_stage_2/train_sen_mse_add_mask_in_input_ablation_token_mean_0106-s-23200-t5-base-2320/test_sen_mse_add_mask_in_input_ablation_token_mean.0"]
    # ablation_files = ["Output_en/ablation/multi_sen/test_sen_mse_add_mask_in_input_ablation_token_mean.0", "Output_en/ablation/one_stage/test_ablation_con_enh.0", "Output_en/ablation/sen_loss/test_transfer_stage_1.0", "Output_en/ablation/style_classifier/test_transfer_stage_1.0"]
    
    # hypothesis_file_en = "baselines/T5+style_embedding/predict/train_st_stop_gradient_cycle_en-Dec29165024-s-10449/test_st_en.0"
    
    
    # 主要的测试文件
    files = [
        "pred_add_sentype/en_train_use_1_stage_1224_e_79/stage_1/test_use_1_stage.0"
        ]
    
    
    for i in files:
    # for i in ablation_files:
    # for i in singel_text:
        # i = "pred_stage_1/test_ablation_con_enh_0301-s-23200/test_ablation_con_enh.0"
        print(i)
        bleu_score = Bleu_Metric(hypothesis_file=i, reference_file=reference_file_en).bleu_en()
        # com_ppl(i) # need activate 4_ppl
        bert_sco(i, reference_file_en)
        style_classify_en(i)
        r_acc_and_a_acc(i, reference_file_en)
        print('-------------------', "\n")
        # repetition_distinct(i)

    # file = "pred_mask_stage_2/tran_transfer_stage_1_0.5_0103-s-37120-t5-base-2320/test_transfer_stage_1.0"
    # bleu_score = Bleu_Metric(hypothesis_file=file, reference_file=reference_file_en).bleu_en()
    # style_classify_en(file)


