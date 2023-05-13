from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch
from data_set import Data_Encoder_eval
import json
import os
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.tokenize import word_tokenize
import jieba
from tqdm import tqdm
import numpy as np
from transformers import T5Tokenizer
from modeling_t5 import Style_Classifier
import torch.nn.functional as F




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

            # a = [r.strip().split() for r in origin_reference]
            # b = origin_candidate.strip().split()
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




if __name__ == '__main__':
    # file = "pred_fill/train_fill_LongLM-distrub-11-s-14912/test_fill_base"
    # reference_file = "./data_ours/final_data/test.json"
    file = "pred_fill/train_fill_LongLM-en-2-style-1222-s-2320/test_fill"
    reference_file_en = 'data_ours/Longtext_en/sp+story/style_transfer_data/test.mask'
    # bleu_score = Bleu_Metric(hypothesis_file=file, reference_file=reference_file).bleu()
    bleu_score = Bleu_Metric(hypothesis_file=file, reference_file=reference_file_en).bleu_en()