import json
from tqdm import tqdm
from nltk import word_tokenize
import numpy as np
import jieba


def load_text(file, author=None):
    hypothesis_list = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        for line in data:
            item = json.loads(line)
            text = item["text"]
            if author is not None and author == item["style"]:
                hypothesis_list.append(text)
            elif author is None:
                hypothesis_list.append(text)
    return hypothesis_list


def load_file(file):
    all_dict = []
    with open(file, 'r') as f:
        for i in f:
            all_dict.append(json.loads(i))
    return all_dict




def com_mask(files):
    for i in files:
        data = load_file(i)
        all_n = []
        for item in data:
            text_mask = item["text_mask"]
            n = 0
            for sen in text_mask:
                words = sen.split()
                for w in words:
                    if w == "<mask>":
                        n += 1
            all_n.append(n)
        print("{} 平均mask数量为{}".format(i, np.mean(all_n)))


def length_analsy(files, author=None):
    texts = load_text(files, author=author)
    length_list = []
    for i in texts:
        inputs = " ".join(i)
        length_list.append(len(inputs.split()))
    print("ave len", np.mean(length_list))
    
    
    
def author_ana(file):
    texts = load_file(file)
    sp = 0
    story = 0
    for text in texts:
        author = text["style"]
        if author == "<St>":
            story += 1
        elif author == "<Sp>":
            sp += 1
    
    print("莎士比亚", sp)
    print("故事", story)
    
    
    

if __name__ == '__main__':
    # files = ["style_merge_data/train.mask.sorted", "style_merge_data/test.mask.sorted"]
    # com_mask(files)
    
    # train_file = "sp+story/style_transfer_data/train.mask.sorted"
    # test_file = "sp+story/style_transfer_data/test.mask.sorted"
    # length_analsy(train_file)
    # length_analsy(test_file)
    
    # author_ana(train_file)
    
    # length analyse
    train_file = "sp+story/style_transfer_data/train.mask"
    vaild_file = "sp+story/style_transfer_data/vaild.mask"
    test_file = "sp+story/style_transfer_data/test.mask"
    length_analsy(train_file, author="<St>")
    length_analsy(vaild_file)
    length_analsy(test_file)