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


def author_ana(file):
    texts = load_file(file)
    jy = 0
    lx = 0
    gs = 0
    for text in texts:
        author = text["style"]
        if author == "<JY>":
            jy += 1
        elif author == "<GS>":
            gs += 1
        elif author == "<LX>":
            lx += 1
        
    print("金庸", jy)
    print("鲁迅", lx)
    print("故事", gs)
    
def length_analsy(files, author=None):
    texts = load_text(files, author=author)
    length_list = []
    for i in texts:
        inputs = " ".join(i)
        # words = jieba.cut(inputs)
        # words = list(words)
        length_list.append(len(inputs))
    print("ave len", np.mean(length_list))

if __name__ == '__main__':
    # file = "auxiliary_data/train.sen.add_index.mask"
    # author_ana(file)
    
    train_file = "auxiliary_data/train.sen.add_index.mask"
    vaild_file = "auxiliary_data/valid.sen.add_index"
    test_file = "auxiliary_data/test.sen.add_index.mask"
    
    length_analsy(train_file, author="<JY>")
    length_analsy(train_file, author="<LX>")
    length_analsy(train_file, author="<GS>")
    length_analsy(vaild_file)
    length_analsy(test_file)