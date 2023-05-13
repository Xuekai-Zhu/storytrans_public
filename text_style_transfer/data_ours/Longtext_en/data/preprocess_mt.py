import json
import os
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt



def find_files(dir):
    files = os.listdir(dir)
    files = [os.path.join(dir, i) for i in files]
    return files

def load_file(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return data


def fileter(content):
    # punc = string.punctuation
    new_con = []
    for i in content:
        words = word_tokenize(i)
        if len(words) == 1 and words[0] in [".", "?", "!", ","]:
            continue
        new_con.append(i)

    return new_con

def get_main_content(data, title, style):
    # 去除换行符和空格
    all_data = []
    for l in data:
        if l == "\n":
            continue
        else:
            l = l.strip()
            if l != " ":
                all_data.append(l)
    # 分句
    all_data = " ".join(all_data)
    content = sent_tokenize(all_data)

    # 去除单个标点符号句子
    content = fileter(content)

    all_item = []
    s = 0
    while s <= len(content):
        n = random.randint(3, 5)
        text = content[s:s+n]
        # words_now = word_tokenize(" ".join(text))
        # if len(words_now) < 40:
        #     continue
        item = {
            "text": text,
            "style": style,
            "title": title,
        }
        all_item.append(item)
        s += n

    return all_item


def save_data(path, data):
    with open(path, 'w') as f:
        for i in data:
            i = json.dumps(i, ensure_ascii=False)
            f.write(i + "\n")

def proprecess(dir):
    files = find_files(dir)
    style = "<MT>"
    all_con = []
    for i in tqdm(files):
        title = i.split('/')[-1].split('.')[0]
        meta_con = load_file(i)
        main_con = get_main_content(meta_con, title, style)
        all_con = all_con + main_con


    save_f = os.path.join("mt+jk", "mt")
    save_data(save_f, all_con)





def load_text(file):
    hypothesis_list = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        for line in data:
            item = json.loads(line)
            text = item["text"]
            hypothesis_list.append(text)
    return hypothesis_list


def statistics_plot(file):
    data = load_text(file)

    length_list = []
    for line in data:
        tokens = word_tokenize(" ".join(line))
        length_list.append(len(tokens))

    length = np.array(length_list)

    series = pd.Series(length)
    series_b = series.value_counts(bins=5, sort=False)
    x = series_b.index
    y = list(series_b)

    fig = plt.figure(figsize=(6, 7))
    plt.bar(range(len(y)), y, width=0.1, color='m', tick_label=x)
    plt.xlabel('Length Distribution', fontsize=10)
    plt.ylabel('Quantity', fontsize=10)
    plt.savefig("data.png", dpi=400)
    plt.show()

    # plt.hist(length, bins=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380])
    # plt.title("histogram")
    # plt.show()
    max_num = np.max(length)
    min_num = np.min(length)
    mean_num = np.mean(length)
    print("最大长度:{}".format(max_num))
    print("最小长度：{}".format(min_num))
    print("平均长度：{}".format(mean_num))


# post filter
def filter_hard_sample(file, save):
    with open(file, 'r') as f:
        data = f.readlines()
    new_con = []
    for i in tqdm(data):
        item = json.loads(i)
        text = [i.replace("_", "") for i in item["text"]]
        # all_texts = " ".join(text).replace("_", "")
        words = word_tokenize(" ".join(text))
        if len(text) == 0:
            continue
        if len(words) < 40 or len(words) > 384:
            continue

        item["text"] = text
        new_item = json.dumps(item, ensure_ascii=False)
        new_con.append(new_item)

    with open(save, 'w') as f:
        for i in new_con:
            f.write(i + '\n')



if __name__ == '__main__':
    # dir = "mt_filter_prefix"
    # proprecess(dir)
    file = "mt+jk/mt"
    # statics(file)
    save = "mt+jk/mt.v1"
    filter_hard_sample(file, save)
    # statistics_plot(save)