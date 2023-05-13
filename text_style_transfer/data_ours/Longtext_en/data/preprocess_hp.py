import os
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import random
import json
import numpy as np
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


def divide_into_item(sens_list, title, style):
    s = 0
    all_item = []
    max_len = len(sens_list)
    while s <= max_len:
        n = random.randint(3, 6)
        text = sens_list[s:s + n]
        item = {
            "text": text,
            "style": style,
            "title": title,
        }
        all_item.append(item)
        s += n

    return all_item


def pre_fileter(sens):
    all_sen = []
    for i in sens:
        if i in [".", "?", "!", ","]:
            continue
        all_sen.append(i)

    return all_sen

def save_data(path, data):
    with open(path, 'w') as f:
        for i in data:
            i = json.dumps(i, ensure_ascii=False)
            f.write(i + "\n")


def preprocess(dir):
    files = find_files(dir)
    style = "<JK>"
    all_con = []
    for i in tqdm(files):
        title = i.split('/')[-1].split('.')[0]
        data = load_file(i)
        sens = sent_tokenize(data[0])
        sens = pre_fileter(sens)
        items = divide_into_item(sens, title, style)
        all_con = all_con + items

    save_f = os.path.join("mt+jk", "jk")
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



# post filter
def filter_hard_sample(file, save):
    with open(file, 'r') as f:
        data = f.readlines()
    new_con = []
    for i in tqdm(data):
        item = json.loads(i)
        text = item["text"]
        words = word_tokenize(" ".join(text))
        if len(text) == 0:
            continue
        if len(words) < 40 or len(words) > 384:
            continue

        new_con.append(i)

    with open(save, 'w') as f:
        for i in new_con:
            f.write(i)


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


if __name__ == '__main__':
    dir = "jk_rowing"
    preprocess(dir)
    file = "mt+jk/jk"
    save = "mt+jk/jk.v1"
    filter_hard_sample(file, save)
    statistics_plot(save)