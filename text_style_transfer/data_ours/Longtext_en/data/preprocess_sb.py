import json
import os
from typing import Text
from nltk.classify.decisiontree import f
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import gutenberg, shakespeare
import xml.etree.ElementTree as ET

# from data_ours.Longtext_en.post_processing import check_keyword

def load_file(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return data


def load_text(file):
    hypothesis_list = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        for line in data:
            item = json.loads(line)
            # text = item["text"]
            hypothesis_list.append(item)
    return hypothesis_list

def merge_short_sens(sens):
    all_data = []
    short_sens = []
    for s in sens:
        if len(short_sens) != 0 and len(s) > 2:
            now_s = short_sens + s
            all_data.append(now_s)
            short_sens = []
        elif len(s) <= 2:
            short_sens = short_sens + s
        else:
            all_data.append(s)

    # num = len(all_data) // 5


    max_length = 85
    min_length = 22
    max_sen = 5
    five_sentence = []
    len_now = 0
    five_sens_now = []
    num = 0
    for i in range(len(all_data)):
        len_now = len_now + len(all_data[i])
        if len_now > max_length and len(five_sens_now) != 0:
            five_sentence.append(five_sens_now)
            five_sens_now = []
            five_sens_now.append(all_data[i])
            len_now = len(all_data[i])

        else:
            five_sens_now.append(all_data[i])
            
    check_repeat(five_sentence)
    
    return five_sentence


def check_repeat(input_list):
    new_list = []
    n = 0
    for i in input_list:
        text_list = [" ".join(j) for j in i]
        text = " ".join(text_list)
        if text in new_list:
            print(text)
            n += 1
        else:    
            new_list.append(text)
    
    print(n)    


def split_data(files):
    all_data = []
    for i in files:
        raw_sents = gutenberg.sents(i)
        sents = merge_short_sens(raw_sents)
        all_data = all_data + sents
    return all_data


def sp_data_process(files, save):
    style = '<Sp>'
    with open(save, 'w') as f_s:
        all_data = split_data(files)
        check_repeat(all_data)
        all_length = []
        for i in tqdm(all_data):
            sens = []
            num = 0
            for sen in i:
                num += len(sen)
                sens.append(" ".join(sen))
            item = {
                "text": sens,
                "style": style,
            }
            all_length.append(num)
            
            new_item = json.dumps(item, ensure_ascii=False)
            f_s.write(new_item + "\n")
            
    # plot_length(all_length)       
    
            
            



def simple_sta(files):
    for fileid in files:
        num_chars = len(gutenberg.raw(fileid))
        num_words = len(gutenberg.words(fileid))
        num_sents = len(gutenberg.sents(fileid))
        num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
        print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)



def statistics_plot(files):
    data = load_file(files)
    data = [" ".join(json.loads(i)["text"]) for i in data]
    length_list = []
    for line in data:
        tokens = word_tokenize(line)
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


def plot_length(length_list):
    all_length = np.array(length_list)
    series = pd.Series(all_length)
    series_b = series.value_counts(bins=5, sort=False)
    x = series_b.index
    y = list(series_b)
    
    fig = plt.figure(figsize=(6, 7))
    plt.bar(range(len(y)), y, width=0.1, color='m', tick_label=x)
    plt.xlabel('Length Distribution', fontsize=10)
    plt.ylabel('Quantity', fontsize=10)
    plt.savefig("data.png", dpi=400)
    plt.show()
    max_num = np.max(all_length)
    min_num = np.min(all_length)
    mean_num = np.mean(all_length)
    print("最大长度:{}".format(max_num))
    print("最小长度：{}".format(min_num))
    print("平均长度：{}".format(mean_num))



def shakespeare_nltk():
    fileids = shakespeare.fileids()
    play = shakespeare.xml(fileids[0])
    text = ''.join(play.itertext())


def filiter(file, save):
    data = load_text(file)
    all_length = []
    with open(save, "w") as f_s:
        for item in tqdm(data):
            text = " ".join(item["text"]).split()
            num = len(text)
            if num < 22 or num > 90:
                continue
            all_length.append(num)
            new_item = json.dumps(item, ensure_ascii=False)
            f_s.write(new_item + '\n')

    plot_length(all_length)





if __name__ == '__main__':
    files = ["shakespeare-caesar.txt", "shakespeare-hamlet.txt", "shakespeare-macbeth.txt"]
    # saves = "data_ours/Longtext_en/data/shakespeare/all_data"
    saves = "shakespeare/all_data"
    save_v1 = "shakespeare/all_data.filter"
    # file = "data_ours/Longtext_en/roc_story/all_stories"
    
    # statistics_plot(file)
    # sp_data_process(files, saves)
    # save_v1 = "shakespeare/all_data.v1"
    filiter(saves, save_v1)

    
    



