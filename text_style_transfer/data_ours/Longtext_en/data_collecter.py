import json
import os
import csv
import pandas as pd
import random
from tqdm import tqdm
from nltk import word_tokenize
import numpy as np
from matplotlib import pyplot as plt

def meger_files(files):
    style = "<St>"
    all_item = []
    for i in files:
        csvFile = pd.read_csv(i)
        file = pd.DataFrame(csvFile)
        for index, row in file.iterrows():
            title = row[1]
            sens = [row[2], row[3], row[4], row[5], row[6]]
            item = {
                "text": sens,
                "title": title,
                "style": style,
            }
            item = json.dumps(item, ensure_ascii=False)
            all_item.append(item)

    return all_item



def divide_into_subset(input_files, file_list):
    train, valid, test = file_list[0], file_list[1], file_list[2]
    train_f = open(train, 'w')
    valid_f = open(valid, 'w')
    test_f = open(test, 'w')


    data = meger_files(input_files)
    random.shuffle(data)
    data_num = len(data)

    train_num = int(data_num * 0.8)
    valid_num = int(data_num * 0.1)
    test_num = int(data_num * 0.1)

    train_data = data[:train_num]
    valid_data = data[train_num:train_num + valid_num]
    test_data = data[train_num + valid_num:]


    for item in tqdm(train_data):
        train_f.write(item + '\n')

    for item in tqdm(valid_data):
        valid_f.write(item + '\n')

    for item in tqdm(test_data):
        test_f.write(item + '\n')


    train_f.close()
    valid_f.close()
    test_f.close()



def statistics_plot(files):
    data = meger_files(files)
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



if __name__ == '__main__':
    files = ["roc_story/metadata/ROCStories__spring2016-ROCStories_spring2016.csv", "roc_story/metadata/ROCStories_winter2017-ROCStories_winter2017.csv"]
    subsets = ["roc_story/train", "roc_story/vaild", "roc_story/test"]
    # divide_into_subset(files, subsets)
    statistics_plot(files)
