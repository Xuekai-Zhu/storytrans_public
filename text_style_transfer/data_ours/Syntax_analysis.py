import json
import os
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import random


def merge_data(dir, save):
    files = os.listdir(dir)
    f = open(save, 'w')
    for file in tqdm(files):
        csv = pd.read_csv(os.path.join(dir, file))
        data_asarry = csv.values
        length = len(csv)
        for i in range(length):
            new_form = {}
            item = data_asarry[i]
            new_form['text'] = item[2]
            new_form['title'] = item[1]
            new_form['author'] = item[3]
            save_json = json.dumps(new_form, ensure_ascii=False)
            f.write(save_json + '\n')


    f.close()

def clean_data(file, save):
    bad_case = ['在线书库HTTP://WWW。XIAOSHUOTxt。net/永无弹窗广告、干净清爽，提供经典小说文学书籍在线阅读，精心筛选只收录和推荐同类精品。',
                'XIAOSHUOTxt。net/',
                '在线书库;HTTP://WWW。XIAOSHUOTxt。net/）',
                '小/说。T/xt。天+',
                '大,学生,小,说,,网',
                "大;学，生，小，说'网",
                "作者/",
                ">T……xt说……天……堂……作者/",
                ">T……xt说……天……堂……"]
    save_f = open(save, 'w')
    all_data = []
    text_list = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            text = data['text']
            if text in text_list:
                continue
            else:
                text_list.append(text)

            for bad_text in bad_case:
                if bad_text in text:
                    text = text.replace(bad_text, '')

            data['text'] = text
            item = json.dumps(data, ensure_ascii=False)
            all_data.append(item)


    random.shuffle(all_data)
    for item in all_data:
        save_f.write(item + '\n')


    save_f.close()



def statistics_data(file, save):

    save_f = open(save, 'w')

    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            text = data['text']
            length = len(text)
            save_f.write(str(length) + '\n')

    save_f.close()

def statistics_plot(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    length_list = []
    for line in lines:
        item = json.loads(line)
        text = item['text']
        length_list.append(len(text))

    length = np.array(length_list)

    series = pd.Series(length)
    series_b = series.value_counts(bins=5, sort=False)
    x = series_b.index
    y = list(series_b)

    plt.bar(range(len(y)), y, width=0.4, color='c', tick_label=x)
    plt.xlabel('length', fontsize=8)
    plt.ylabel('frequency', fontsize=8)
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

def statistics_num(dir):
    files = os.listdir(dir)
    author_num = len(files)
    print("一共{}个作家".format(author_num))
    for file in files:
        csv = pd.read_csv(os.path.join(dir, file))
        data_asarry = csv.values
        length = len(csv)
        author = file.split('.')[0]
        print("author:{};data_num:{}".format(author, length))

def single_author_statistics(file):
    csv = pd.read_csv(file)
    data = csv.values
    len_list = []
    for line in data:
        text = line[2]
        len_list.append(len(text))

    length = np.array(len_list)

    series = pd.Series(length)
    series_b = series.value_counts(bins=5, sort=False)
    x = series_b.index
    y = list(series_b)

    plt.bar(range(len(y)), y, width=0.4, color='c', tick_label=x)
    plt.xlabel('length', fontsize=8)
    plt.ylabel('frequency', fontsize=8)
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


def csv2json(csv, min_len, save):
    bad_case = ['在线书库HTTP://WWW。XIAOSHUOTxt。net/永无弹窗广告、干净清爽，提供经典小说文学书籍在线阅读，精心筛选只收录和推荐同类精品。',
                'XIAOSHUOTxt。net/',
                '在线书库;HTTP://WWW。XIAOSHUOTxt。net/）',
                '小/说。T/xt。天+',
                '大,学生,小,说,,网',
                "大;学，生，小，说'网",
                "作者/",
                ">T……xt说……天……堂……作者/",
                ">T……xt说……天……堂……"]
    title_bad_case = ['（精品公版）']
    csv = pd.read_csv(csv)
    data_asarry = csv.values
    length = len(data_asarry)
    f = open(save, 'w', encoding='utf-8')
    for i in tqdm(range(length)):
        item = data_asarry[i]
        text = item[2]
        title = item[1]
        if len(text) < min_len:
            continue
        for bad_text in bad_case:
            if bad_text in text:
                text = text.replace(bad_text, '')

        for bad_title in title_bad_case:
            if bad_title in title:
                title = title.replace(bad_title, '')
        new_form = {}
        new_form['text'] = text
        new_form['title'] = title
        new_form['author'] = item[3]

        save_json = json.dumps(new_form, ensure_ascii=False)
        f.write(save_json + '\n')

    f.close()




def random_sampling(file_list, save, sampling_num):
    save_f = open(save, 'w', encoding='utf-8')
    luxun_data ,jinyong_data = file_list[0], file_list[1]
    with open(luxun_data, 'r') as f:
        luxun_list = f.readlines()
    with open(jinyong_data, 'r') as f:
        jinyong_list = f.readlines()

    new_luxun_list = random.sample(luxun_list, sampling_num)
    new_jinyong_list = random.sample(jinyong_list, sampling_num)

    new_list = new_luxun_list + new_jinyong_list
    random.shuffle(new_list)

    for item in tqdm(new_list):
        save_f.write(item)

    save_f.close()



def story_len(file):
    len_list = []
    with open(file, 'r') as f:
        data = f.readlines()
        for line in data:
            item = json.loads(line)
            story = item['story']
            len_list.append(len(story))

    length = np.array(len_list)

    series = pd.Series(length)
    series_b = series.value_counts(bins=5, sort=False)
    x = series_b.index
    y = list(series_b)

    plt.bar(range(len(y)), y, width=0.4, color='c', tick_label=x)
    plt.xlabel('length', fontsize=8)
    plt.ylabel('frequency', fontsize=8)
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


def divide2train(file, file_list, percentage_list):
    train, valid, test = file_list[0], file_list[1], file_list[2]
    train_f = open(train, 'w')
    valid_f = open(valid, 'w')
    test_f = open(test, 'w')

    train_num, valid_num, test_num = percentage_list[0], percentage_list[1], percentage_list[2]

    with open(file, 'r') as f:
        data = f.readlines()

    data_num = len(data)

    train_num = int(data_num * train_num)
    valid_num = int(data_num * valid_num)
    test_num = int(data_num * test_num)

    random.shuffle(data)

    train_data = data[:train_num]
    valid_data = data[train_num:train_num+valid_num]
    test_data = data[train_num+valid_num:]

    for item in train_data:
        train_f.write(item)

    for item in valid_data:
        valid_f.write(item)

    for item in test_data:
        test_f.write(item)


    train_f.close()
    valid_f.close()
    test_f.close()

def merge_train_set(author_data_list, story_data_list, filal_data_list):
    for author_data, story_data, save_path in zip(author_data_list, story_data_list, filal_data_list):
        with open(author_data, 'r') as f:
            author_data = f.readlines()

            author_data_new_list = []
            for line in author_data:
                item = json.loads(line)
                new_item = {
                    'text': item['text'],
                    'title': item['title'],
                    'style': item['author'],
                }
                new_item = json.dumps(new_item, ensure_ascii=False)
                author_data_new_list.append(new_item + '\n')


        with open(story_data, 'r') as f:
            story_data = f.readlines()

            story_data_new_list = []
            style = '故事'
            for line in story_data:
                item = json.loads(line)
                new_item = {
                    'text': item['story'],
                    'title': item['title'],
                    'style': style,
                }
                new_item = json.dumps(new_item, ensure_ascii=False)
                story_data_new_list.append(new_item + '\n')

        with open(save_path, 'w') as f:
            all_data = author_data_new_list + story_data_new_list
            random.shuffle(all_data)
            for item in all_data:
                f.write(item)



if __name__ == '__main__':
    # dir = 'data_from_cym/styletransfer/11-construct-csv-per-author-wholepara'
    dir_5_author = 'data_from_cym/styletransfer/5-author-csv'
    save_v1 = 'final_data/data_v1.json'
    save_v2 = 'final_data/data_v2.json'
    save_v3 = 'final_data/data_v3.json'
    save_length = 'final_data/length.txt'

    story_train = './story_data/train.jsonl'
    story_valid = './story_data/valid.jsonl'
    story_test = './story_data/test_private.jsonl'

    dir_2_author = 'data_from_cym/styletransfer/2-author-csv'
    luxun_csv = 'data_from_cym/styletransfer/2-author-csv/luxun.csv'
    jinyong_csv = 'data_from_cym/styletransfer/2-author-csv/jinyong.csv'

    luxun_json = 'data_from_cym/styletransfer/2-author-csv/luxun.json'
    jinyong_json = 'data_from_cym/styletransfer/2-author-csv/jinyong.json'

    author_train_set = 'final_data/author_train.json'
    author_valid_set = 'final_data/author_valid.json'
    author_test_set = 'final_data/author_test.json'
    author_data_list = [author_train_set, author_valid_set, author_test_set]

    percentage_list = [0.6, 0.1, 0.3]

    train_set = 'final_data/train.json'
    valid_set = 'final_data/valid.json'
    test_set = 'final_data/test.json'


    story_data_list = ['story_data/train.jsonl', 'story_data/valid.jsonl', 'story_data/test_private.jsonl']
    final_data_list = [train_set, valid_set, test_set]



    # merge_data(dir_5_author, save_v2)
    # clean_data(save_v2, save_v3)
    # statistics_data(save_v3, save_length)
    # statistics_plot(save_length)
    # statistics_num(dir_2_author)

    # story_len(story_test)

    # single_author_statistics(jinyong_csv)
    # csv2json(luxun_csv, 64, luxun_json)
    # sampling_num = 5000
    # random_sampling([luxun_json, jinyong_json], save_v1, sampling_num)


    # divide2train(save_v1, [author_train_set, author_valid_set, author_test_set], percentage_list)

    # merge_train_set(author_data_list, story_data_list, final_data_list)
    # for data in final_data_list:
    #     statistics_plot(data)