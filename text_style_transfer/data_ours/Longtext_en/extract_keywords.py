from flashtext import KeywordProcessor
import json
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from nltk import word_tokenize, pos_tag
from nltk.text import TextCollection
import nltk
from tqdm import tqdm
import numpy as np
import spacy
import string
import random

def merge_data(files):
    all_data = []
    for i in files:
        with open(i, 'r') as f:
            all_data = all_data + f.readlines()

    with open("style_merge_data/all_stories", "w") as f:
        for i in all_data:
            f.write(i)

def load_text(file):
    hypothesis_list = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        for line in data:
            item = json.loads(line)
            text = item["text"]
            hypothesis_list.append(text)
    return hypothesis_list


def load_file(file):
    all_dict = []
    with open(file, 'r') as f:
        for i in f:
            all_dict.append(json.loads(i))
    return all_dict


def mask_keywords(file, save):
    data = load_text(file)
    data = [" ".join(i) for i in data]
    # data = [" ".join(word_tokenize(" ".join(i))) for i in data]
    all_items = load_file(file)
    num = 10
    new_items = exact_keywords(data, all_items, num)

    with open(save, 'w') as f:
        for i in new_items:
            new_i = json.dumps(i, ensure_ascii=False)
            f.write(new_i + '\n')


def fileter_pos_word(words):
    pos_set = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RP', 'RB', 'RBR', 'RBS', "PRP"}
    # tokens = word_tokenize(words)
    pos_tags = pos_tag(words)
    filter_keyword = [word for word, pos in pos_tags if pos in pos_set]
    # filter_keyword = list(set(filter_keyword))

    return filter_keyword


def exact_keywords(corpus, all_items, num):
    # corpus_collection = TextCollection(corpus[:200])
    corpus_collection = TextCollection(corpus)
    for i, sens in enumerate(corpus):
        tokens = list(set(word_tokenize(sens)))
        tf_idf_list = []
        for j in tokens:
            tf_idf = corpus_collection.tf_idf(j, sens)
            tf_idf_list.append(tf_idf)
        index = np.argsort(np.array(tf_idf_list))[-num:]

        tf_idf_keywords = [tokens[n] for n in index]
        tf_idf_keywords = fileter_pos_word(tf_idf_keywords)
        all_items[i]["mask_word"] = tf_idf_keywords

    return all_items


def statics_tf(files, save):
    def merge_data(file):
        datas = load_text(file)
        datas = [" ".join(i) for i in datas]
        return " ".join(datas), len(datas)

    stories, num_s = merge_data(files[0])
    mts, num_m = merge_data(files[1])
    jks, num_j = merge_data(files[2])

    stories_sta = nltk.FreqDist(word_tokenize(stories))
    mts_sta = nltk.FreqDist(word_tokenize(mts))
    jks_sta = nltk.FreqDist(word_tokenize(jks))

    def fre_array(sta, threshold):
        freq = np.array([i for i in sta.values()])
        words = np.array([i for i in sta.keys()])
        high_words = words[freq > threshold]
        return high_words.tolist()

    high_stories = fre_array(stories_sta, int(num_s * 0.02))
    high_mts = fre_array(mts_sta, int(num_m * 0.02))
    high_jks = fre_array(jks_sta, int(num_j * 0.02))

    high_words = [i for i in tqdm(high_mts) if (i in high_stories) and (i in high_jks)]
    with open(save, "w") as f:
        for i in high_words:
            f.write(i + '\n')


def filter_high_fre_words(files, saves):
    with open("keywords_dataset/high_frequency", 'r') as f:
        high_fre_words = f.readlines()
    high_fre_words = [i.strip() for i in high_fre_words]
    for i, j in zip(files, saves):
        f_s = open(j, "w")
        items = load_file(i)
        for item in items:
            keywords = item["mask_word"]
            filter_keywords = []
            for key in keywords:
                if key not in high_fre_words and key not in ["’", ".", "“", "”", ""] and len(key) > 1:
                    filter_keywords.append(key)


            # text = [n.replace("_", "") for n in item["text"]]
            # filter_keywords = [i.replace("_", "") for i in filter_keywords]
            item["mask_word"] = filter_keywords
            # item["text"] = text
            new_item = json.dumps(item, ensure_ascii=False)
            f_s.write(new_item + '\n')
        f_s.close()

def divide_into_subset(input_files, file_list):
    train, valid, test = file_list[0], file_list[1], file_list[2]
    train_f = open(train, 'w')
    valid_f = open(valid, 'w')
    test_f = open(test, 'w')


    data = load_file(input_files)
    data = [json.dumps(i, ensure_ascii=False) for i in data]
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



def check_keyword(files):
    for file in files:
        data = load_file(file)
        for i in data:
            masks = i["mask_word"]
            for j in masks:
                if len(j) <= 1:
                    print(j)





if __name__ == '__main__':
    # merge_data(["roc_story/train", "roc_story/test", "roc_story/vaild"])
    # file = ["roc_story/all_stories", "data/mt+jk/mt.v1", "data/mt+jk/jk.v1"]
    # save = ["keywords_dataset/all_stories.keywords", "keywords_dataset/mt.keywords", "keywords_dataset/jk.keywords"]
    # # file = ["data/mt+jk/mt.v1"]
    # # save = ["keywords_dataset/mt.keywords"]
    # for i, j in zip(file, save):
    #     mask_keywords(i, j)
    #
    # # save = "keywords_dataset/high_frequency"
    # # statics_tf(file, save)
    # saves = ["keywords_dataset/all_stories.keywords.filter_h", "keywords_dataset/mt.keywords.filter_h", "keywords_dataset/jk.keywords.filter_h"]
    #
    # filter_high_fre_words(save, saves)
    # #
    # stories = ["keywords_dataset/stories/train", "keywords_dataset/stories/vaild", "keywords_dataset/stories/test"]
    # mt = ["keywords_dataset/mt/train", "keywords_dataset/mt/vaild", "keywords_dataset/mt/test"]
    # jk = ["keywords_dataset/jk/train", "keywords_dataset/jk/vaild", "keywords_dataset/jk/test"]
    # for i, j in zip(saves, [stories, mt, jk]):
    #     divide_into_subset(i, j)

    # check_keyword(saves)
    mask_keywords("data/shakespeare/all_data.filter", "data/shakespeare/all_data.filter.keywords")