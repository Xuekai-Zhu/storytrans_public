import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def get_text_list(file):
    data_dict = {
        "<LX>": [],
        "<JY>": [],
        "<GS>": [],
    }
    with open(file, "r") as f:
        data = f.readlines()
        for line in tqdm(data):
            item = json.loads(line)
            text = item["text"]
            seg = jieba.cut(text)
            seg = " ".join(seg)
            style = item["style"]
            data_dict[style].append(seg)
    return data_dict


def keywords(file, save):
    f_s = open(save, "w")
    num = 20
    data_dict = get_text_list(file)
    key_list = data_dict.keys()
    vectorizer = CountVectorizer()
    trans = TfidfTransformer()
    for key in key_list:
        corpus = data_dict[key]
        if len(corpus) == 0:
            continue
        mat = vectorizer.fit_transform(corpus)
        tf_idf = trans.fit_transform(mat)
        # df_word_tfidf = pd.DataFrame(tf_idf.toarray(), columns=vectorizer.get_feature_names())
        tf_idf = tf_idf.toarray()
        words = vectorizer.get_feature_names()
        for i, res in enumerate(tqdm(tf_idf)):
            index = np.argsort(res)[-num:]
            # index = np.sort(index)
            key_word = [words[i] for i in index.tolist()]
            key_word = [word for word in corpus[i].split() if word in key_word]
            new_item = {
                "text": corpus[i].replace(" ", ""),
                # "text": corpus[i],
                "key_words": key_word,
                "style": key,
            }
            new_item = json.dumps(new_item, ensure_ascii=False)
            f_s.write(new_item + "\n")

    f_s.close()

def check_keywords_len(file):
    len_list = []
    with open(file, 'r') as f:
        data = f.readlines()
        for line in data:
            item = json.loads(line)
            text = item["text"]
            key_words = item["key_words"]
            num = len(text) - len("".join(key_words))
            len_list.append(num)

    series = pd.Series(len_list)
    series_b = series.value_counts(bins=5, sort=False)
    x = series_b.index
    y = list(series_b)

    plt.bar(range(len(y)), y, width=0.4, color='c', tick_label=x)
    plt.xlabel('length', fontsize=8)
    plt.ylabel('frequency', fontsize=8)
    plt.show()

    max_num = np.max(len_list)
    min_num = np.min(len_list)
    mean_num = np.mean(len_list)
    print("最大长度:{}".format(max_num))
    print("最小长度：{}".format(min_num))
    print("平均长度：{}".format(mean_num))



if __name__ == '__main__':
    # file = "../data_ours/final_data/train.json"
    save = "../data_ours/data_keyword/train.keywords"
    # keywords(file, save)
    # file = "../data_ours/final_data/test.json"
    # save = "../data_ours/data_keyword/test.keywords"
    # keywords(file, save)

    check_keywords_len(save)
