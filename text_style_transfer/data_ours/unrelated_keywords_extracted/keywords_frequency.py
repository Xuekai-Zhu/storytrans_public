import jieba
import json
from collections import Counter
from tqdm import tqdm
from nltk.tokenize import word_tokenize


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

def frquency_analyse_ch(texts, style):
    all_words = []
    for i in texts:
        words = jieba.cut("".join(i), cut_all=True)
        words = list(words)
        all_words = all_words + words

    results = Counter(all_words)
    sorted_re = results.most_common()
    record_file = "word_frequency.{}".format(style)
    with open(record_file, 'w') as f:
        for word,fre in tqdm(sorted_re):
            f.write(word + " " + str(fre) + "\n")
    


def extract_keywords_for_cn(file):
    lx_text = load_text(file, "<LX>")
    jy_text = load_text(file, "<JY>")
    gs_text = load_text(file, "<GS>") + load_text("../auxiliary_data/valid.sen.add_index", "<GS>") + load_text("../auxiliary_data/test.sen.add_index.mask", "<GS>")
    frquency_analyse_ch(lx_text, "lx")
    frquency_analyse_ch(jy_text, "jy")
    frquency_analyse_ch(gs_text, "gs")
    
    
def frquency_analyse_en(texts, style):
    all_words = []
    for i in texts:
        words = word_tokenize(" ".join(i))
        words = list(words)
        all_words = all_words + words

    results = Counter(all_words)
    sorted_re = results.most_common()
    record_file = "word_frequency.{}".format(style)
    with open(record_file, 'w') as f:
        for word,fre in tqdm(sorted_re):
            f.write(word + " " + str(fre) + "\n")

def extract_keywords_for_en(file):
    shakespeare_text = load_text(file, "<Sp>")
    story_text = load_text(file, "<St>") + load_text("../Longtext_en/sp+story/style_transfer_data/vaild.mask", "<St>") + load_text("../Longtext_en/sp+story/style_transfer_data/test.mask.sorted", "<St>")
    frquency_analyse_en(shakespeare_text, "shakespeare")
    frquency_analyse_en(story_text, "story")

    
    
    
    

if __name__ == '__main__':
    train_file = "../auxiliary_data/train.sen.add_index.mask"
    # test_file = "../auxiliary_data/test.sen.add_index.mask"
    # vaild_file = "../auxiliary_data/vaild.sen.add_index"
    # extract_keywords_for_cn(train_file)
    
    
    en_train_file = "../Longtext_en/sp+story/style_transfer_data/train.mask.sorted"
    extract_keywords_for_en(en_train_file)
    