import json
from tqdm import tqdm

def load_text(file, author=None):
    hypothesis_list = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        for line in data:
            item = json.loads(line)
            text = item["mask_word"]
            if author is not None and author == item["style"]:
                hypothesis_list.append(text)
            elif author is None:
                hypothesis_list.append(text)
    return hypothesis_list

def load_stye_unrelated_words(file):
    words = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words.append(line.strip())
    return words


def check_words(dataset, unrelated_words):
    keywords = load_text(dataset)
    style_unrelated_word = load_stye_unrelated_words(unrelated_words)
    for samples in tqdm(keywords):
        for word in samples:
            if word in style_unrelated_word:
                print(word)
                
                
if __name__=="__main__":
    ch_unstyle = "style_unrelated_words.ch"
    ch_datasets = ["../auxiliary_data/train.sen.add_index.mask", "../auxiliary_data/test.sen.add_index.mask"] 
    # for i in ch_datasets:
    #     check_words(i, ch_unstyle)
        
        
    en_unstyle  = "style_unrelated_words.en"
    en_datasets = ["../Longtext_en/sp+story/style_transfer_data/train.mask", "../Longtext_en/sp+story/style_transfer_data/test.mask"] 
    for i in en_datasets:
        check_words(i, en_unstyle)

