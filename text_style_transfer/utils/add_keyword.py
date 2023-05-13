import json
from tqdm import tqdm


def load_text(file):
    hypothesis_list = []
    keyword_list = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        for line in data:
            item = json.loads(line)
            text = item["text"]
            keyword = item["mask_word"]
            keyword_list.append(keyword)
            hypothesis_list.append(text)
    # return hypothesis_list
    return hypothesis_list, keyword_list


def save_keyword(file, save):
    _, keyword_list = load_text(file)
    with open(save, "w") as f:
        for i in tqdm(keyword_list):
            f.write(" ".join(i) + '\n')



if __name__ == '__main__':
    file = "../data_ours/auxiliary_data/train.sen.add_index.mask"
    save = "../data_ours/auxiliary_data/train.keyword"

    save_keyword(file, save)