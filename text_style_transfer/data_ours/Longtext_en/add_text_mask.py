import json
from tqdm import tqdm
from nltk import word_tokenize

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

def filter_mask_word(mask_words):
    new_mask_word = []
    for word in mask_words:
        if len(word) <= 1:
            continue
        new_mask_word.append(word)
    return new_mask_word




def add_mask(files, saves):
    for i, file in enumerate(tqdm(files)):
        data = load_file(file)
        with open(saves[i], 'w') as f_s:
            for item in data:
                text = item["text"]
                mask_words = item["mask_word"]
                mask_words = filter_mask_word(mask_words)
                new_text_mask = []
                new_text = []
                for sen in text:
                    words = word_tokenize(sen)
                    new_sen_mask = []
                    for num, w in enumerate(words):
                        if w in mask_words:
                            new_sen_mask.append("<mask>")
                        else:
                            new_sen_mask.append(w)
                    new_text_mask.append(" ".join(new_sen_mask))
                    new_text.append(" ".join(words))
                item["text_mask"] = new_text_mask
                item["text"] = new_text
                item["mask_word"] = mask_words
                new_item = json.dumps(item, ensure_ascii=False)
                f_s.write(new_item + '\n')

def sorted_keywords(files, saves):
    for i, file in enumerate(tqdm(files)):
        data = load_file(file)
        with open(saves[i], 'w') as f_s:
            for item in data:
                text = item["text"]
                mask_words = item["mask_word"]
                all_words = word_tokenize(" ".join(text))
                sorted_words = []
                for w in all_words:
                    if w in mask_words:
                        sorted_words.append(w)
                for mask in mask_words:
                    if mask not in sorted_words:
                        sorted_words.append(mask)

                item["sorted_mask_word"] = sorted_words
                new_item = json.dumps(item, ensure_ascii=False)
                f_s.write(new_item + '\n')

if __name__ == '__main__':
    # files = ["style_merge_data/train", "style_merge_data/vaild", "style_merge_data/test"]
    # saves = ["style_merge_data/train.mask", "style_merge_data/vaild.mask", "style_merge_data/test.mask"]
    
    # files = ["sp+story/style_transfer_data/train", "sp+story/style_transfer_data/test"]
    # saves = ["sp+story/style_transfer_data/train.mask", "sp+story/style_transfer_data/test.mask"]
    # files = ["sp+story/classifier_data/train", "sp+story/classifier_data/test"]
    # saves = ["sp+story/classifier_data/train.tokenize", "sp+story/classifier_data/test.tokenize"]

    files = ["sp+story/style_transfer_data/vaild"]
    saves = ["sp+story/style_transfer_data/vaild.mask"]
    
    
    add_mask(files, saves)
    # files = ["sp+story/style_transfer_data/train.mask", "sp+story/style_transfer_data/test.mask"]
    # saves = ["sp+story/style_transfer_data/train.mask.sorted", "sp+story/style_transfer_data/test.mask.sorted"]
    # sorted_keywords(files, saves)
