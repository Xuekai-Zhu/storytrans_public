import json
import random
import os
from tqdm import tqdm
import numpy as np


def load_text(file):
    hypothesis_list = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        for line in data:
            item = json.loads(line)
            text = item["text"]
            hypothesis_list.append(text)
    return hypothesis_list

def get_save_name(file):
    end_name = file.split("/")[-1]
    model_name = file.split("/")[-2]
    new_dir = "human_eval_samples/{}".format(model_name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    return os.path.join(new_dir, end_name)


def save(file, data):
    with open(file, "w") as f:
        for i in tqdm(data):
            f.write(i + "\n")



def save_ref(sample_index):
    data = load_text("data_ours/final_data/test.json")
    sample = [data[i] for i in sample_index]
    save_f = get_save_name("data_ours/final_data/test.json")
    save(save_f, sample)



def save_sample_index(file, num):
    data = load_text(file)
    seq = np.arange(0, len(data)).tolist()
    sample_index = random.sample(seq, num)
    with open("human_eval_samples/index.txt", "w") as f:
        sample_index.sort()
        for i in sample_index:
            f.write(str(i) + '\n')
    return sample_index

def get_sample_index():
    file = "human_eval_samples/index.txt"
    index = []
    with open(file, "r") as f:
        for line in f:
            index.append(int(line))

    return index



def random_sample(file, sample_index):
    data = load_text(file)
    save_f = get_save_name(file)
    sample = [data[i] for i in sample_index]
    save(save_f, sample)

if __name__ == '__main__':

    # dir_list = ["Keywords2story", "reverse_attention", "Stage_2", "Style_LongLM"]
    dir_list = ["cross_entropy+style_class_loss+distangle_loss+sen_order_loss_after_70+69_1117"]
    file = "test"
    ref = "data_ours/final_data/test.json"
    # save_sample_index(ref, 100)
    sample_index = get_sample_index()
    # save_ref(sample_index)
    for dir in dir_list:
        hypothesis_file_0 = "Output/{}/{}.0".format(dir, file)
        hypothesis_file_1 = "Output/{}/{}.1".format(dir, file)
        hypothesis_file_list = [hypothesis_file_0, hypothesis_file_1]
        for i in hypothesis_file_list:
            random_sample(i, sample_index)
