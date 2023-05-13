import random
from tqdm import tqdm
import json

def load_file(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return data


def pick_data_for_train(files, save):
    st, jk, mt = files[0], files[1], files[2]
    st_data = load_file(st)
    jk_data = load_file(jk)
    mt_data = load_file(mt)

    # random.shuffle(st_data)
    # random.shuffle(jk_data)
    # random.shuffle(mt_data)


    all_data = []

    st_num = int(len(st_data) * 0.05)
    jk_num = int(len(jk_data) * 0.5)
    mt_num = int(len(mt_data) * 1)

    all_data = random.sample(st_data, st_num) + random.sample(jk_data, jk_num) + random.sample(mt_data, mt_num)

    random.shuffle(all_data)
    with open(save, 'w') as f:
        for i in tqdm(all_data):
            f.write(i)


def build_v_and_t(trainset, ori_files, save_files):
    vaild, test = ori_files[0], ori_files[1]
    s_vaild, s_test = save_files[0], save_files[1]

    train_data = load_file(trainset)
    vaild_data = load_file(vaild)
    test_data = load_file(test)

    tran_num = len(train_data)

    vaild_num = int(tran_num / 8)
    test_num = int(tran_num / 8)

    v_data = random.sample(vaild_data, vaild_num)
    t_data = random.sample(test_data, test_num)

    with open(s_vaild, 'w') as f:
        for i in v_data:
            f.write(i)

    with open(s_test, 'w') as f:
        for i in t_data:
            f.write(i)




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



def compute_author(file):
    data = load_file(file)
    st_list = []
    mt_list = []
    jk_list = []
    for i in data:
        item = json.loads(i)
        style = item["style"]

        if style == "<St>":
            st_list.append(item)
        elif style == "<MT>":
            mt_list.append(item)
        elif style == "<JK>":
            jk_list.append(item)

    print("故事：{}".format(len(st_list)))
    print("马克吐温：{}".format(len(mt_list)))
    print("jk罗玲：{}".format(len(jk_list)))





if __name__ == '__main__':
    train_files = ["keywords_dataset/stories/train", "keywords_dataset/jk/train", "keywords_dataset/mt/train"]
    train_save = "style_merge_data/train"
    test_files = ["keywords_dataset/stories/test", "keywords_dataset/jk/test", "keywords_dataset/mt/test"]
    test_save = "data_for_style_classifier/test"
    pick_data_for_train(train_files, train_save)
    pick_data_for_train(test_files, test_save)

    trainset = "style_merge_data/train"
    ori_files = ["keywords_dataset/stories/vaild", "keywords_dataset/stories/test"]
    subsets = ["style_merge_data/vaild", "style_merge_data/test"]
    build_v_and_t(trainset, ori_files, subsets)
    # train_file_st = "style_merge_data/train"
    # test_files_sc = "data_for_style_classifier/test"
    # compute_author(test_files_sc)


