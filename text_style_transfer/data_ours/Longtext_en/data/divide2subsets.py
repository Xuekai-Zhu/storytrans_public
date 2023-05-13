import random
from tqdm import tqdm

def load_file(file):
    with open(file, 'r') as f:
        data = f.readlines()

    return data

def divide_into_subset(input_file, file_list):
    train, valid, test = file_list[0], file_list[1], file_list[2]
    train_f = open(train, 'w')
    valid_f = open(valid, 'w')
    test_f = open(test, 'w')


    data = load_file(input_file)
    random.shuffle(data)
    data_num = len(data)

    train_num = int(data_num * 0.8)
    valid_num = int(data_num * 0.1)
    test_num = int(data_num * 0.1)

    train_data = data[:train_num]
    valid_data = data[train_num:train_num + valid_num]
    test_data = data[train_num + valid_num:]

    test_data = valid_data + test_data
    
    
    for item in tqdm(train_data):
        train_f.write(item)

    for item in tqdm(valid_data):
        valid_f.write(item)

    for item in tqdm(test_data):
        test_f.write(item)


    train_f.close()
    valid_f.close()
    test_f.close()


if __name__ == '__main__':
    mt_file = "mt+jk/mt.v1"
    jk_file = "mt+jk/jk.v1"
    sp_file = "shakespeare/all_data.filter.keywords"
    mt_subsets = ["mt_subsets/train", "mt_subsets/vaild", "mt_subsets/test"]
    jk_subsets = ["jk_subsets/train", "jk_subsets/vaild", "jk_subsets/test"]
    sp_subsets = ["shakespeare/train", "shakespeare/vaild", "shakespeare/test"]
    # divide_into_subset(mt_file, mt_subsets)
    # divide_into_subset(jk_file, jk_subsets)
    divide_into_subset(sp_file, sp_subsets)
