import random
from tqdm import tqdm
import json



def load_file(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return data

def merge_for_style_classifier(files, save):
    sp_train, sp_test = files[0], files[1]
    st_train, st_test = files[2], files[3]
    
    sp_train_data = load_file(sp_train)
    sp_test_data = load_file(sp_test)
    st_train_data = load_file(st_train)
    st_test_data = load_file(st_test)
    
    train_num = len(sp_train_data)
    test_num = len(sp_test_data)
    
    st_train_data = random.sample(st_train_data, train_num)
    st_test_data = random.sample(st_test_data, test_num)
    
    train_data = sp_train_data + st_train_data
    test_data = sp_test_data + st_test_data
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    
    with open(save[0], "w") as f:
        for i in train_data:
            f.write(i)
            
    with open(save[1], "w") as f:
        for i in test_data:
            f.write(i)
    
    
    

def pick_data_for_train(files, save):
    sp, st = files[0], files[1]
    sp_data = load_file(sp)
    st_data = load_file(st)

    # random.shuffle(st_data)
    # random.shuffle(jk_data)
    # 


    all_data = []

    # st_num = int(len(st_data) * 0.05)
    # jk_num = int(len(jk_data) * 0.5)
    # mt_num = int(len(mt_data) * 1)
    train_num = len(sp_data)

    all_data = random.sample(st_data, train_num) + sp_data

    random.shuffle(all_data)
    with open(save, 'w') as f:
        for i in tqdm(all_data):
            f.write(i)



def pick_data_for_test(files, ref, save):
    test = load_file(files)
    test_num = len(load_file(ref)) // 8
    
    test_data  = random.sample(test, test_num)
    
    with open(save, 'w') as f:
        for i in tqdm(test_data):
            f.write(i)
    
    
    
            
            
if __name__ == '__main__':
    # style classifier data
    # all_files = ["data/shakespeare/train", "data/shakespeare/test", "roc_story/stories_keywords/train", "roc_story/stories_keywords/test"]
    # saves = ["sp+story/classifier_data/train", "sp+story/classifier_data/test"]
    # merge_for_style_classifier(all_files, saves)
    
    # transfer data
    files = ["data/shakespeare/all_data.filter.keywords", "roc_story/stories_keywords/train"]
    save = "sp+story/style_transfer_data/train"
    # pick_data_for_train(files, save)
    input = "roc_story/stories_keywords/test"
    save_test = "sp+story/style_transfer_data/test"
    
    input = "roc_story/stories_keywords/vaild"
    save_test = "sp+story/style_transfer_data/vaild"
    pick_data_for_test(input, save, save_test)
    
    
    
    