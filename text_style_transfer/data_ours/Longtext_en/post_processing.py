import json


def check_keyword(files):
    for file in files:
        data = load_file(file)
        for i, j in enumerate(data):
            masks = j["mask_word"]
            for w in masks:
                if len(w) <= 2:
                    print(w)
                    print(i)
                    print("---")



def load_file(file):
    all_dict = []
    with open(file, 'r') as f:
        for i in f:
            all_dict.append(json.loads(i))
    return all_dict

def check_repeat(files):
    all_text = []
    n = 0
    with open(files, 'r') as f:
        for line in f:
            text = " ".join(json.loads(line)["text"])
            if text in all_text:
                n += 1
                print(text)
            else:
                all_text.append(text)
        
        print("----------------", "\n", n)

def check_author(file):
    sp_list = []
    st_list = []
    # for i in file:
    data = load_file(file)
    for j in data:
        style = j["style"]
        if style == "<Sp>":
            sp_list.append(j)
        elif style == "<St>":
            st_list.append(j)
            
    print(len(sp_list))
    print(len(st_list))
    
    
    

if __name__ == '__main__':
    # saves = ["style_merge_data/train.mask", "style_merge_data/vaild.mask", "style_merge_data/test.mask"]
    # check_keyword(saves)
    # file = "data/shakespeare/all_data"
    
    # file = "sp+story/classifier_data/train"
    file = "sp+story/classifier_data/test"
    
    check_author(file)