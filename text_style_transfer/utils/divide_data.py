import json
from tqdm import tqdm


def load_text(file):
    lx_list = []
    jy_list = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        for line in data:
            item = json.loads(line)
            style = item["style"]
            if style == "<LX>":
                lx_list.append(item)
            elif style == "<JY>":
                jy_list.append(item)
    # return hypothesis_list
    return lx_list, jy_list

def write2text(input, save):
    with open(save, 'w') as f:
        for i in tqdm(input):
            new_item = json.dumps(i, ensure_ascii=False)
            f.write(new_item + "\n")




def divide_data(file, save_1, save_2):
    lx_list, jy_list = load_text(file)
    write2text(lx_list, save_1)
    write2text(jy_list, save_2)


if __name__ == '__main__':
    file = "../data_ours/auxiliary_data/train.sen.add_index.mask"
    save_1 = "../data_ours/auxiliary_data/train.sen.add_index.mask.lx"
    save_2 = "../data_ours/auxiliary_data/train.sen.add_index.mask.jy"
    divide_data(file, save_1, save_2)