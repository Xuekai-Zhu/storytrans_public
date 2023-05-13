import json
from tqdm import tqdm


def add2data(file, save):
    with open(save, 'w') as f_s:
        with open(file, 'r') as f:
            data = f.readlines()
            for i, line in enumerate(tqdm(data)):
                item = json.loads(line)
                item["index"] = i
                new_item = json.dumps(item, ensure_ascii=False)
                f_s.write(new_item + '\n')




if __name__ == '__main__':
    # file = "../data_ours/auxiliary_data/train.sen"
    # save = "../data_ours/auxiliary_data/train.sen.add_index"

    # file = "../data_ours/auxiliary_data/test.sen"
    # save = "../data_ours/auxiliary_data/test.sen.add_index"


    file = "../data_ours/auxiliary_data/valid.sen"
    save = "../data_ours/auxiliary_data/valid.sen.add_index"

    add2data(file, save)