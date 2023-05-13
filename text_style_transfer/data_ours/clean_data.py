import os
import json



def replace_key(file_list):
    new_id = {
        "jinyong": "<JY>",
        "luxun": "<LX>",
        "故事": "<GS>",
    }

    for file in file_list:
        with open(file, 'r') as f:
            data = f.readlines()
        with open(file, 'w') as f:
            for line in data:
                item = json.loads(line)
                style = item["style"]
                item["style"] = new_id[style]
                w_item = json.dumps(item, ensure_ascii=False)
                f.write(w_item + '\n')


def filter_other_data(file, save):
    for f_in, s_out in zip(file, save):
        s = open(s_out, 'w')
        with open(f_in, 'r') as f:
            data = f.readlines()
            for line in data:
                item = json.loads(line)
                style = item["style"]
                if style in ['<JY>', '<LX>']:
                    print(style)
                    continue
                s.write(line)

        s.close()



if __name__ == '__main__':
    train_set = 'final_data/train.json'
    valid_set_v1 = 'final_data_v1/valid.json'
    test_set_v1 = 'final_data_v1/test.json'


    valid_set = 'final_data/valid.json'
    test_set = 'final_data/test.json'



    # file_list = [train_set, valid_set, test_set]
    # replace_key(file_list)
    file_list = [valid_set_v1, test_set_v1]
    save_list = [valid_set, test_set]
    filter_other_data(file_list, save_list)

