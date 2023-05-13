import json



def load_text(file):
    hypothesis_list = []
    with open(file, 'r') as fin:
        data = fin.readlines()
        for line in data:
            item = json.loads(line)
            text = item["text"]
            hypothesis_list.append(text)
    return hypothesis_list

def print_sen(file):
    data = load_text(file)
    for i in data:
        print(i)

if __name__ == '__main__':
    reference = "../data_ours/final_data/test.json"

    dir = "../pred_mask_stage_2/train_sen_mse_add_mask_in_input_ablation_2_after_49_1122-s-29824-s-LongLM-7456"
    file_0 = "{}/test_sen_mse_add_mask_in_input.0".format(dir)
    file_1 = "{}/test_sen_mse_add_mask_in_input.1".format(dir)
    file_list = [file_0, file_1]
    print_sen(reference)