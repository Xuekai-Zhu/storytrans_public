from torch.utils.data import Dataset, DataLoader
import os
import json
import torch
import pickle
import random



# def load_vocab(vocab_file):
#     """Loads a vocabulary file into a dictionary."""
#     vocab = collections.OrderedDict()
#     with open(vocab_file, "r", encoding="utf-8") as reader:
#         tokens = reader.readlines()
#     for index, token in enumerate(tokens):
#         token = token.rstrip("\n")
#         vocab[token] = index
#     return vocab

class Data_Encoder(Dataset):

    def __init__(self, file):
        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        self.label_dict = {
            '<LX>': 0,
            '<JY>': 1,
            '<GS>': 2,
        }

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        data = json.loads(self.seqs[index])
        text = data['text']
        text_add_sen = "<SEN>".join(text) + "<SEN>"
        text = "".join(text)
        # title = data['title']
        style = data['style']
        num = data["index"]
        # sen_emd = self.sen_embs[index]
        label = torch.tensor(self.label_dict[style])
        num = torch.tensor(num, dtype=torch.int)

        # device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
        # text = "<STY>" + text
        text_add_sen = "<STY>" + text_add_sen
        return text, text_add_sen, label, num #, sen_emd


class Data_Encoder_En(Dataset):

    def __init__(self, file):
        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        self.label_dict = {
            '<Sp>': 0,
            # '<JK>': 1,
            '<St>': 1,
        }

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        data = json.loads(self.seqs[index])
        text = data['text']
        text_add_sen = " <SEN> ".join(text) + " <SEN>"
        text = " ".join(text)
        # title = data['title']
        style = data['style']
        # num = data["index"]
        # sen_emd = self.sen_embs[index]
        label = torch.tensor(self.label_dict[style])
        # num = torch.tensor(num, dtype=torch.int)

        # device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
        # text = "<STY>" + text
        text_add_sen = "<STY> " + text_add_sen
        return text, text_add_sen, label

class Data_Encoder_Sen(Dataset):

    def __init__(self, file):
        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        self.label_dict = {
            '<LX>': 0,
            '<JY>': 1,
            '<GS>': 2,
        }

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        data = json.loads(self.seqs[index])
        text = data['text']
        text_add_sen = "<SEN>".join(text) + "<SEN>"
        text = "".join(text)
        # title = data['title']
        style = data['style']
        num = data["index"]
        # sen_emd = self.sen_embs[index]
        label = torch.tensor(self.label_dict[style])
        num = torch.tensor(num, dtype=torch.int)

        # device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
        # text = "<STY>" + text
        # text_add_sen = "<STY>" + text_add_sen
        return text, text_add_sen, label, num #, sen_emd


class Data_Encoder_Mask(Dataset):

    def __init__(self, file):
        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        self.label_dict = {
            '<LX>': 0,
            '<JY>': 1,
            '<GS>': 2,
        }

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        data = json.loads(self.seqs[index])
        text_ori = data['text']
        text = data['text_mask']

        text_ori_add_sen = "<SEN>".join(text_ori) + "<SEN>"
        # text_add_sen = "<SEN>".join(text) + "<SEN>"
        text = "".join(text)
        # title = data['title']
        style = data['style']
        num = data["index"]
        # sen_emd = self.sen_embs[index]
        label = torch.tensor(self.label_dict[style])
        num = torch.tensor(num, dtype=torch.int)

        # device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
        # text = "<STY>" + text
        text_ori_add_sen = "<STY>" + text_ori_add_sen
        # text_add_sen = "<STY>" + text_add_sen
        # text_add_sen = "<STY>" + text_add_sen
        return text_ori_add_sen, text, label, num #, sen_emd


class Data_Encoder_Mask_Input(Dataset):

    def __init__(self, file):
        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        self.label_dict = {
            '<LX>': 0,
            '<JY>': 1,
            '<GS>': 2,
        }

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        data = json.loads(self.seqs[index])
        text = data['text_mask']

        text_add_sen = "<SEN>".join(text) + "<SEN>"
        text = "".join(text)
        style = data['style']
        num = data["index"]
        label = torch.tensor(self.label_dict[style])
        num = torch.tensor(num, dtype=torch.int)

        # text_add_sen = "<STY>" + text_add_sen
        text_add_sen = text_add_sen

        return text_add_sen, text, label, num
        # return text_add_sen, text, label

class Data_Encoder_Mask_Input_without_mask(Dataset):

    def __init__(self, file):
        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        self.label_dict = {
            '<LX>': 0,
            '<JY>': 1,
            '<GS>': 2,
        }

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        data = json.loads(self.seqs[index])
        text = data['text']

        text_add_sen = "<SEN>".join(text) + "<SEN>"
        text = "".join(text)
        style = data['style']
        num = data["index"]
        label = torch.tensor(self.label_dict[style])
        num = torch.tensor(num, dtype=torch.int)

        # text_add_sen = "<STY>" + text_add_sen
        text_add_sen = text_add_sen

        return text_add_sen, text, label, num
        # return text_add_sen, text, label


class Data_Encoder_Mask_Input_En(Dataset):

    def __init__(self, file, mask=True):
        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        # self.label_dict = {
        #     '<MT>': 0,
        #     '<JK>': 1,
        #     '<St>': 2,
        # }
        self.label_dict = {
            '<Sp>': 0,
            '<St>': 1,
        }
        self.mask = mask

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        data = json.loads(self.seqs[index])
        if self.mask == False:
            text = data['text']    
        else:
            text = data['text_mask']

        text_add_sen = " <SEN> ".join(text) + " <SEN>"
        text = " ".join(text)
        style = data['style']
        # num = data["index"]
        label = torch.tensor(self.label_dict[style])
        # num = torch.tensor(num, dtype=torch.int)

        # text_add_sen = "<STY>" + text_add_sen
        # text_add_sen = "<STY>" + text_add_sen

        # return text_add_sen, text, label, num
        return text_add_sen, text, label



class Data_Encoder_Fill(Dataset):

    def __init__(self, file, drop_ratio=None):
        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        self.label_dict = {
            '<LX>': 0,
            '<JY>': 1,
            '<GS>': 2,
        }
        self.drop_ratio =drop_ratio
        self.all_key_words = []
        with open("data_ours/auxiliary_data/train.keyword", 'r') as f:
            for line in f:
                self.all_key_words = self.all_key_words + line.strip().split()

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        data = json.loads(self.seqs[index])
        text = data['text']
        text_mask = data['text_mask']
        # text_add_sen = "<SEN>".join(text) + "<SEN>"
        text = "".join(text)
        text_mask = "".join(text_mask)
        keywords = data["mask_word"]
        if self.drop_ratio is not None:
            n = random.random()
            # 删除
            if n < 0.25:
                keywords_num = int(len(keywords) * (1 - self.drop_ratio))
                keywords = random.sample(keywords, keywords_num)
                random.shuffle(keywords)
            # 增加
            elif n >= 0.25 and n < 0.5:
                keywords_num = int(len(keywords) * self.drop_ratio)
                add_keywords = random.sample(self.all_key_words, keywords_num)
                keywords = keywords + add_keywords
                random.shuffle(keywords)

            # 替换
            elif n >= 0.5 and n < 0.75:
                keywords_num = int(len(keywords) * self.drop_ratio)
                add_keywords = random.sample(self.all_key_words, keywords_num)
                sub_keywords = random.sample(keywords, keywords_num)
                for i in sub_keywords:
                    keywords.remove(i)
                keywords = keywords + add_keywords
                random.shuffle(keywords)
        keywords = "<KEY>" + "<KEY>".join(keywords)
        input_text = text_mask + keywords
        # title = data['title']
        # style = data['style']
        # num = data["index"]
        # sen_emd = self.sen_embs[index]
        # label = torch.tensor(self.label_dict[style])
        # num = torch.tensor(num, dtype=torch.int)

        # device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
        # text = "<STY>" + text
        # text_add_sen = "<STY>" + text_add_sen
        # return text, text_add_sen, label, num #, sen_emd
        return text, input_text


class Data_Encoder_Fill_En(Dataset):

    def __init__(self, file, drop_ratio=None):
        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        # self.label_dict = {
        #     '<MT>': 0,
        #     '<JK>': 1,
        #     '<St>': 2,
        # }
        self.label_dict = {
            '<Sp>': 0,
            '<St>': 1,
        }
        self.drop_ratio =drop_ratio
        self.all_key_words = []
        # with open("data_ours/auxiliary_data/train.keyword", 'r') as f:
        #     for line in f:
        #         self.all_key_words = self.all_key_words + line.strip().split()

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        data = json.loads(self.seqs[index])
        text = data['text']
        text_mask = data['text_mask']
        text = " ".join(text)
        text_mask = " ".join(text_mask)
        keywords = data["mask_word"]

        # if self.drop_ratio is not None:
        #     n = random.random()
        #     # 删除
        #     if n < 0.25:
        #         keywords_num = int(len(keywords) * (1 - self.drop_ratio))
        #         keywords = random.sample(keywords, keywords_num)
        #         random.shuffle(keywords)
        #     # 增加
        #     elif n >= 0.25 and n < 0.5:
        #         keywords_num = int(len(keywords) * self.drop_ratio)
        #         add_keywords = random.sample(self.all_key_words, keywords_num)
        #         keywords = keywords + add_keywords
        #         random.shuffle(keywords)
        #
        #     # 替换
        #     elif n >= 0.5 and n < 0.75:
        #         keywords_num = int(len(keywords) * self.drop_ratio)
        #         add_keywords = random.sample(self.all_key_words, keywords_num)
        #         sub_keywords = random.sample(keywords, keywords_num)
        #         for i in sub_keywords:
        #             keywords.remove(i)
        #         keywords = keywords + add_keywords
        #         random.shuffle(keywords)

        keywords = " <KEY> " + " <KEY> ".join(keywords)
        input_text = text_mask + keywords

        return text, input_text



class Data_Encoder_Fill_Res(Dataset):

    def __init__(self, result, file):
        # load result
        with open(result, 'r') as f:
            self.res = f.readlines()

        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        # self.label_dict = {
        #     '<LX>': 0,
        #     '<JY>': 1,
        #     '<GS>': 2,
        # }

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        result = json.loads(self.res[index])
        text_mask = result["text"]
        style = result['style']



        data = json.loads(self.seqs[index])
        # text = data['text']
        # text_mask = data['text_mask']
        # text_add_sen = "<SEN>".join(text) + "<SEN>"
        # text = "".join(text)
        # text_mask = "".join(text_mask)
        keywords = data["mask_word"]
        keywords = "<KEY>" + "<KEY>".join(keywords)
        input_text = text_mask + keywords
        return input_text, style


class Data_Encoder_Fill_Res_En(Dataset):

    def __init__(self, result, file):
        # load result
        with open(result, 'r') as f:
            self.res = f.readlines()

        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        # self.label_dict = {
        #     '<LX>': 0,
        #     '<JY>': 1,
        #     '<GS>': 2,
        # }

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        result = json.loads(self.res[index])
        text_mask = result["text"]
        style = result['style']



        data = json.loads(self.seqs[index])
        # text = data['text']
        # text_mask = data['text_mask']
        # text_add_sen = "<SEN>".join(text) + "<SEN>"
        # text = "".join(text)
        # text_mask = "".join(text_mask)
        keywords = data["mask_word"]
        keywords = " <KEY> " + " <KEY> ".join(keywords)
        input_text = text_mask + keywords
        return input_text, style

# class Data_Encoder_BT(Dataset):
#
#     def __init__(self, file, bt_file):
#         # load data
#         with open(file, 'r') as f:
#             self.seqs = f.readlines()
#
#         with open(bt_file, 'r') as f:
#             self.bt = f.readlines()
#         # with open(sentence_emd, 'rb') as f:
#         #     self.sen_embs = pickle.load(f)
#         # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
#         # self.max_length = max_length
#         self.label_dict = {
#             '<LX>': 0,
#             '<JY>': 1,
#             '<GS>': 2,
#         }
#
#     def __len__(self):
#         """Denotes the total number of samples"""
#         return len(self.seqs)
#
#     def __getitem__(self, index):
#         data = json.loads(self.seqs[index])
#         bt_sens = json.loads(self.bt[index])
#         text = data['text']
#         # bt_list = bt_sens["text"]
#         text_add_sen = "<SEN>".join(text) + "<SEN>"
#         text = "".join(text)
#         # title = data['title']
#         style = data['style']
#         num = data["index"]
#         # sen_emd = self.sen_embs[index]
#
#
#         label = torch.tensor(self.label_dict[style])
#         num = torch.tensor(num, dtype=torch.int)
#         # device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
#
#         return text, text_add_sen, label, num, bt_sens #, sen_emd


class Data_Encoder_eval(Dataset):

    def __init__(self, file):
        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        self.label_dict = {
            '<LX>': 0,
            '<JY>': 1,
            '<GS>': 2,
        }

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        data = json.loads(self.seqs[index])
        text = data['text']
        # text = "<SEN>".join(text) + "<SEN>"
        # title = data['title']
        style = data['style']
        # num = data["index"]
        # sen_emd = self.sen_embs[index]


        label = torch.tensor(self.label_dict[style])
        # num = torch.tensor(num, dtype=torch.int)
        # device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')

        return text, label


class Data_Encoder_eval_en(Dataset):

    def __init__(self, file):
        # load data
        with open(file, 'r') as f:
            self.seqs = f.readlines()
        # with open(sentence_emd, 'rb') as f:
        #     self.sen_embs = pickle.load(f)
        # self.tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")
        # self.max_length = max_length
        self.label_dict = {
            '<Sp>': 0,
            # '<JK>': 1,
            '<St>': 1,
        }

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.seqs)

    def __getitem__(self, index):
        data = json.loads(self.seqs[index])
        # text = " ".join(data['text'])
        text = data['text']
        # text = "<SEN>".join(text) + "<SEN>"
        # title = data['title']
        style = data['style']
        # num = data["index"]
        # sen_emd = self.sen_embs[index]


        label = torch.tensor(self.label_dict[style])
        # num = torch.tensor(num, dtype=torch.int)
        # device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')

        return text, label


if __name__ == '__main__':
    dataset = Data_Encoder("data_ours/auxiliary_data/train.sen.add_index")
    training_generator = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)
    for i, (batch_text, style, index) in enumerate(training_generator):


        print('----------------------------------------------------')

    # tokenizer = BertTokenizer.from_pretrained("pretrained_model/bert-base-chinese")