import json, pickle
from data_set import Data_Encoder, Data_Encoder_Sen, Data_Encoder_Mask, Data_Encoder_Fill_Res_En, Data_Encoder_Fill_Res, Data_Encoder_Mask_Input_En, Data_Encoder_Fill_En, Data_Encoder_En
from T5Tokenizer import T5Tokenizer
from modeling_t5 import T5ForConditionalGeneration, Style_Classifier, T5ForLongText_ST_Sen_Sty, T5ForLongText_ST_Sen_Sty_En, LongTextST_Style_Attention
from modeling_t5 import T5ForLongText_ST_Sen_token_mean, T5ForLongText_ST_Sen_Sty_token_mean_En
from transformers import AutoModel, AutoConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import os, torch, time
import torch.nn.functional as F
import numpy as np
# from pytorch_metric_learning.losses import NTXentLoss
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.utils.rnn import pad_sequence
from random import shuffle




class LongTextStyleTrans_En(nn.Module):
    def __init__(self, config):
        super(LongTextStyleTrans_En, self).__init__()
        self.config = config

    def load_data(self, dataset, sen_embs, batch_size):
        # load data
        dataset = Data_Encoder(dataset)
        data_generator = DataLoader(dataset, batch_size, shuffle=True)

        # style_list = list(dataset.label_dict)
        with open(sen_embs, 'rb') as f:
            sen_embs_list = pickle.load(f)

        return data_generator, sen_embs_list#, style_list

    def load_data_test(self, dataset, batch_size):
        # load data
        dataset = Data_Encoder(dataset)
        data_generator = DataLoader(dataset, batch_size, shuffle=False)

        return data_generator

    def load_data_bt(self, dataset, batch_size, shuffle=True, drop_last=True):
        dataset = Data_Encoder_En(dataset)
        data_generator = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)

        # sen_bt_list = []
        # with open(sen_bt, 'rb') as f:
        #     sen_bt = f.readlines()
        #     for sens in sen_bt:
        #         item = json.loads(sens)
        #         text = item["text"]
        #         sen_bt_list.append(text)

        return data_generator

    def load_data_ori(self, dataset, batch_size):
        dataset = Data_Encoder_Sen(dataset)
        data_generator = DataLoader(dataset, batch_size, shuffle=True)
        return data_generator

    def load_data_ori_test(self, dataset, batch_size):
        dataset = Data_Encoder_Sen(dataset)
        data_generator = DataLoader(dataset, batch_size, shuffle=False)
        return data_generator

    def load_data_mask(self, dataset, batch_size, shuffle=True):
        dataset = Data_Encoder_Mask(dataset)
        data_generator = DataLoader(dataset, batch_size, shuffle=shuffle)
        return data_generator

    def load_data_mask_test(self, dataset, batch_size):
        dataset = Data_Encoder_Mask(dataset)
        data_generator = DataLoader(dataset, batch_size, shuffle=False)
        return data_generator

    def load_data_mask_in_input(self, dataset, batch_size, shuffle=True, drop_last=True, mask=True):
        dataset = Data_Encoder_Mask_Input_En(dataset, mask=mask)
        data_generator = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
        # data_generator = DataLoader(dataset, batch_size, shuffle=False, drop_last=drop_last)
        return data_generator

    def get_sen_id(self, tokenizer):
        id = tokenizer("<SEN>",
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length).input_ids.to(self.config.device)[0, 0]
        return id


    def get_sty_id(self, tokenizer):
        id = tokenizer("<STY>",
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length).input_ids.to(self.config.device)[0, 0]
        return id

    def load_style_classifier(self, tokenizer):
        style_classifer = Style_Classifier(self.config).to(self.config.device)
        style_classifer.encoder.resize_token_embeddings(len(tokenizer))
        style_classifer.load_state_dict(torch.load(self.config.pretrained_sc))
        style_classifer.eval()

        return style_classifer

    def print_tip_for_train(self, num_step):
        print('------------------create model---------------------------')
        print('epoch num : {}'.format(self.config.epoch))
        print('step num : {}'.format(num_step))
        print('batch size : {}'.format(self.config.batch_size))
        print('learning rate : {}'.format(self.config.learning_rate))
        print('begin training')

    def get_sen_position(self, batch_text, tokenizer):
        d_1 = []
        d_2 = []
        tokens = [tokenizer.tokenize(i) for i in list(batch_text)]
        for j, texts in enumerate(tokens):
            for i, token in enumerate(texts):
                if token == self.config.sen_token:
                    d_1.append(j)
                    d_2.append(i)

        return d_1, d_2

    def get_ids(self, batch_text, tokenizer, attention_mask=False, sen_id=None):
        ids = tokenizer(batch_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length)


        input_ids = ids.input_ids.to(self.config.device)
        mask = ids.attention_mask.to(self.config.device)

        # if input_ids.size(-1) == 512 and sen_id != None:
            # for i in range(len(batch_text)):
            #     if input_ids[i, -1] == tokenizer.eos_token_id and input_ids[i, -2] != sen_id:
            #         input_ids[i, -2] = sen_id

        if attention_mask:
            return input_ids, mask
        else:
            return input_ids

    def get_ids_and_sentence_order(self, batch_text, tokenizer, sen_id, attention_mask=False, sentence_type=False):
        ids = tokenizer(batch_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length)
        # test_1 = ids["input_ids"][0].numpy().tolist()
        # tokens = tokenizer.convert_ids_to_tokens(test_1)
        type_mats = torch.zeros(ids["input_ids"].shape).long()
        sen_index = torch.where(ids["input_ids"] == sen_id.detach().cpu())
        
        # get sentence order label
        sentence_order_label = []
        step = 0
        for i in range(self.config.batch_size):
            batch_index = sen_index[0] == i
            num = torch.sum(batch_index)
            sentence_order_label.append(torch.arange(num) + 1)
            if sentence_type:
                start = 0
                for j in range(num):
                    end = sen_index[1][step]
                    type_mats[i, start:end+1] = j + 1
                    # print(type_mats[i, start:end+1])
                    start = end + 1
                    step += 1
        sentence_order_labels = pad_sequence(sentence_order_label, batch_first=True)   
        if attention_mask:
            return ids.input_ids.to(self.config.device), ids.attention_mask.to(self.config.device), type_mats.to(self.config.device)
        elif sentence_type:
            return ids.input_ids.to(self.config.device), type_mats.to(self.config.device), sentence_order_labels.to(self.config.device).long()
        else:
            return ids.input_ids.to(self.config.device), sentence_order_labels.to(self.config.device).long()
        
    def get_shuffle_sen_ids(self, batch_text, tokenizer, sen_id, sentence_type=False):
        batch_sentences = [text_i.split("<SEN>") for text_i in batch_text]
        pair_list = []
        order_list = []
        for i in batch_sentences:
            single_batch_list = []
            for index, sen in enumerate(i):
                if len(sen) == 0:
                    continue
                single_batch_list.append((sen + "<SEN>", index))
            shuffle(single_batch_list)
            order_list.append(torch.tensor([i[-1] for i in single_batch_list], dtype=torch.long) + 1)
            pair_list.append(single_batch_list)  
        
        shuffle_batch_text = ["".join([pair[0] for pair in i]) for i in pair_list]

        ids = tokenizer(shuffle_batch_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length)
        type_mats = torch.zeros(ids["input_ids"].shape).long()
        sen_index = torch.where(ids["input_ids"] == sen_id.detach().cpu())
        
        if sentence_type:
            step = 0

            for i in range(self.config.batch_size):
                batch_index = sen_index[0] == i
                num = torch.sum(batch_index)
                start = 0
                for j in range(num):
                    end = sen_index[1][step]
                    b = pair_list[i][j][-1]
                    type_mats[i, start:end+1] = pair_list[i][j][-1] + 1
                    # print(type_mats[i, start:end+1])
                    start = end + 1
                    step += 1
        
        sentence_order_labels = pad_sequence(order_list, batch_first=True)   
        if sentence_type:
            return ids.input_ids.to(self.config.device), type_mats.to(self.config.device), sentence_order_labels.to(self.config.device)
        else:
            return ids.input_ids.to(self.config.device), sentence_order_labels.to(self.config.device)     
        
    def get_sen_order_loss(self, predict_logits, labels):
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        real_label = labels.view(-1) - 1
        loss = loss_fct(predict_logits.view(-1, predict_logits.size(-1)), real_label)
        return loss
    
    def get_content_label(self, sen_embs_list, index):
        content_label = [sen_embs_list[i] for i in index]
        batch_content_label = [np.mean(sen_embs_list[i], axis=0, keepdims=True) for i in index]

        content_label = np.concatenate(content_label, axis=0)
        batch_content_label = np.concatenate(batch_content_label, axis=0)

        content_label = torch.tensor(content_label).to(self.config.device)
        batch_content_label = torch.tensor(batch_content_label).to(self.config.device)
        return content_label, batch_content_label

    def get_trans_style(self, labels):
        candidate_style = torch.arange(0, self.config.style_num, 1).expand(self.config.batch_size, -1).to(self.config.device)
        candidate_style = candidate_style[candidate_style != labels.unsqueeze(-1)].view(self.config.batch_size, -1)
        random_posi = torch.randint_like(labels, 0, 2)
        transfer_style = candidate_style[torch.arange(0, self.config.batch_size).unsqueeze(-1), random_posi.unsqueeze(-1)].squeeze(-1)

        return transfer_style

    def get_bt_content_label(self, sen_bt_list, index, model, tokenizer, batch_sen=True):
        # sen_bts_ori = [sen_bt_list[i] for i in index]
        sen_bts = [j for i in index for j in sen_bt_list[i]]
        sen_num = torch.cat([torch.ones(len(sen_bt_list[j])) * i for i, j in enumerate(index)], dim=0)
        batch_sen_list = []
        with torch.no_grad():
            ids, mask = self.get_ids(sen_bts, tokenizer, attention_mask=True)
            num = torch.sum(mask, dim=-1, keepdim=True)
            last_hidden = model.encoder(input_ids=ids).last_hidden_state
            last_hidden = torch.mul(last_hidden, mask.float().unsqueeze(-1))
            last_hidden = torch.div(torch.sum(last_hidden, dim=1), num)
            if batch_sen:
                for i in range(self.config.batch_size):
                    single_sample = torch.mean(last_hidden[sen_num == i], dim=0, keepdim=True)
                    batch_sen_list.append(single_sample)
                batch_sen = torch.cat(batch_sen_list, dim=0)

                return last_hidden, batch_sen
            else:
                return last_hidden


    def get_sen_emb_loss(self, sen_emb, label, model):
        sen_emb = model.project_content(sen_emb)

        # loss_cos = nn.CosineEmbeddingLoss(reduction="sum")
        # target = torch.ones(sen_emb.size(0)).to(self.config.device)
        # loss_1 = loss_cos(sen_emb, label, target) / self.config.batch_size

        cos_sim = F.cosine_similarity(sen_emb, label, dim=-1)
        loss = 1 - cos_sim
        loss[loss < self.config.margin] = self.config.margin
        # loss = torch.where(loss < self.config.margin, self.config.margin, loss)
        loss = torch.sum(loss) / self.config.batch_size


        return loss

    def get_sen_mse_loss(self, sen_emb, label):
        # sen_emb = model.encoder.project_content_768(sen_emb)
        # loss_cos = nn.MSELoss(reduction="sum")
        loss_cos = nn.MSELoss()
        # target = torch.ones(sen_emb.size(0)).to(self.config.device)
        # loss = loss_cos(sen_emb, label) / self.config.batch_size
        loss = loss_cos(sen_emb, label)

        return loss

    def get_sentence_mse_loss(self, sentence_emb, label):
        # sen_emb = model.project_content_768(sen_emb)
        # a = torch.isnan(sentence_emb)
        # b = torch.isnan(label)
        # print(torch.sum(a))
        # print(torch.sum(b))
        # step = torch.norm((sentence_emb - label), p=2, dim=1)
        # loss = torch.mean(step)
        loss_cos = nn.MSELoss()
        # loss_cos = nn.MSELoss()
        loss = loss_cos(sentence_emb, label)
        return loss

    # def get_style_contrastive_loss(self, style_representation, model, style):
    #     # detach_style = style_representation.detach()
    #     loss_func = NTXentLoss()
    #     emb_label = torch.arange(0, self.config.style_num, 1).to(self.config.device)
    #     label = torch.cat((style, emb_label), dim=0)
    #     emb = torch.cat((style_representation, model.encoder.style_embedding.weight), dim=0)
    #     # emb = torch.cat((style_representation, model.mid_module.style_embedding.weight), dim=0)
    #     loss = loss_func(emb, label)
    #     return loss

    def get_cross_entropy_loss(self, logits, label, tokenizer, margin=False):
        # loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=config.pad_token_id)
        loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        # loss = loss_fct(logits.permute(0, 2, 1), label) / config.batch_size
        # loss_1 = loss_fct(logits.permute(0, 2, 1), label)
        # a = label.view(-1)
        # b = logits.view(-1, logits.size(-1))
        loss = loss_fct(logits.view(-1, logits.size(-1)), label.view(-1))
        # if margin:
        #     # loss = self.config.margin
        #     loss = torch.max(loss, torch.tensor(self.config.margin).to(loss.device))
        return loss

    def get_cycle_loss(self, batch_content_bert, encoder_out, model, affine=True, margin=False):
        # content_1 = torch.mean(content_1, dim=1)
        if affine:
            encoder_out = model.project_cycle_content(encoder_out)
        cycle_content = torch.mean(encoder_out, dim=1)

        cos_sim = F.cosine_similarity(batch_content_bert, cycle_content, dim=-1)
        loss = 1 - cos_sim
        if margin:
            loss[loss < self.config.margin] = self.config.margin
        # loss = torch.where(loss < self.config.margin, self.config.margin, loss)
        loss = torch.sum(loss) / self.config.batch_size

        return loss

    def get_style_loss(self, pred, label):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pred, label)
        return loss

    def get_content_disturibution(self, model, hidden, ids, sen_ids):
        multi_normal = MultivariateNormal(model.noraml_mean_weight, torch.diag(model.noraml_covariance_matrix))
        content_hidden = hidden[ids != sen_ids]
        loss_all = -multi_normal.log_prob(content_hidden)
        loss = torch.mean(loss_all)
        if loss < 0:
            loss = -loss
        return loss

    def get_content_single_normal(self, hidden, ids, sen_ids):
        # multi_normal = MultivariateNormal(model.noraml_mean_weight, model.noraml_covariance_matrix)
        # loss_all = -multi_normal.log_prob(hidden)
        # content_hidden = hidden[ids != sen_ids]
        content_hidden = hidden[ids != sen_ids][:, :384]
        var = torch.var(content_hidden.permute(1, 0), dim=-1)
        mu = torch.mean(content_hidden.permute(1, 0), dim=-1)

        multi_normal = MultivariateNormal(mu, torch.diag(var))
        standard_multi_normal = MultivariateNormal(torch.zeros_like(mu, device=mu.device), torch.eye(var.size(-1), device=var.device))

        kl_loss = torch.distributions.kl.kl_divergence(multi_normal, standard_multi_normal) / self.config.batch_size

        # kl_div = torch.log(var) - mu.pow(2) - var + 1
        # kl_loss = torch.log(kl_div)
        return kl_loss

    def get_content_normal_half_sen(self, hidden):
        # multi_normal = MultivariateNormal(model.noraml_mean_weight, model.noraml_covariance_matrix)
        # loss_all = -multi_normal.log_prob(hidden)
        # content_hidden = hidden[ids != sen_ids]
        # content_hidden = hidden[ids != sen_ids][:, :384]
        var = torch.var(hidden.permute(1, 0), dim=-1)
        mu = torch.mean(hidden.permute(1, 0), dim=-1)

        multi_normal = MultivariateNormal(mu, torch.diag(var))
        standard_multi_normal = MultivariateNormal(torch.zeros_like(mu, device=mu.device), torch.eye(var.size(-1), device=var.device))

        kl_loss = torch.distributions.kl.kl_divergence(multi_normal, standard_multi_normal) / self.config.batch_size

        # kl_div = torch.log(var) - mu.pow(2) - var + 1
        # kl_loss = torch.log(kl_div)
        return kl_loss


    def get_batch_mid_sen_loss(self, content_hidden, ids_add_sen, sen_id):
        distutils_list = []
        sen_posi = torch.where(ids_add_sen == sen_id)
        for i in range(self.config.batch_size):
            num_dim = torch.sum(sen_posi[0] == i)
            single_content = content_hidden[i, :num_dim]
            # index = torch.where(single_content != 0)
            # data_point = single_content[index]
            var = torch.var(single_content)
            mu = torch.mean(single_content)
            # multi_normal = MultivariateNormal(mu, torch.diag(var))
            multi_normal = torch.distributions.normal.Normal(mu, var)
            distutils_list.append(multi_normal)

        loss_list = []

        for i in range(self.config.batch_size):
            d_i = distutils_list[i]
            for j in range(1, self.config.batch_size):
                if (i+j) >= self.config.batch_size:
                    break
                d_j = distutils_list[i + j]
                kl_loss = torch.distributions.kl.kl_divergence(d_i, d_j)
                loss_list.append(kl_loss.unsqueeze(0))
        loss = torch.sum(torch.cat(loss_list)) / self.config.batch_size

        return loss

    # def get_batch_distance(self, content_hidden, ids_add_sen, sen_id, no_pading=False):
    #     l_fct = nn.MSELoss()
    #     distutils_list = []
    #     sen_posi = torch.where(ids_add_sen == sen_id)
    #     batch_size = content_hidden.size(0)
    #     if no_pading is True:
    #         for i in range(batch_size):
    #             single_content = content_hidden[i]
    #             distutils_list.append(single_content)
    #     else:
    #         # 提取每个样本中的sen hidden state的表示，并且取mean，然后对每个sample之中的互相算距离，优化目标为减小这个距离。
    #         for i in range(batch_size):
    #             num_dim = torch.sum(sen_posi[0] == i)
    #             single_content = content_hidden[i, :num_dim]
    #             single = torch.mean(single_content, dim=0, keepdim=True)
    #             # index = torch.where(single_content != 0)
    #             # data_point = single_content[index]
    #             # var = torch.var(single_content)
    #             # mu = torch.mean(single_content)
    #             # multi_normal = MultivariateNormal(mu, torch.diag(var))
    #             # multi_normal = torch.distributions.normal.Normal(mu, var)
    #             distutils_list.append(single)

    #     loss_list = []

    #     for i in range(batch_size):
    #         d_i = distutils_list[i]
    #         for j in range(1, batch_size):
    #             if (i + j) >= batch_size:
    #                 break
    #             d_j = distutils_list[i + j]
    #             mse = l_fct(d_i, d_j)
    #             # cos_sim = F.cosine_similarity(d_i, d_j, dim=-1)
    #             # cos_loss = 1 - cos_sim
    #             # kl_loss = torch.distributions.kl.kl_divergence(d_i, d_j)
    #             # loss_list.append(cos_loss)
    #             loss_list.append(mse.unsqueeze(0))
    #     # a = torch.cat(loss_list)
    #     loss = torch.sum(torch.cat(loss_list)) / batch_size

    #     return loss

    def get_batch_distance(self, content_hidden, ids_add_sen, sen_id, style, no_pading=False, all_token=False):
        l_fct = nn.MSELoss()
        # distutils_list = []
        # sen_posi = torch.where(ids_add_sen == sen_id)
        
        # if no_pading is True:
        #     for i in range(self.config.batch_size):
        #         single_content = content_hidden[i]
        #         distutils_list.append(single_content)
        # elif all_token == True:
        #     a = torch.sum(content_hidden, dim=-1)
        #     # b = a > 0
        #     num = torch.sum(a > 0, dim=-1)
        #     batch_sample = torch.div(torch.sum(content_hidden, dim=1), num.unsqueeze(-1))
            
        #     for i in range(self.config.batch_size):
        #         single_content = batch_sample[i]
        #         distutils_list.append(single_content)
        # else:
        #     for i in range(self.config.batch_size):
        #         num_dim = torch.sum(sen_posi[0] == i)
        #         single_content = content_hidden[i, :num_dim]
        #         single = torch.mean(single_content, dim=0, keepdim=True)
        #         # index = torch.where(single_content != 0)
        #         # data_point = single_content[index]
        #         # var = torch.var(single_content)
        #         # mu = torch.mean(single_content)
        #         # multi_normal = MultivariateNormal(mu, torch.diag(var))
        #         # multi_normal = torch.distributions.normal.Normal(mu, var)
        #         distutils_list.append(single)
        sen_num = torch.sum(torch.sum(content_hidden != 0, dim=-1) != 0, dim=-1, keepdim=True)
        batch_representation = torch.div(torch.sum(content_hidden, dim=1), sen_num)
        
        loss_list = []
        for i, style_i in enumerate(style):
            index = torch.where(style != style_i)
            if not index[0].numel():
                continue
            opposed_tensor = batch_representation[index]
            loss_list.append(l_fct(opposed_tensor, batch_representation[i].expand(opposed_tensor.shape)))
        
        # if len(loss_list) == 0:
        #     loss = torch.zeros(1).to(self.config.device)
        # else:
        loss = torch.mean(torch.tensor(loss_list)) / 2

        # for i in range(self.config.batch_size):
        #     d_i = distutils_list[i]
        #     for j in range(1, self.config.batch_size):
        #         if (i + j) >= self.config.batch_size:
        #             break
        #         d_j = distutils_list[i + j]
        #         mse = l_fct(d_i, d_j)
        #         # cos_sim = F.cosine_similarity(d_i, d_j, dim=-1)
        #         # cos_loss = 1 - cos_sim
        #         # kl_loss = torch.distributions.kl.kl_divergence(d_i, d_j)
        #         # loss_list.append(cos_loss)
        #         loss_list.append(mse.unsqueeze(0))
        # a = torch.cat(loss_list)
        # loss = torch.sum(torch.cat(loss_list)) / self.config.batch_size

        return loss
    
    
    def write2text(self, f, predict, tokenizer, transfer_to):
        # label_list = ['<MT>', '<JK>', '<St>']
        label_list = ['<Sp>', '<St>']

        def convert_tokens_to_string(tokens, tokenizer):
            """Converts a sequence of tokens (string) in a single string."""
            tokens = filter(tokens)
            current_sub_tokens = []
            out_string = ""
            for token in tokens:
                # make sure that special tokens are not decoded using sentencepiece model
                if token in tokenizer.all_special_tokens:
                    out_string += tokenizer.sp_model.decode_pieces(current_sub_tokens)+  " " + token + " "
                    current_sub_tokens = []
                else:
                    current_sub_tokens.append(token)
            out_string += tokenizer.sp_model.decode_pieces(current_sub_tokens)
            return out_string.strip()
        
        def filter(tokens):
            final_output = []
            for token in tokens:
                if token in ["<s>"] or "extra_id" in token:
                    continue
                if token == "</s>":
                    break
                final_output.append(token)
            return final_output
            
        for res, label in zip(predict, transfer_to):
            tokens = tokenizer.convert_ids_to_tokens(res)
            strings = convert_tokens_to_string(tokens, tokenizer)
            # print(strings)
            
            item = {
                "text": strings,
                "style": label_list[label],
            }
            item = json.dumps(item, ensure_ascii=False)
            f.write(item + '\n')


    def tran_transfer_stage_1(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
        # model = T5ForLongText_ST_Sen_Sty_En.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        if self.config.init is not None:
            model.load_state_dict(torch.load(self.config.init), strict=True)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # sty_id = self.get_sty_id(tokenizer)
        style_classifier = self.load_style_classifier(tokenizer)

        # optimizer
        # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # log
        writer = SummaryWriter(self.config.log_dir)
        num_step = self.config.epoch * len(data_generator)
        step = 0
        save_step = num_step // 10
        self.print_tip_for_train(num_step)

        for epoch in range(self.config.epoch):
            for i, (input_text, label_text, style) in enumerate(data_generator):
                style = style.to(self.config.device)
                # label_text_ids = self.get_ids(label_text, tokenizer)
                input_text_ids, sentence_order_label = self.get_ids_and_sentence_order(input_text, tokenizer, sen_id)
                shuffle_input_text_ids, shuffle_sentence_order_label = self.get_shuffle_sen_ids(input_text, tokenizer, sen_id)
                label_text_ids, _ = self.get_ids_and_sentence_order(label_text, tokenizer, sen_id)
                
                # input_text_ids = self.get_ids(input_text, tokenizer)

                # tokens = tokenizer.convert_ids_to_tokens(label_text_ids[0].detach().cpu().numpy())
                # self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                shuffle_outputs = model(input_ids=shuffle_input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                
                shuffle_sen_order_loss = self.get_sen_order_loss(shuffle_outputs.pointing_res, shuffle_sentence_order_label)
                sen_order_loss = self.get_sen_order_loss(self_output.pointing_res, sentence_order_label)
                final_sen_order_loss = (shuffle_sen_order_loss + sen_order_loss) / 2
                
                # sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id)
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids, tokenizer)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_classifier_loss = self.get_style_loss(pred_style, style)

                # cycle content perseveration
                # predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9,
                #                                     temperature=1.0,
                #                                     max_length=self.config.max_length, transfer_to=trans_style,
                #                                     eos_id=tokenizer.eos_token_id, return_logits=True)
                # predict_logits_exp = F.softmax(predict_logits, dim=-1)
                # cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                # cycle_out = model.encoder(inputs_embeds=cycle_input)
                # cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model, affine=False,
                #                                  margin=False)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)

                
                # if not torch.isnan(sen_loss):
                #     loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss + final_sen_order_loss
                # else:
                #     loss = 0.5 * cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                
                # 去掉0.5
                
                if not torch.isnan(sen_loss):
                    loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss + final_sen_order_loss
                else:
                    loss = 0.5 * cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                
                # loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss
                # loss = 0.5 * cross_entro_loss + sen_loss
                # loss = 0.5 * cross_entro_loss + style_classifier_loss
                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss
                # if torch.sum(torch.isnan(loss)) >= 1:
                #     print("here")
                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
                writer.add_scalar('style_classifier_loss', style_classifier_loss, global_step=step)
                writer.add_scalar('sen_order_loss', final_sen_order_loss, global_step=step)
                # writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      # + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
                      # + ' content_distribution ' + str(content_distribution.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      + ' sen_loss ' + str(sen_loss.cpu().detach().numpy())
                      + ' sen_order_loss ' + str(final_sen_order_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()



    def test_transfer_stage_1(self):
        data_generator = self.load_data_mask_in_input(self.config.test_set_mask, self.config.test_batch, shuffle=False, drop_last=False)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
            model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.resize_token_embeddings(len(tokenizer))
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

            if not os.path.exists(self.config.pred_result_dir):
                os.makedirs(self.config.pred_result_dir)

            # transfer_label_list = [0, 1]
            transfer_label_list = [0]
            result = self.config.pred_result_dir + '/' + '{}.'.format(self.config.task_name)
            out_file_list = [result + str(i) for i in transfer_label_list]
            print('begin predicting')

            for rev_label, out_file in zip(transfer_label_list, out_file_list):
                with open(out_file, 'w') as f:
                    for i, (input_text, label_text, style) in enumerate(tqdm(data_generator)):
                        # ids = self.get_ids(batch_text, tokenizer)
                        # style = style.to(self.config.device)
                        # label_text_ids = self.get_ids(label_text, tokenizer)
                        input_text_ids = self.get_ids(input_text, tokenizer)
                        style = torch.ones(input_text_ids.size(0), dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=input_text_ids, decoder_start_token_id=tokenizer.bos_token_id, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    # new ablation
    # 1) token mean
    # 2) style classifier loss
    # 3) sen loss

    def train_sen_mse_add_mask_in_input_ablation_token_mean(self):
        # data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)
        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
        model = T5ForLongText_ST_Sen_Sty_token_mean_En.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        # model = T5ForLongText_ST_Sen_token_mean.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        if self.config.init is not None:
            model.load_state_dict(torch.load(self.config.init), strict=True)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # sty_id = self.get_sty_id(tokenizer)
        style_classifier = self.load_style_classifier(tokenizer)

        # optimizer
        # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # log
        writer = SummaryWriter(self.config.log_dir)
        num_step = self.config.epoch * len(data_generator)
        step = 0
        save_step = num_step // 10
        self.print_tip_for_train(num_step)

        for epoch in range(self.config.epoch):
            # for i, (input_text, label_text, style, index) in enumerate(data_generator):
            for i, (input_text, label_text, style)  in enumerate(data_generator):
                style = style.to(self.config.device)
                label_text_ids = self.get_ids(label_text, tokenizer)
                input_text_ids = self.get_ids(input_text, tokenizer)



                # self construction
                
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids, tokenizer)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_classifier_loss = self.get_style_loss(pred_style, style)

                # cycle content perseveration
                # predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9,
                #                                     temperature=1.0,
                #                                     max_length=self.config.max_length, transfer_to=trans_style,
                #                                     eos_id=tokenizer.eos_token_id, return_logits=True)
                # predict_logits_exp = F.softmax(predict_logits, dim=-1)
                # cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                # cycle_out = model.encoder(inputs_embeds=cycle_input)
                # cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model, affine=False,
                #                                  margin=False)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)

                # loss = cross_entro_loss + sen_loss + style_classifier_loss
                loss = cross_entro_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                # writer.add_scalar('sen_loss', sen_loss, global_step=step)
                writer.add_scalar('style_classifier_loss', style_classifier_loss, global_step=step)
                # writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      # + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
                      # + ' content_distribution ' + str(content_distribution.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      # + ' sen_loss ' + str(sen_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()


    def test_sen_mse_add_mask_in_input_ablation_token_mean(self):
        data_generator = self.load_data_mask_in_input(self.config.test_set_mask, self.config.test_batch, shuffle=False, drop_last=False)
        with torch.no_grad():
            # data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
            model = T5ForLongText_ST_Sen_Sty_token_mean_En.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.resize_token_embeddings(len(tokenizer))
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

            if not os.path.exists(self.config.pred_result_dir):
                os.mkdir(self.config.pred_result_dir)

            # transfer_label_list = [0, 1]
            transfer_label_list = [0]
            result = self.config.pred_result_dir + '/' + '{}.'.format(self.config.task_name)
            out_file_list = [result + str(i) for i in transfer_label_list]
            print('begin predicting')

            for rev_label, out_file in zip(transfer_label_list, out_file_list):
                with open(out_file, 'w') as f:
                    # for i, (input_text, label_text, style, index) in enumerate(tqdm(data_generator)):
                    for i, (input_text, label_text, style) in enumerate(tqdm(data_generator)):
                        input_text_ids = self.get_ids(input_text, tokenizer)
                        style = torch.ones(input_text_ids.size(0), dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=input_text_ids, decoder_start_token_id=tokenizer.bos_token_id, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    def train_sen_mse_add_mask_in_input_ablation_style_classifier(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        if self.config.init is not None:
            model.load_state_dict(torch.load(self.config.init), strict=True)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # sty_id = self.get_sty_id(tokenizer)
        style_classifier = self.load_style_classifier()

        # optimizer
        # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # log
        writer = SummaryWriter(self.config.log_dir)
        num_step = self.config.epoch * len(data_generator)
        step = 0
        save_step = num_step // 10
        self.print_tip_for_train(num_step)

        for epoch in range(self.config.epoch):
            for i, (input_text, label_text, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                label_text_ids = self.get_ids(label_text, tokenizer)
                input_text_ids = self.get_ids(input_text, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                # content_distribution = self.get_content_disturibution(model, self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_single_normal(self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_normal_half_sen(self_output.content_representation) * 0.5
                # style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids)

                # soft_input = F.softmax(self_output.logits, dim=-1)
                # pred_style = style_classifier(soft_input, soft_sampling=True)
                # style_classifier_loss = self.get_style_loss(pred_style, style)

                # cycle content perseveration
                # predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9,
                #                                     temperature=1.0,
                #                                     max_length=self.config.max_length, transfer_to=trans_style,
                #                                     eos_id=tokenizer.eos_token_id, return_logits=True)
                # predict_logits_exp = F.softmax(predict_logits, dim=-1)
                # cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                # cycle_out = model.encoder(inputs_embeds=cycle_input)
                # cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model, affine=False,
                #                                  margin=False)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)

                loss = cross_entro_loss + sen_loss
                # loss = cross_entro_loss + style_classifier_loss
                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
                # writer.add_scalar('style_classifier_loss', style_classifier_loss, global_step=step)
                # writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      # + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      # + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
                      # + ' content_distribution ' + str(content_distribution.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      + ' sen_loss ' + str(sen_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()


    def train_sen_mse_add_mask_in_input_ablation_sen_loss(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        if self.config.init is not None:
            model.load_state_dict(torch.load(self.config.init), strict=True)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # sty_id = self.get_sty_id(tokenizer)
        style_classifier = self.load_style_classifier()

        # optimizer
        # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # log
        writer = SummaryWriter(self.config.log_dir)
        num_step = self.config.epoch * len(data_generator)
        step = 0
        save_step = num_step // 10
        self.print_tip_for_train(num_step)

        for epoch in range(self.config.epoch):
            for i, (input_text, label_text, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                label_text_ids = self.get_ids(label_text, tokenizer)
                input_text_ids = self.get_ids(input_text, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                # content_distribution = self.get_content_disturibution(model, self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_single_normal(self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_normal_half_sen(self_output.content_representation) * 0.5
                # style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                # sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_classifier_loss = self.get_style_loss(pred_style, style)

                # cycle content perseveration
                # predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9,
                #                                     temperature=1.0,
                #                                     max_length=self.config.max_length, transfer_to=trans_style,
                #                                     eos_id=tokenizer.eos_token_id, return_logits=True)
                # predict_logits_exp = F.softmax(predict_logits, dim=-1)
                # cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                # cycle_out = model.encoder(inputs_embeds=cycle_input)
                # cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model, affine=False,
                #                                  margin=False)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)

                # loss = cross_entro_loss + sen_loss + style_classifier_loss
                loss = cross_entro_loss + style_classifier_loss
                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                # writer.add_scalar('sen_loss', sen_loss, global_step=step)
                writer.add_scalar('style_classifier_loss', style_classifier_loss, global_step=step)
                # writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      # + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
                      # + ' content_distribution ' + str(content_distribution.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      # + ' sen_loss ' + str(sen_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()


    # ablation 1129
    
    def ablation_sen_loss(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
        # model = T5ForLongText_ST_Sen_Sty_En.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        if self.config.init is not None:
            model.load_state_dict(torch.load(self.config.init), strict=True)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # sty_id = self.get_sty_id(tokenizer)
        style_classifier = self.load_style_classifier(tokenizer)

        # optimizer
        # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # log
        writer = SummaryWriter(self.config.log_dir)
        num_step = self.config.epoch * len(data_generator)
        step = 0
        save_step = num_step // 10
        self.print_tip_for_train(num_step)

        for epoch in range(self.config.epoch):
            for i, (input_text, label_text, style) in enumerate(data_generator):
                style = style.to(self.config.device)
                # label_text_ids = self.get_ids(label_text, tokenizer)
                input_text_ids, sentence_order_label = self.get_ids_and_sentence_order(input_text, tokenizer, sen_id)
                shuffle_input_text_ids, shuffle_sentence_order_label = self.get_shuffle_sen_ids(input_text, tokenizer, sen_id)
                label_text_ids, _ = self.get_ids_and_sentence_order(label_text, tokenizer, sen_id)
                
                # input_text_ids = self.get_ids(input_text, tokenizer)

                # tokens = tokenizer.convert_ids_to_tokens(label_text_ids[0].detach().cpu().numpy())
                # self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                shuffle_outputs = model(input_ids=shuffle_input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                
                shuffle_sen_order_loss = self.get_sen_order_loss(shuffle_outputs.pointing_res, shuffle_sentence_order_label)
                sen_order_loss = self.get_sen_order_loss(self_output.pointing_res, sentence_order_label)
                final_sen_order_loss = (shuffle_sen_order_loss + sen_order_loss) / 2
                
                # sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id)
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids, tokenizer)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_classifier_loss = self.get_style_loss(pred_style, style)

                # cycle content perseveration
                # predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9,
                #                                     temperature=1.0,
                #                                     max_length=self.config.max_length, transfer_to=trans_style,
                #                                     eos_id=tokenizer.eos_token_id, return_logits=True)
                # predict_logits_exp = F.softmax(predict_logits, dim=-1)
                # cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                # cycle_out = model.encoder(inputs_embeds=cycle_input)
                # cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model, affine=False,
                #                                  margin=False)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)

                
                # if not torch.isnan(sen_loss):
                #     loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss + final_sen_order_loss
                # else:
                #     loss = 0.5 * cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                
                loss = 0.5 * cross_entro_loss + style_classifier_loss + final_sen_order_loss

                
                
                # loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss
                # loss = 0.5 * cross_entro_loss + sen_loss
                # loss = 0.5 * cross_entro_loss + style_classifier_loss
                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss
                # if torch.sum(torch.isnan(loss)) >= 1:
                #     print("here")
                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
                writer.add_scalar('style_classifier_loss', style_classifier_loss, global_step=step)
                writer.add_scalar('sen_order_loss', final_sen_order_loss, global_step=step)
                # writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      # + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
                      # + ' content_distribution ' + str(content_distribution.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      + ' sen_loss ' + str(sen_loss.cpu().detach().numpy())
                      + ' sen_order_loss ' + str(final_sen_order_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()
    
    
    
    def ablation_style_classifier_loss(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
        # model = T5ForLongText_ST_Sen_Sty_En.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        if self.config.init is not None:
            model.load_state_dict(torch.load(self.config.init), strict=True)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # sty_id = self.get_sty_id(tokenizer)
        style_classifier = self.load_style_classifier(tokenizer)

        # optimizer
        # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # log
        writer = SummaryWriter(self.config.log_dir)
        num_step = self.config.epoch * len(data_generator)
        step = 0
        save_step = num_step // 10
        self.print_tip_for_train(num_step)

        for epoch in range(self.config.epoch):
            for i, (input_text, label_text, style) in enumerate(data_generator):
                style = style.to(self.config.device)
                # label_text_ids = self.get_ids(label_text, tokenizer)
                input_text_ids, sentence_order_label = self.get_ids_and_sentence_order(input_text, tokenizer, sen_id)
                shuffle_input_text_ids, shuffle_sentence_order_label = self.get_shuffle_sen_ids(input_text, tokenizer, sen_id)
                label_text_ids, _ = self.get_ids_and_sentence_order(label_text, tokenizer, sen_id)
                
                # input_text_ids = self.get_ids(input_text, tokenizer)

                # tokens = tokenizer.convert_ids_to_tokens(label_text_ids[0].detach().cpu().numpy())
                # self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                shuffle_outputs = model(input_ids=shuffle_input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                
                shuffle_sen_order_loss = self.get_sen_order_loss(shuffle_outputs.pointing_res, shuffle_sentence_order_label)
                sen_order_loss = self.get_sen_order_loss(self_output.pointing_res, sentence_order_label)
                final_sen_order_loss = (shuffle_sen_order_loss + sen_order_loss) / 2
                
                # sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id)
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids, tokenizer)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_classifier_loss = self.get_style_loss(pred_style, style)

                # cycle content perseveration
                # predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9,
                #                                     temperature=1.0,
                #                                     max_length=self.config.max_length, transfer_to=trans_style,
                #                                     eos_id=tokenizer.eos_token_id, return_logits=True)
                # predict_logits_exp = F.softmax(predict_logits, dim=-1)
                # cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                # cycle_out = model.encoder(inputs_embeds=cycle_input)
                # cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model, affine=False,
                #                                  margin=False)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)

                
                # if not torch.isnan(sen_loss):
                #     loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss + final_sen_order_loss
                # else:
                #     loss = 0.5 * cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                
                # 去掉0.5
                
                if not torch.isnan(sen_loss):
                    loss = 0.5 * cross_entro_loss + sen_loss + final_sen_order_loss
                else:
                    loss = 0.5 * cross_entro_loss + final_sen_order_loss
                
                
                # loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss
                # loss = 0.5 * cross_entro_loss + sen_loss
                # loss = 0.5 * cross_entro_loss + style_classifier_loss
                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss
                # if torch.sum(torch.isnan(loss)) >= 1:
                #     print("here")
                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
                writer.add_scalar('style_classifier_loss', style_classifier_loss, global_step=step)
                writer.add_scalar('sen_order_loss', final_sen_order_loss, global_step=step)
                # writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      # + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
                      # + ' content_distribution ' + str(content_distribution.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      + ' sen_loss ' + str(sen_loss.cpu().detach().numpy())
                      + ' sen_order_loss ' + str(final_sen_order_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()
    
    
    
    def ablation_sen_order_loss(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
        # model = T5ForLongText_ST_Sen_Sty_En.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        if self.config.init is not None:
            model.load_state_dict(torch.load(self.config.init), strict=True)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # sty_id = self.get_sty_id(tokenizer)
        style_classifier = self.load_style_classifier(tokenizer)

        # optimizer
        # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # log
        writer = SummaryWriter(self.config.log_dir)
        num_step = self.config.epoch * len(data_generator)
        step = 0
        save_step = num_step // 10
        self.print_tip_for_train(num_step)

        for epoch in range(self.config.epoch):
            for i, (input_text, label_text, style) in enumerate(data_generator):
                style = style.to(self.config.device)
                # label_text_ids = self.get_ids(label_text, tokenizer)
                input_text_ids, sentence_order_label = self.get_ids_and_sentence_order(input_text, tokenizer, sen_id)
                shuffle_input_text_ids, shuffle_sentence_order_label = self.get_shuffle_sen_ids(input_text, tokenizer, sen_id)
                label_text_ids, _ = self.get_ids_and_sentence_order(label_text, tokenizer, sen_id)
                
                # input_text_ids = self.get_ids(input_text, tokenizer)

                # tokens = tokenizer.convert_ids_to_tokens(label_text_ids[0].detach().cpu().numpy())
                # self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                shuffle_outputs = model(input_ids=shuffle_input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                
                shuffle_sen_order_loss = self.get_sen_order_loss(shuffle_outputs.pointing_res, shuffle_sentence_order_label)
                sen_order_loss = self.get_sen_order_loss(self_output.pointing_res, sentence_order_label)
                final_sen_order_loss = (shuffle_sen_order_loss + sen_order_loss) / 2
                
                # sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id)
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids, tokenizer)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_classifier_loss = self.get_style_loss(pred_style, style)

                # cycle content perseveration
                # predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9,
                #                                     temperature=1.0,
                #                                     max_length=self.config.max_length, transfer_to=trans_style,
                #                                     eos_id=tokenizer.eos_token_id, return_logits=True)
                # predict_logits_exp = F.softmax(predict_logits, dim=-1)
                # cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                # cycle_out = model.encoder(inputs_embeds=cycle_input)
                # cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model, affine=False,
                #                                  margin=False)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)

                
                # if not torch.isnan(sen_loss):
                #     loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss + final_sen_order_loss
                # else:
                #     loss = 0.5 * cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                
                if not torch.isnan(sen_loss):
                    loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss 
                else:
                    loss = 0.5 * cross_entro_loss + style_classifier_loss 
                
                
                # loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss
                # loss = 0.5 * cross_entro_loss + sen_loss
                # loss = 0.5 * cross_entro_loss + style_classifier_loss
                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss
                # if torch.sum(torch.isnan(loss)) >= 1:
                #     print("here")
                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
                writer.add_scalar('style_classifier_loss', style_classifier_loss, global_step=step)
                writer.add_scalar('sen_order_loss', final_sen_order_loss, global_step=step)
                # writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      # + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
                      # + ' content_distribution ' + str(content_distribution.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      + ' sen_loss ' + str(sen_loss.cpu().detach().numpy())
                      + ' sen_order_loss ' + str(final_sen_order_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()
    
    
    # one stage transfer
    def tran_transfer_use1_stage(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size, mask=False)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
        # model = T5ForLongText_ST_Sen_Sty_En.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        if self.config.init is not None:
            model.load_state_dict(torch.load(self.config.init), strict=True)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # sty_id = self.get_sty_id(tokenizer)
        style_classifier = self.load_style_classifier(tokenizer)

        # optimizer
        # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # log
        writer = SummaryWriter(self.config.log_dir)
        num_step = self.config.epoch * len(data_generator)
        step = 0
        save_step = num_step // 10
        self.print_tip_for_train(num_step)

        for epoch in range(self.config.epoch):
            for i, (input_text, label_text, style) in enumerate(data_generator):
                style = style.to(self.config.device)
                # label_text_ids = self.get_ids(label_text, tokenizer)
                input_text_ids, sentence_order_label = self.get_ids_and_sentence_order(input_text, tokenizer, sen_id)
                shuffle_input_text_ids, shuffle_sentence_order_label = self.get_shuffle_sen_ids(input_text, tokenizer, sen_id)
                label_text_ids, _ = self.get_ids_and_sentence_order(label_text, tokenizer, sen_id)
                
                # input_text_ids = self.get_ids(input_text, tokenizer)

                # tokens = tokenizer.convert_ids_to_tokens(label_text_ids[0].detach().cpu().numpy())
                # self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                shuffle_outputs = model(input_ids=shuffle_input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                
                shuffle_sen_order_loss = self.get_sen_order_loss(shuffle_outputs.pointing_res, shuffle_sentence_order_label)
                sen_order_loss = self.get_sen_order_loss(self_output.pointing_res, sentence_order_label)
                final_sen_order_loss = (shuffle_sen_order_loss + sen_order_loss) / 2
                
                # sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id)
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids, tokenizer)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_classifier_loss = self.get_style_loss(pred_style, style)

                # cycle content perseveration
                # predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9,
                #                                     temperature=1.0,
                #                                     max_length=self.config.max_length, transfer_to=trans_style,
                #                                     eos_id=tokenizer.eos_token_id, return_logits=True)
                # predict_logits_exp = F.softmax(predict_logits, dim=-1)
                # cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                # cycle_out = model.encoder(inputs_embeds=cycle_input)
                # cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model, affine=False,
                #                                  margin=False)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)

                
                # if not torch.isnan(sen_loss):
                #     loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss + final_sen_order_loss
                # else:
                #     loss = 0.5 * cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                
                # 去掉0.5
                
                if not torch.isnan(sen_loss):
                    loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss + final_sen_order_loss
                else:
                    loss = 0.5 * cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                
                # loss = 0.5 * cross_entro_loss + sen_loss + style_classifier_loss
                # loss = 0.5 * cross_entro_loss + sen_loss
                # loss = 0.5 * cross_entro_loss + style_classifier_loss
                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss
                # if torch.sum(torch.isnan(loss)) >= 1:
                #     print("here")
                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
                writer.add_scalar('style_classifier_loss', style_classifier_loss, global_step=step)
                writer.add_scalar('sen_order_loss', final_sen_order_loss, global_step=step)
                # writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      # + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
                      # + ' content_distribution ' + str(content_distribution.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      + ' sen_loss ' + str(sen_loss.cpu().detach().numpy())
                      + ' sen_order_loss ' + str(final_sen_order_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()


    def test_transfer_use1_stage(self):
        data_generator = self.load_data_mask_in_input(self.config.test_set_mask, self.config.test_batch, mask=False, shuffle=False, drop_last=False)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
            model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.resize_token_embeddings(len(tokenizer))
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

            if not os.path.exists(self.config.pred_result_dir):
                os.makedirs(self.config.pred_result_dir)

            # transfer_label_list = [0, 1]
            transfer_label_list = [0]
            result = self.config.pred_result_dir + '/' + '{}.'.format(self.config.task_name)
            out_file_list = [result + str(i) for i in transfer_label_list]
            print('begin predicting')

            for rev_label, out_file in zip(transfer_label_list, out_file_list):
                with open(out_file, 'w') as f:
                    for i, (input_text, label_text, style) in enumerate(tqdm(data_generator)):
                        # ids = self.get_ids(batch_text, tokenizer)
                        # style = style.to(self.config.device)
                        # label_text_ids = self.get_ids(label_text, tokenizer)
                        input_text_ids = self.get_ids(input_text, tokenizer)
                        style = torch.ones(input_text_ids.size(0), dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=input_text_ids, decoder_start_token_id=tokenizer.bos_token_id, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)




class Fill_Mask_Model(nn.Module):
    def __init__(self, config):
        super(Fill_Mask_Model, self).__init__()
        self.config = config

    def load_data_mask(self, dataset, batch_size, shuffle=True, drop=None):
        dataset = Data_Encoder_Fill_En(dataset, drop_ratio=drop)
        data_generator = DataLoader(dataset, batch_size, shuffle=shuffle)
        return data_generator

    def load_data_and_res(self, result, dataset, batch_size, shuffle=True):
        dataset = Data_Encoder_Fill_Res_En(result, dataset)
        data_generator = DataLoader(dataset, batch_size, shuffle=shuffle)
        return data_generator

    def get_ids(self, batch_text, tokenizer, attention_mask=False):
        ids = tokenizer(batch_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length)


        if attention_mask:
            return ids.input_ids.to(self.config.device), ids.attention_mask.to(self.config.device)
        else:
            return ids.input_ids.to(self.config.device)

    def get_cross_entropy_loss(self, logits, label, margin=False):
        # loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=config.pad_token_id)
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        # loss = loss_fct(logits.permute(0, 2, 1), label) / config.batch_size
        loss = loss_fct(logits.permute(0, 2, 1), label)
        # if margin:
        #     # loss = self.config.margin
        #     loss = torch.max(loss, torch.tensor(self.config.margin).to(loss.device))
        return loss


    def print_tip_for_train(self, num_step):
        print('------------------create model---------------------------')
        print('epoch num : {}'.format(self.config.mask_fill_epoch))
        print('step num : {}'.format(num_step))
        print('batch size : {}'.format(self.config.batch_size))
        print('learning rate : {}'.format(self.config.mask_fill_learning_rate))
        print('begin training')

    def write2text(self, f, predict, tokenizer, label):
        # label_list = ['<LX>', '<JY>', '<GS>']
        def convert_tokens_to_string(tokens, tokenizer):
            """Converts a sequence of tokens (string) in a single string."""
            tokens = filter(tokens)
            current_sub_tokens = []
            out_string = ""
            for token in tokens:
                # make sure that special tokens are not decoded using sentencepiece model
                if token in tokenizer.all_special_tokens:
                    out_string += tokenizer.sp_model.decode_pieces(current_sub_tokens)+  " " + token + " "
                    current_sub_tokens = []
                else:
                    current_sub_tokens.append(token)
            out_string += tokenizer.sp_model.decode_pieces(current_sub_tokens)
            return out_string.strip()
        
        def filter(tokens):
            final_output = []
            for token in tokens:
                if token in ["<s>"] or "extra_id" in token:
                    continue
                if token == "</s>":
                    break
                final_output.append(token)
            return final_output
            
        
        for res, sty in zip(predict, label):
            tokens = tokenizer.convert_ids_to_tokens(res)
            strings = convert_tokens_to_string(tokens, tokenizer)

            # final_output = []
            item = {
                "text": strings,
                "style": sty,
            }
            item = json.dumps(item, ensure_ascii=False)
            f.write(item + '\n')

    def write2text_simple(self, f, predict, tokenizer):
        # label_list = ['<LX>', '<JY>', '<GS>']

        def convert_tokens_to_string(tokens, tokenizer):
            """Converts a sequence of tokens (string) in a single string."""
            tokens = filter(tokens)
            current_sub_tokens = []
            out_string = ""
            for token in tokens:
                # make sure that special tokens are not decoded using sentencepiece model
                if token in tokenizer.all_special_tokens:
                    out_string += tokenizer.sp_model.decode_pieces(current_sub_tokens)+  " " + token + " "
                    current_sub_tokens = []
                else:
                    current_sub_tokens.append(token)
            out_string += tokenizer.sp_model.decode_pieces(current_sub_tokens)
            return out_string.strip()
        
        def filter(tokens):
            final_output = []
            for token in tokens:
                if token in ["<s>"] or "extra_id" in token:
                    continue
                if token == "</s>":
                    break
                final_output.append(token)
            return final_output
        
        for res in predict:
            tokens = tokenizer.convert_ids_to_tokens(res)
            strings = convert_tokens_to_string(tokens, tokenizer)
            # for token in tokens:
                # if token in ["<s>", "▁"] or "extra_id" in token:
                #     continue
                # if "▁" in token:
                #     token = token.replace("▁", "")
                # if token == "</s>":
                #     break
                # final_output.append(token)

            item = {
                "text": strings,
                # "style": sty,
            }
            item = json.dumps(item, ensure_ascii=False)
            f.write(item + '\n')


    def test_base(self):
        data_generator = self.load_data_mask(self.config.test_set_mask, self.config.test_batch, shuffle=False)
        # t5_config = AutoConfig.from_pretrained('pretrained_model/t5_small')
        # model = T5ForConditionalGeneration(t5_config).to(self.config.device)
        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token_fill})
        model = T5ForConditionalGeneration.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.load_state_dict(torch.load(self.config.init), strict=True)
        model.eval()

        if not os.path.exists(self.config.pred_result_dir):
            os.mkdir(self.config.pred_result_dir)

        # transfer_label_list = [0, 1]
        result = self.config.pred_result_dir + '/' + '{}'.format(self.config.task_name)
        # out_file_list = [result + str(i) for i in transfer_label_list]
        # out_file_list = [result + str(i) for i in transfer_label_list]
        print('begin predicting')

        with torch.no_grad():
            with open(result, 'w') as f:
                for i, (text, input_text) in enumerate(tqdm(data_generator)):
                    ids = self.get_ids(input_text, tokenizer)
                    # label = self.get_ids(text, tokenizer)

                    pre = model.generate(ids, do_sample=True, decoder_start_id=model.config.decoder_start_token_id, top_p=0.9, max_length=512)
                    pre = pre.cpu().numpy().tolist()
                    # transfer_to = style.cpu().numpy().tolist()
                    self.write2text_simple(f, pre, tokenizer)





    def train_fill_LongLM(self):
        # data_generator = self.load_data_mask(self.config.train_set_mask, self.config.batch_size, drop=0.2)
        data_generator = self.load_data_mask(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token_fill})
        model = T5ForConditionalGeneration.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id

        # model = T5ForConditionalGeneration.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        # tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        # model = T5ForLongText_ST.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        # print(model.shared.weight)
        # sen_id = self.get_sen_id(tokenizer)
        # sty_id = self.get_sty_id(tokenizer)
        # style_classifier = self.load_style_classifier()

        # optimizer
        # opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.config.learning_rate)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.mask_fill_learning_rate)

        # log
        writer = SummaryWriter(self.config.log_dir)
        num_step = self.config.mask_fill_epoch * len(data_generator)
        step = 0
        save_step = num_step // 10
        self.print_tip_for_train(num_step)

        for epoch in range(self.config.mask_fill_epoch):
            for i, (text, input_text) in enumerate(data_generator):
                # style = style.to(self.config.device)
                # unk = tokenizer("<unk>").input_ids
                # unk_str = tokenizer.convert_ids_to_tokens(unk)
                # mask = tokenizer("<mask>").input_ids
                # mask_str = tokenizer.convert_ids_to_tokens(mask)
                ids = self.get_ids(input_text, tokenizer)
                label = self.get_ids(text, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                self_output = model(input_ids=ids, labels=label)
                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                # content_distribution = self.get_content_disturibution(model, self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_single_normal(self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_normal_half_sen(self_output.content_representation) * 0.5
                # style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label)

                # soft_input = F.softmax(self_output.logits, dim=-1)
                # pred_style = style_classifier(soft_input, soft_sampling=True)
                # style_classifier_loss = self.get_style_loss(pred_style, style)

                # cycle content perseveration
                # predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9,
                #                                     temperature=1.0,
                #                                     max_length=self.config.max_length, transfer_to=trans_style,
                #                                     eos_id=tokenizer.eos_token_id, return_logits=True)
                # predict_logits_exp = F.softmax(predict_logits, dim=-1)
                # cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                # cycle_out = model.encoder(inputs_embeds=cycle_input)
                # cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model, affine=False,
                #                                  margin=False)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)

                loss = cross_entro_loss #+ style_contrastive_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                # writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                # writer.add_scalar('sen_loss', sen_loss, global_step=step)
                # writer.add_scalar('style_classifier_loss', style_classifier_loss, global_step=step)
                # writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      # + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      # + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      # + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
                      # + ' content_distribution ' + str(content_distribution.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      # + ' sen_loss ' + str(sen_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()



    def fill_mask(self):
        data_generator = self.load_data_and_res(self.config.need_fill, self.config.test_set_mask, self.config.test_batch, shuffle=False)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token_fill})
        model = T5ForConditionalGeneration.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.load_state_dict(torch.load(self.config.init), strict=True)


        # model = T5ForConditionalGeneration.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        # tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        # model.load_state_dict(torch.load(self.config.init), strict=True)
        model.eval()

        if not os.path.exists(self.config.pred_result_dir):
            os.mkdir(self.config.pred_result_dir)

        # transfer_label_list = [0, 1]
        file_name = self.config.need_fill.split("/")[-1]
        result = self.config.pred_result_dir + '/' + '{}'.format(file_name)
        # out_file_list = [result + str(i) for i in transfer_label_list]
        # out_file_list = [result + str(i) for i in transfer_label_list]
        print('begin predicting')

        with torch.no_grad():
            with open(result, 'w') as f:
                for i, (input_text, style) in enumerate(tqdm(data_generator)):
                    ids = self.get_ids(input_text, tokenizer)
                    # label = self.get_ids(text, tokenizer)

                    pre = model.generate(ids, do_sample=True, decoder_start_id=1, top_p=0.9, max_length=512)
                    pre = pre.cpu().numpy().tolist()
                    # transfer_to = style.cpu().numpy().tolist()
                    self.write2text(f, pre, tokenizer, style)
