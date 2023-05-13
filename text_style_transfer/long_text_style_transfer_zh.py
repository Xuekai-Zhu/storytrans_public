import json, pickle
from random import shuffle
from data_set import Data_Encoder, Data_Encoder_Sen, Data_Encoder_Mask, Data_Encoder_Fill, Data_Encoder_Fill_Res, Data_Encoder_Mask_Input, Data_Encoder_Mask_Input_without_mask
from T5Tokenizer import T5Tokenizer
from modeling_t5 import T5ForConditionalGeneration, Style_Classifier, LongTextST, LongTextST_Disentangled, LongTextST_Concat, LongTextST_Concat_Inter, LongTextST_Disturb, LongTextST_Attention, LongTextST_Test, LongTextST_Style, LongTextST_Style_Attention, LongTextST_Style_Change, LongTextST_Content_Dis, LongTextST_Content_Dis_And_Attention, T5ForLongText_ST, T5ForLongText_ST_without_sty, T5Model, T5ForLongText_ST_Sen_Sty, T5ForLongText_ST_Sen_Sty_Ablation_Sen, T5ForLongText_ST_Sen_Sty_ProSen, T5ForLongText_ST_Sen_Sty_ProSen_Att
from modeling_t5 import T5ForLongText_ST_Sen_Sty_ProSen_Att_Fix_Len, T5ForLongText_ST_Sen_token_mean, T5ForLongText_ST_Sen_Sty_get_all_token
from transformers import AutoModel, AutoConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import os, torch, time
import torch.nn.functional as F
import numpy as np
from pytorch_metric_learning.losses import NTXentLoss
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.utils.rnn import pad_sequence

class LongTextStyleTrans(nn.Module):
    def __init__(self, config):
        super(LongTextStyleTrans, self).__init__()
        self.config = config

    def load_data_mask_in_input(self, dataset, batch_size, shuffle=True):
        dataset = Data_Encoder_Mask_Input(dataset)
        data_generator = DataLoader(dataset, batch_size, shuffle=shuffle)
        return data_generator

    def load_no_mask_data(self, dataset, batch_size, shuffle=True):
        dataset = Data_Encoder_Mask_Input_without_mask(dataset)
        data_generator = DataLoader(dataset, batch_size, shuffle=shuffle)
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

    def load_style_classifier(self):
        style_classifer = Style_Classifier(self.config).to(self.config.device)
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

    def get_ids(self, batch_text, tokenizer, sen_id, attention_mask=False, sentence_type=False):
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
        
        
        

    def get_cross_entropy_loss(self, logits, label, margin=False):
        # loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=config.pad_token_id)
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        # loss = loss_fct(logits.permute(0, 2, 1), label) / config.batch_size
        loss = loss_fct(logits.permute(0, 2, 1), label)
        # if margin:
        #     # loss = self.config.margin
        #     loss = torch.max(loss, torch.tensor(self.config.margin).to(loss.device))
        return loss

    
    def get_sen_order_loss(self, predict_logits, labels):
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        real_label = labels.view(-1) - 1
        loss = loss_fct(predict_logits.view(-1, predict_logits.size(-1)), real_label)
        return loss

    def get_style_loss(self, pred, label):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pred, label)
        return loss



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
        label_list = ['<LX>', '<JY>', '<GS>']

        for res, label in zip(predict, transfer_to):
            tokens = tokenizer.convert_ids_to_tokens(res)
            final_output = []
            for token in tokens:
                if token in ["<s>", "▁"] or "extra_id" in token:
                    continue
                if "▁" in token:
                    token = token.replace("▁", "")
                if token == "</s>":
                    break
                final_output.append(token)

            item = {
                "text": "".join(final_output),
                "style": label_list[label],
            }
            item = json.dumps(item, ensure_ascii=False)
            f.write(item + '\n')

    def test(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)

        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = LongTextST.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()

            if not os.path.exists(self.config.pred_result_dir):
                os.mkdir(self.config.pred_result_dir)

            transfer_label_list = [0, 1]
            result = self.config.pred_result_dir + '/' + '{}.'.format(self.config.task_name)
            out_file_list = [result + str(i) for i in transfer_label_list]
            print('begin predicting')

            for rev_label, out_file in zip(transfer_label_list, out_file_list):
                with open(out_file, 'w') as f:
                    for i, (batch_text, batch_text_add_sen, style, index) in enumerate(tqdm(data_generator)):
                        # ids = self.get_ids(batch_text, tokenizer)
                        ids = self.get_ids(batch_text_add_sen, tokenizer)
                        style = torch.ones(self.config.test_batch, dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0, max_length=self.config.max_length, transfer_to=style, eos_id=tokenizer.eos_token_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    # ablation
    # 1) cross_entropy
    # 2) contrastive
    # 3) sen
    # 4) style classifier
    # 5) sen(all mean)
    #
    def train_sen_mse_add_mask_in_input_ablation_2(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        # model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
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
                input_text_ids, sentence_order_label = self.get_ids(input_text, tokenizer, sen_id)
                label_text_ids, _ = self.get_ids(label_text, tokenizer, sen_id)
                shuffle_input_text_ids, shuffle_sentence_order_label = self.get_shuffle_sen_ids(input_text, tokenizer, sen_id)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                # self_output = model(input_ids=input_text_ids, input_sentence_types=input_sentence_types, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                # shuffle_outputs = model(input_ids=shuffle_input_text_ids, input_sentence_types=shuffle_input_sentence_types, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                # 去掉 sentence type id
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                shuffle_outputs = model(input_ids=shuffle_input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                # sentence order loss
                shuffle_sen_order_loss = self.get_sen_order_loss(shuffle_outputs.pointing_res, shuffle_sentence_order_label)
                sen_order_loss = self.get_sen_order_loss(self_output.pointing_res, sentence_order_label)
                final_sen_order_loss = (shuffle_sen_order_loss + sen_order_loss) / 2
                
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id, style)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_classifier_loss = self.get_style_loss(pred_style, style) 
                
                if not torch.isnan(sen_loss):
                    loss = cross_entro_loss + sen_loss + style_classifier_loss + final_sen_order_loss
                else:
                    loss = cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                # # ablation sen loss
                # loss = cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                # # ablation style_classifier_loss
                # if not torch.isnan(sen_loss):
                #     loss = cross_entro_loss + sen_loss + final_sen_order_loss
                # else:
                #     loss = cross_entro_loss + final_sen_order_loss

                # # ablation final_sen_order_loss
                # if not torch.isnan(sen_loss):
                #     loss = cross_entro_loss + sen_loss + style_classifier_loss
                # else:
                #     loss = cross_entro_loss + style_classifier_loss
                
                
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
                        os.makedirs(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()


    def test_sen_mse_add_mask_in_input(self):
        data_generator = self.load_data_mask_in_input(self.config.test_set_mask, self.config.test_batch, shuffle=False)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
            model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            # model.resize_token_embeddings(len(tokenizer))
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

            if not os.path.exists(self.config.pred_result_dir):
                os.makedirs(self.config.pred_result_dir)

            transfer_label_list = [0, 1]
            result = self.config.pred_result_dir + '/' + '{}.'.format(self.config.task_name)
            out_file_list = [result + str(i) for i in transfer_label_list]
            print('begin predicting')

            for rev_label, out_file in zip(transfer_label_list, out_file_list):
                with open(out_file, 'w') as f:
                    for i, (input_text, label_text, style, index) in enumerate(tqdm(data_generator)):
                        # ids = self.get_ids(batch_text, tokenizer)
                        # style = style.to(self.config.device)
                        # label_text_ids = self.get_ids(label_text, tokenizer)
                        input_text_ids, sentence_order_label = self.get_ids(input_text, tokenizer, sen_id)

                        # input_text_ids = self.get_ids(input_text, tokenizer)
                        style = torch.ones(self.config.test_batch, dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=input_text_ids,
                                              decoder_start_token_id=tokenizer.bos_token_id, 
                                              top_p=0.9, 
                                              temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)

    
    # one_stage_train
    def train_1119(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_no_mask_data(self.config.train_set, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        # model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
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
                input_text_ids, sentence_order_label = self.get_ids(input_text, tokenizer, sen_id)
                label_text_ids, _ = self.get_ids(label_text, tokenizer, sen_id)
                shuffle_input_text_ids, shuffle_sentence_order_label = self.get_shuffle_sen_ids(input_text, tokenizer, sen_id)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                # self_output = model(input_ids=input_text_ids, input_sentence_types=input_sentence_types, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                # shuffle_outputs = model(input_ids=shuffle_input_text_ids, input_sentence_types=shuffle_input_sentence_types, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                # 去掉 sentence type id
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                shuffle_outputs = model(input_ids=shuffle_input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                # sentence order loss
                shuffle_sen_order_loss = self.get_sen_order_loss(shuffle_outputs.pointing_res, shuffle_sentence_order_label)
                sen_order_loss = self.get_sen_order_loss(self_output.pointing_res, sentence_order_label)
                final_sen_order_loss = (shuffle_sen_order_loss + sen_order_loss) / 2
                
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id, style)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_classifier_loss = self.get_style_loss(pred_style, style) 
                
                if not torch.isnan(sen_loss):
                    loss = cross_entro_loss + sen_loss + style_classifier_loss + final_sen_order_loss
                else:
                    loss = cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                # # ablation sen loss
                # loss = cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                # # ablation style_classifier_loss
                # if not torch.isnan(sen_loss):
                #     loss = cross_entro_loss + sen_loss + final_sen_order_loss
                # else:
                #     loss = cross_entro_loss + final_sen_order_loss

                # # ablation final_sen_order_loss
                # if not torch.isnan(sen_loss):
                #     loss = cross_entro_loss + sen_loss + style_classifier_loss
                # else:
                #     loss = cross_entro_loss + style_classifier_loss
                
                
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
                        os.makedirs(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()
    
    # one stage generation 
    def test_1119(self):
        data_generator = self.load_no_mask_data(self.config.test_set, self.config.test_batch, shuffle=False)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
            model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            # model.resize_token_embeddings(len(tokenizer))
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

            if not os.path.exists(self.config.pred_result_dir):
                os.makedirs(self.config.pred_result_dir)

            transfer_label_list = [0, 1]
            result = self.config.pred_result_dir + '/' + '{}.'.format(self.config.task_name)
            out_file_list = [result + str(i) for i in transfer_label_list]
            print('begin predicting')

            for rev_label, out_file in zip(transfer_label_list, out_file_list):
                with open(out_file, 'w') as f:
                    for i, (input_text, label_text, style, index) in enumerate(tqdm(data_generator)):
                        # ids = self.get_ids(batch_text, tokenizer)
                        # style = style.to(self.config.device)
                        # label_text_ids = self.get_ids(label_text, tokenizer)
                        input_text_ids, sentence_order_label = self.get_ids(input_text, tokenizer, sen_id)

                        # input_text_ids = self.get_ids(input_text, tokenizer)
                        style = torch.ones(self.config.test_batch, dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=input_text_ids,
                                              decoder_start_token_id=tokenizer.bos_token_id, 
                                              top_p=0.9, 
                                              temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)

    # ablation sen loss 11.27
    def ablation_sen_loss(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        # model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
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
                input_text_ids, sentence_order_label = self.get_ids(input_text, tokenizer, sen_id)
                label_text_ids, _ = self.get_ids(label_text, tokenizer, sen_id)
                shuffle_input_text_ids, shuffle_sentence_order_label = self.get_shuffle_sen_ids(input_text, tokenizer, sen_id)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                # self_output = model(input_ids=input_text_ids, input_sentence_types=input_sentence_types, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                # shuffle_outputs = model(input_ids=shuffle_input_text_ids, input_sentence_types=shuffle_input_sentence_types, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                # 去掉 sentence type id
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                shuffle_outputs = model(input_ids=shuffle_input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                # sentence order loss
                shuffle_sen_order_loss = self.get_sen_order_loss(shuffle_outputs.pointing_res, shuffle_sentence_order_label)
                sen_order_loss = self.get_sen_order_loss(self_output.pointing_res, sentence_order_label)
                final_sen_order_loss = (shuffle_sen_order_loss + sen_order_loss) / 2
                
                # sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id, style)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_classifier_loss = self.get_style_loss(pred_style, style) 
                
                # if not torch.isnan(sen_loss):
                #     loss = cross_entro_loss + sen_loss + style_classifier_loss + final_sen_order_loss
                # else:
                loss = cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                # # ablation sen loss
                # loss = cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                # # ablation style_classifier_loss
                # if not torch.isnan(sen_loss):
                #     loss = cross_entro_loss + sen_loss + final_sen_order_loss
                # else:
                #     loss = cross_entro_loss + final_sen_order_loss

                # # ablation final_sen_order_loss
                # if not torch.isnan(sen_loss):
                #     loss = cross_entro_loss + sen_loss + style_classifier_loss
                # else:
                #     loss = cross_entro_loss + style_classifier_loss
                
                
                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                # writer.add_scalar('sen_loss', sen_loss, global_step=step)
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
                    #   + ' sen_loss ' + str(sen_loss.cpu().detach().numpy())
                      + ' sen_order_loss ' + str(final_sen_order_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.makedirs(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()
    
    # ablation style classifier loss
    def ablation_style_classifier_loss(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        # model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
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
                input_text_ids, sentence_order_label = self.get_ids(input_text, tokenizer, sen_id)
                label_text_ids, _ = self.get_ids(label_text, tokenizer, sen_id)
                shuffle_input_text_ids, shuffle_sentence_order_label = self.get_shuffle_sen_ids(input_text, tokenizer, sen_id)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                # self_output = model(input_ids=input_text_ids, input_sentence_types=input_sentence_types, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                # shuffle_outputs = model(input_ids=shuffle_input_text_ids, input_sentence_types=shuffle_input_sentence_types, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                # 去掉 sentence type id
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                shuffle_outputs = model(input_ids=shuffle_input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                # sentence order loss
                shuffle_sen_order_loss = self.get_sen_order_loss(shuffle_outputs.pointing_res, shuffle_sentence_order_label)
                sen_order_loss = self.get_sen_order_loss(self_output.pointing_res, sentence_order_label)
                final_sen_order_loss = (shuffle_sen_order_loss + sen_order_loss) / 2
                
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id, style)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_classifier_loss = self.get_style_loss(pred_style, style) 
                
                if not torch.isnan(sen_loss):
                    loss = cross_entro_loss + sen_loss + final_sen_order_loss
                else:
                    loss = cross_entro_loss + final_sen_order_loss
                
                # # ablation sen loss
                # loss = cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                # # ablation style_classifier_loss
                # if not torch.isnan(sen_loss):
                #     loss = cross_entro_loss + sen_loss + final_sen_order_loss
                # else:
                #     loss = cross_entro_loss + final_sen_order_loss

                # # ablation final_sen_order_loss
                # if not torch.isnan(sen_loss):
                #     loss = cross_entro_loss + sen_loss + style_classifier_loss
                # else:
                #     loss = cross_entro_loss + style_classifier_loss
                
                
                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
                # writer.add_scalar('style_classifier_loss', style_classifier_loss, global_step=step)
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
                    #   + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
                      # + ' content_distribution ' + str(content_distribution.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      + ' sen_loss ' + str(sen_loss.cpu().detach().numpy())
                      + ' sen_order_loss ' + str(final_sen_order_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.makedirs(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()
    
    # ablation sen_order_loss
    def ablation_sen_order_loss(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        tokenizer.add_special_tokens({"bos_token": "<s>", "additional_special_tokens": self.config.special_token})
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        # model.resize_token_embeddings(len(tokenizer))
        model.config.decoder_start_token_id = tokenizer.bos_token_id
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
                input_text_ids, sentence_order_label = self.get_ids(input_text, tokenizer, sen_id)
                label_text_ids, _ = self.get_ids(label_text, tokenizer, sen_id)
                shuffle_input_text_ids, shuffle_sentence_order_label = self.get_shuffle_sen_ids(input_text, tokenizer, sen_id)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                # self_output = model(input_ids=input_text_ids, input_sentence_types=input_sentence_types, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                # shuffle_outputs = model(input_ids=shuffle_input_text_ids, input_sentence_types=shuffle_input_sentence_types, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                # 去掉 sentence type id
                self_output = model(input_ids=input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                shuffle_outputs = model(input_ids=shuffle_input_text_ids, labels=label_text_ids, transfer_to=style, sen_id=sen_id)
                
                # sentence order loss
                shuffle_sen_order_loss = self.get_sen_order_loss(shuffle_outputs.pointing_res, shuffle_sentence_order_label)
                sen_order_loss = self.get_sen_order_loss(self_output.pointing_res, sentence_order_label)
                final_sen_order_loss = (shuffle_sen_order_loss + sen_order_loss) / 2
                
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id, style)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_classifier_loss = self.get_style_loss(pred_style, style) 
                
                if not torch.isnan(sen_loss):
                    loss = cross_entro_loss + sen_loss + style_classifier_loss #+ final_sen_order_loss
                else:
                    loss = cross_entro_loss + style_classifier_loss #+ final_sen_order_loss
                
                # # ablation sen loss
                # loss = cross_entro_loss + style_classifier_loss + final_sen_order_loss
                
                # # ablation style_classifier_loss
                # if not torch.isnan(sen_loss):
                #     loss = cross_entro_loss + sen_loss + final_sen_order_loss
                # else:
                #     loss = cross_entro_loss + final_sen_order_loss

                # # ablation final_sen_order_loss
                # if not torch.isnan(sen_loss):
                #     loss = cross_entro_loss + sen_loss + style_classifier_loss
                # else:
                #     loss = cross_entro_loss + style_classifier_loss
                
                
                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
                writer.add_scalar('style_classifier_loss', style_classifier_loss, global_step=step)
                # writer.add_scalar('sen_order_loss', final_sen_order_loss, global_step=step)
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
                    #   + ' sen_order_loss ' + str(final_sen_order_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.makedirs(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()
    
    
    
class Fill_Mask_Model(nn.Module):
    def __init__(self, config):
        super(Fill_Mask_Model, self).__init__()
        self.config = config

    def load_data_mask(self, dataset, batch_size, shuffle=True, drop=None):
        dataset = Data_Encoder_Fill(dataset, drop_ratio=drop)
        data_generator = DataLoader(dataset, batch_size, shuffle=shuffle)
        return data_generator

    def load_data_and_res(self, result, dataset, batch_size, shuffle=True):
        dataset = Data_Encoder_Fill_Res(result, dataset)
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
        label_list = ['<LX>', '<JY>', '<GS>']

        for res, sty in zip(predict, label):
            tokens = tokenizer.convert_ids_to_tokens(res)
            final_output = []
            for token in tokens:
                if token in ["<s>", "▁"] or "extra_id" in token:
                    continue
                if "▁" in token:
                    token = token.replace("▁", "")
                if token == "</s>":
                    break
                final_output.append(token)

            item = {
                "text": "".join(final_output),
                "style": sty,
            }
            item = json.dumps(item, ensure_ascii=False)
            f.write(item + '\n')

    def write2text_simple(self, f, predict, tokenizer):
        # label_list = ['<LX>', '<JY>', '<GS>']

        for res in predict:
            tokens = tokenizer.convert_ids_to_tokens(res)
            final_output = []
            for token in tokens:
                if token in ["<s>", "▁"] or "extra_id" in token:
                    continue
                if "▁" in token:
                    token = token.replace("▁", "")
                if token == "</s>":
                    break
                final_output.append(token)

            item = {
                "text": "".join(final_output),
                # "style": sty,
            }
            item = json.dumps(item, ensure_ascii=False)
            f.write(item + '\n')


    def train_fill(self):
        data_generator = self.load_data_mask(self.config.train_set_mask, self.config.batch_size)
        t5_config = AutoConfig.from_pretrained('pretrained_model/t5_small')
        model = T5ForConditionalGeneration(t5_config).to(self.config.device)
        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        # model = T5ForLongText_ST.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
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
                # unk = tokenizer("<KEY>").input_ids
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

    def test_base(self):
        data_generator = self.load_data_mask(self.config.test_set_mask, self.config.test_batch, shuffle=False)
        # t5_config = AutoConfig.from_pretrained('pretrained_model/t5_small')
        # model = T5ForConditionalGeneration(t5_config).to(self.config.device)
        model = T5ForConditionalGeneration.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
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

                    pre = model.generate(ids, do_sample=True, decoder_start_id=1, top_p=0.9, max_length=512)
                    pre = pre.cpu().numpy().tolist()
                    # transfer_to = style.cpu().numpy().tolist()
                    self.write2text_simple(f, pre, tokenizer)

    def test(self):
        data_generator = self.load_data_mask(self.config.test_set_mask, self.config.test_batch, shuffle=False)
        t5_config = AutoConfig.from_pretrained('pretrained_model/t5_small')
        model = T5ForConditionalGeneration(t5_config).to(self.config.device)
        # model = T5ForConditionalGeneration.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
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

                    pre = model.generate(ids, do_sample=True, decoder_start_id=1, top_p=0.9, max_length=512)
                    pre = pre.cpu().numpy().tolist()
                    # transfer_to = style.cpu().numpy().tolist()
                    self.write2text_simple(f, pre, tokenizer)


    def train_fill_base(self):
        # data_generator = self.load_data_mask(self.config.train_set_mask, self.config.batch_size, drop=0.2)
        data_generator = self.load_data_mask(self.config.train_set_mask, self.config.batch_size)
        # t5_config = AutoConfig.from_pretrained('pretrained_model/t5_small')
        t5_config = AutoConfig.from_pretrained(self.config.pre_trained_t5)
        model = T5ForConditionalGeneration(t5_config).to(self.config.device)
        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
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

    def train_fill_LongLM(self):
        # data_generator = self.load_data_mask(self.config.train_set_mask, self.config.batch_size)
        data_generator = self.load_data_mask(self.config.train_set_mask, self.config.batch_size, drop=0.2)

        # t5_config = AutoConfig.from_pretrained('pretrained_model/t5_small')
        # t5_config = AutoConfig.from_pretrained(self.config.pre_trained_t5)
        model = T5ForConditionalGeneration.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
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

        model = T5ForConditionalGeneration.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model.load_state_dict(torch.load(self.config.init), strict=True)
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


    def train_fill_LongLM_jy(self):
        # data_generator = self.load_data_mask(self.config.train_set_mask, self.config.batch_size)
        data_generator = self.load_data_mask(self.config.train_set_mask_jy, self.config.batch_size, drop=0.2)

        # t5_config = AutoConfig.from_pretrained('pretrained_model/t5_small')
        # t5_config = AutoConfig.from_pretrained(self.config.pre_trained_t5)
        model = T5ForConditionalGeneration.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
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
        num_step = self.config.epoch * len(data_generator)
        step = 0
        save_step = num_step // 10
        self.print_tip_for_train(num_step)

        for epoch in range(self.config.epoch):
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


    def train_fill_LongLM_lx(self):
        # data_generator = self.load_data_mask(self.config.train_set_mask, self.config.batch_size)
        data_generator = self.load_data_mask(self.config.train_set_mask_lx, self.config.batch_size, drop=0.2)

        # t5_config = AutoConfig.from_pretrained('pretrained_model/t5_small')
        # t5_config = AutoConfig.from_pretrained(self.config.pre_trained_t5)
        model = T5ForConditionalGeneration.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
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
        num_step = self.config.epoch * len(data_generator)
        step = 0
        save_step = num_step // 10
        self.print_tip_for_train(num_step)

        for epoch in range(self.config.epoch):
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
        
