import json, pickle
from data_set import Data_Encoder, Data_Encoder_Sen, Data_Encoder_Mask, Data_Encoder_Fill, Data_Encoder_Fill_Res, Data_Encoder_Mask_Input
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

class LongTextStyleTrans(nn.Module):
    def __init__(self, config):
        super(LongTextStyleTrans, self).__init__()
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

    def load_data_bt(self, dataset, sen_bt, batch_size):
        dataset = Data_Encoder(dataset)
        data_generator = DataLoader(dataset, batch_size, shuffle=True)

        sen_bt_list = []
        with open(sen_bt, 'rb') as f:
            sen_bt = f.readlines()
            for sens in sen_bt:
                item = json.loads(sens)
                text = item["text"]
                sen_bt_list.append(text)

        return data_generator, sen_bt_list

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

    def load_data_mask_in_input(self, dataset, batch_size, shuffle=True):
        dataset = Data_Encoder_Mask_Input(dataset)
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

    def get_style_contrastive_loss(self, style_representation, model, style):
        # detach_style = style_representation.detach()
        loss_func = NTXentLoss()
        emb_label = torch.arange(0, self.config.style_num, 1).to(self.config.device)
        label = torch.cat((style, emb_label), dim=0)
        # emb = torch.cat((detach_style, model.encoder.style_embedding.weight), dim=0)
        emb = torch.cat((style_representation, model.mid_module.style_embedding.weight), dim=0)
        loss = loss_func(emb, label)
        return loss

    def get_cross_entropy_loss(self, logits, label, margin=False):
        # loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=config.pad_token_id)
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        # loss = loss_fct(logits.permute(0, 2, 1), label) / config.batch_size
        loss = loss_fct(logits.permute(0, 2, 1), label)
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

    def get_batch_distance(self, content_hidden, ids_add_sen, sen_id, no_pading=False, all_token=False):
        l_fct = nn.MSELoss()
        distutils_list = []
        sen_posi = torch.where(ids_add_sen == sen_id)
        if no_pading is True:
            for i in range(self.config.batch_size):
                single_content = content_hidden[i]
                distutils_list.append(single_content)
        elif all_token == True:
            a = torch.sum(content_hidden, dim=-1)
            # b = a > 0
            num = torch.sum(a > 0, dim=-1)
            batch_sample = torch.div(torch.sum(content_hidden, dim=1), num.unsqueeze(-1))
            
            for i in range(self.config.batch_size):
                single_content = batch_sample[i]
                distutils_list.append(single_content)
        else:
            for i in range(self.config.batch_size):
                num_dim = torch.sum(sen_posi[0] == i)
                single_content = content_hidden[i, :num_dim]
                single = torch.mean(single_content, dim=0, keepdim=True)
                # index = torch.where(single_content != 0)
                # data_point = single_content[index]
                # var = torch.var(single_content)
                # mu = torch.mean(single_content)
                # multi_normal = MultivariateNormal(mu, torch.diag(var))
                # multi_normal = torch.distributions.normal.Normal(mu, var)
                distutils_list.append(single)

        loss_list = []

        for i in range(self.config.batch_size):
            d_i = distutils_list[i]
            for j in range(1, self.config.batch_size):
                if (i + j) >= self.config.batch_size:
                    break
                d_j = distutils_list[i + j]
                mse = l_fct(d_i, d_j)
                # cos_sim = F.cosine_similarity(d_i, d_j, dim=-1)
                # cos_loss = 1 - cos_sim
                # kl_loss = torch.distributions.kl.kl_divergence(d_i, d_j)
                # loss_list.append(cos_loss)
                loss_list.append(mse.unsqueeze(0))
        # a = torch.cat(loss_list)
        loss = torch.sum(torch.cat(loss_list)) / self.config.batch_size

        return loss

    def get_batch_distance_simple(self, content_hidden):
        l_fct = nn.MSELoss()
        loss_list = []

        for i in range(self.config.batch_size):
            d_i = content_hidden[i]
            for j in range(1, self.config.batch_size):
                if (i + j) >= self.config.batch_size:
                    break
                d_j = content_hidden[i + j]
                mse = l_fct(d_i, d_j)
                # cos_sim = F.cosine_similarity(d_i, d_j, dim=-1)
                # cos_loss = 1 - cos_sim
                # kl_loss = torch.distributions.kl.kl_divergence(d_i, d_j)
                # loss_list.append(cos_loss)
                loss_list.append(mse.unsqueeze(0))
        # a = torch.cat(loss_list)
        loss = torch.sum(torch.cat(loss_list)) / self.config.batch_size

        return loss

    def train_transfer(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt, self.config.batch_size)
        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        # model.load_state_dict(torch.load(self.config.init), strict=True)
        model.train()
        # style_classifer = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)
                # content_label = self.get_content_label(sen_embs_list, index).to(self.config.device)
                content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)

                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label)
                sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, self.config)

                loss = cross_entro_loss + style_contrastive_loss + sen_emb_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy()))

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(), self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step, loss))


        print('training  over')
        writer.close()

    def train_encoder(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST_Disentangled.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        # style_classifer = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                # ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model.encoder(input_ids=ids_add_sen, transfer_to=style)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                sen_emb_loss = self.get_sentence_mse_loss(self_output.content_representation, content_label)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

                # cycle content perseveration
                # predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9,
                #                                     temperature=1.0,
                #                                     max_length=self.config.max_length, transfer_to=trans_style,
                #                                     eos_id=tokenizer.eos_token_id, return_logits=True)
                # predict_logits_exp = F.softmax(predict_logits, dim=-1)
                # cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                # cycle_out = model.encoder(inputs_embeds=cycle_input)
                # cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model, affine=False, margin=False)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)

                loss = style_contrastive_loss + sen_emb_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                # writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('cycle_loss', cycle_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      # + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' cycle_loss ' + str(cycle_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()

    def train_transfer_add_cycle(self):
        data_generator, sen_embs_list = self.load_data(self.config.train_set, self.config.sen_embs, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        # style_classifer = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)
                content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style)

                sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, self.config)

                # cycle content perseveration
                predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                      max_length=self.config.max_length, transfer_to=trans_style,
                                      eos_id=tokenizer.eos_token_id, return_logits=True)
                predict_logits_exp = F.softmax(predict_logits, dim=-1)
                cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                cycle_out = model.encoder(inputs_embeds=cycle_input)
                cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)




                loss = cross_entro_loss + style_contrastive_loss + sen_emb_loss + cycle_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                writer.add_scalar('cycle_loss', cycle_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      + ' cycle_loss ' + str(cycle_loss.cpu().detach().numpy()))

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(), self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step, loss))


        print('training  over')
        writer.close()

    def train_transfer_bt_sen_and_cycle(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        # style_classifer = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                content_label, batch_content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer)
                trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, self.config)

                # cycle content perseveration
                predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9,
                                                    temperature=1.0,
                                                    max_length=self.config.max_length, transfer_to=trans_style,
                                                    eos_id=tokenizer.eos_token_id, return_logits=True)
                predict_logits_exp = F.softmax(predict_logits, dim=-1)
                cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                cycle_out = model.encoder(inputs_embeds=cycle_input)
                cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model, affine=False, margin=False)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)

                loss = cross_entro_loss + style_contrastive_loss + sen_emb_loss + cycle_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                writer.add_scalar('cycle_loss', cycle_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      + ' cycle_loss ' + str(cycle_loss.cpu().detach().numpy()))

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()


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


    def train_transfer_token_mean(self):
        # 使用所有的token mean来计算欧式距离
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST_Disentangled.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        # style_classifer = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                sen_emb_loss = self.get_sentence_mse_loss(self_output.content_representation, content_label)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

                # cycle content perseveration
                # predict_logits, _ = model.inference(input_ids=ids_add_sen, decoder_start_token_id=1, top_p=0.9,
                #                                     temperature=1.0,
                #                                     max_length=self.config.max_length, transfer_to=trans_style,
                #                                     eos_id=tokenizer.eos_token_id, return_logits=True)
                # predict_logits_exp = F.softmax(predict_logits, dim=-1)
                # cycle_input = torch.matmul(predict_logits_exp, model.shared.weight)
                # cycle_out = model.encoder(inputs_embeds=cycle_input)
                # cycle_loss = self.get_cycle_loss(batch_content_label, cycle_out.last_hidden_state, model, affine=False, margin=False)
                # cycle_loss = self.get_cycle_loss(self_output.batch_content, cycle_out.last_hidden_state, model)
                # cycle_loss = self.get_sen_mse_loss(self_output.content_representation, predict.content_representation)

                loss = 0.5 * cross_entro_loss + 0.5 * style_contrastive_loss + sen_emb_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('cycle_loss', cycle_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' cycle_loss ' + str(cycle_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()

    def test_for_token_mean(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)

        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = LongTextST_Disentangled.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)

    def train_concat_new_sen(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST_Concat.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        # style_classifer = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

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

                loss = cross_entro_loss + style_contrastive_loss + sen_emb_loss #+ cycle_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('cycle_loss', cycle_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' cycle_loss ' + str(cycle_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()

    def test_concat_new_sen(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)

        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = LongTextST_Concat.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)

    def train_concat_new_sen_add_inter(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST_Concat_Inter.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)

                soft_input = F.softmax(self_output.logits, dim=-1)
                pred_style = style_classifier(soft_input, soft_sampling=True)
                style_loss = self.get_style_loss(pred_style, style)

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

                loss = cross_entro_loss + style_contrastive_loss + sen_emb_loss + style_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()

    def train_hidden_disturb(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST_Disturb.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # style_classifier = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, bt_sen_representation=content_label, sen_id=sen_id)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

                # soft_input = F.softmax(self_output.logits, dim=-1)
                # pred_style = style_classifier(soft_input, soft_sampling=True)
                # style_loss = self.get_style_loss(pred_style, style)

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

                loss = cross_entro_loss + style_contrastive_loss + sen_emb_loss #+ style_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()


    def test_disturb(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = LongTextST_Disturb.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id, project_linear=model.project_content_768)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    def train_hidden_attention(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST_Attention.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # style_classifier = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, bt_sen_representation=content_label, sen_id=sen_id)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

                # soft_input = F.softmax(self_output.logits, dim=-1)
                # pred_style = style_classifier(soft_input, soft_sampling=True)
                # style_loss = self.get_style_loss(pred_style, style)

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

                loss = cross_entro_loss + style_contrastive_loss + sen_emb_loss #+ style_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()

    def test_hidden_attention(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = LongTextST_Attention.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    # 内容保留实验：直接拼接style emb
    def train_hidden_test(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST_Test.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # style_classifier = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, bt_sen_representation=content_label, sen_id=sen_id)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

                # soft_input = F.softmax(self_output.logits, dim=-1)
                # pred_style = style_classifier(soft_input, soft_sampling=True)
                # style_loss = self.get_style_loss(pred_style, style)

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

                loss = cross_entro_loss + style_contrastive_loss + sen_emb_loss #+ style_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()

    def test_hidden_test(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = LongTextST_Test.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)

    #
    def train_content_attention(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST_Style.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # style_classifier = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, bt_sen_representation=content_label, sen_id=sen_id)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

                # soft_input = F.softmax(self_output.logits, dim=-1)
                # pred_style = style_classifier(soft_input, soft_sampling=True)
                # style_loss = self.get_style_loss(pred_style, style)

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

                loss = cross_entro_loss + style_contrastive_loss + sen_emb_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()

    def test_content_attention(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = LongTextST_Style.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)

    # 利用sen 对token计算 reverse attention；并替换sen 表示为 style emb
    def train_reverse_style_attention(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST_Style_Attention.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # style_classifier = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

                # soft_input = F.softmax(self_output.logits, dim=-1)
                # pred_style = style_classifier(soft_input, soft_sampling=True)
                # style_loss = self.get_style_loss(pred_style, style)

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

                loss = cross_entro_loss + style_contrastive_loss #+ sen_emb_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()

    def test_reverse_style_attention(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = LongTextST_Style_Attention.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)

    # 替换 sen -> style emb
    def train_style_change(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST_Style_Change.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # style_classifier = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

                # soft_input = F.softmax(self_output.logits, dim=-1)
                # pred_style = style_classifier(soft_input, soft_sampling=True)
                # style_loss = self.get_style_loss(pred_style, style)

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

                loss = cross_entro_loss + style_contrastive_loss #+ sen_emb_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()

    def test_style_change(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = LongTextST_Style_Change.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)

    #
    def train_content_distribution(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST_Content_Dis.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # style_classifier = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                # content_distribution = self.get_content_disturibution(model, self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                content_distribution = self.get_content_single_normal(self_output.encoder_last_hidden_state, ids_add_sen, sen_id) * 0.5
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

                # soft_input = F.softmax(self_output.logits, dim=-1)
                # pred_style = style_classifier(soft_input, soft_sampling=True)
                # style_loss = self.get_style_loss(pred_style, style)

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

                loss = cross_entro_loss + style_contrastive_loss + content_distribution

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' content_distribution ' + str(content_distribution.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()

    def test_content_distribution(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = LongTextST_Content_Dis.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    def train_content_distribution_on_sen(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = LongTextST_Content_Dis_And_Attention.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # style_classifier = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)

                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                # content_distribution = self.get_content_disturibution(model, self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_single_normal(self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                content_distribution = self.get_content_normal_half_sen(self_output.content_representation) * 0.5
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

                # soft_input = F.softmax(self_output.logits, dim=-1)
                # pred_style = style_classifier(soft_input, soft_sampling=True)
                # style_loss = self.get_style_loss(pred_style, style)

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

                loss = cross_entro_loss + style_contrastive_loss + content_distribution

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' content_distribution ' + str(content_distribution.cpu().detach().numpy())
                      # + ' sen_emb_loss ' + str(sen_emb_loss.cpu().detach().numpy())
                      # + ' style_loss ' + str(style_loss.cpu().detach().numpy())
                      )

                if epoch >= 0 and step % save_step == 0:
                    if not os.path.exists(self.config.model_save_dir):
                        os.mkdir(self.config.model_save_dir)
                    torch.save(model.state_dict(),
                               self.config.model_save_dir + '/' + 'epoch-{}-step-{}-loss-{}.pth'.format(epoch, step,
                                                                                                        loss))

        print('training  over')
        writer.close()

    def test_content_distribution_on_sen(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = LongTextST_Content_Dis_And_Attention.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)

    def train_sen_generate(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # sty_id = self.get_sty_id(tokenizer)
        # style_classifier = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                # content_distribution = self.get_content_disturibution(model, self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_single_normal(self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_normal_half_sen(self_output.content_representation) * 0.5
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

                # soft_input = F.softmax(self_output.logits, dim=-1)
                # pred_style = style_classifier(soft_input, soft_sampling=True)
                # style_loss = self.get_style_loss(pred_style, style)

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

                loss = cross_entro_loss + style_contrastive_loss + sen_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
                # writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
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


    def test_sen_generate(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    def train_toekn_mean_generate(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
        model.train()
        sen_id = self.get_sen_id(tokenizer)
        # sty_id = self.get_sty_id(tokenizer)
        # style_classifier = self.load_style_classifier()

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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                # content_distribution = self.get_content_disturibution(model, self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_single_normal(self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_normal_half_sen(self_output.content_representation) * 0.5
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                # sen_loss = self.get_sen_distribution_loss(self_output.content_representation, ids_add_sen, sen_id)
                sen_loss = self.get_batch_mid_sen_loss(self_output.content_representation)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

                # soft_input = F.softmax(self_output.logits, dim=-1)
                # pred_style = style_classifier(soft_input, soft_sampling=True)
                # style_loss = self.get_style_loss(pred_style, style)

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

                loss = cross_entro_loss + style_contrastive_loss + sen_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
                # writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
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


    def test_toekn_mean_generate(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    def train_toekn_mean_generate_without_sty(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_ori(self.config.train_set, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST_without_sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                # content_distribution = self.get_content_disturibution(model, self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_single_normal(self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_normal_half_sen(self_output.content_representation) * 0.5
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                # sen_loss = self.get_sen_distribution_loss(self_output.content_representation, ids_add_sen, sen_id)
                sen_loss = self.get_batch_mid_sen_loss(self_output.content_representation, ids_add_sen, sen_id)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids) * 0.4

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

                loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
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
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
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


    def test_toekn_mean_generate_without_sty(self):
        data_generator = self.load_data_ori_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST_without_sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    def train_sen_generate_with_kl_loss(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                # content_distribution = self.get_content_disturibution(model, self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_single_normal(self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_normal_half_sen(self_output.content_representation) * 0.5
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                sen_loss = self.get_batch_mid_sen_loss(self_output.content_representation, ids_add_sen, sen_id)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

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

                loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
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
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
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


    def test_sen_generate_with_kl_loss(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    def train_sen_generate_add_mask(self):
        data_generator = self.load_data_mask(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
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
            for i, (text_ori, batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                # content_distribution = self.get_content_disturibution(model, self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_single_normal(self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_normal_half_sen(self_output.content_representation) * 0.5
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

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

                loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
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
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
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


    def test_sen_generate_add_mask(self):
        data_generator = self.load_data_mask_test(self.config.test_set_mask, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    def train_sen_mse(self):
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
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
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                ids = self.get_ids(batch_text, tokenizer)
                ids_add_sen = self.get_ids(batch_text_add_sen, tokenizer)

                # content_label, batch_content_label = self.get_content_label(sen_embs_list, index)
                # content_label = self.get_bt_content_label(sen_bt_list, index, model, tokenizer, batch_sen=False)
                # trans_style = self.get_trans_style(style)

                # self construction
                # self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                self_output = model(input_ids=ids_add_sen, labels=ids, transfer_to=style, sen_id=sen_id)
                # sen_emb_loss = self.get_sen_emb_loss(self_output.content_representation, content_label, model)
                # sen_emb_loss = self.get_sen_mse_loss(self_output.content_representation, content_label, model)
                # content_distribution = self.get_content_disturibution(model, self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_single_normal(self_output.encoder_last_hidden_state, ids_add_sen, sen_id)
                # content_distribution = self.get_content_normal_half_sen(self_output.content_representation) * 0.5
                # style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                sen_loss = self.get_batch_distance(self_output.content_representation, ids_add_sen, sen_id)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids)

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

                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss
                loss = cross_entro_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
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


    def test_sen_mse(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    def train_sen_mse_add_mask(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask(self.config.train_set_mask, self.config.batch_size)

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
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id)
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

                loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
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
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
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


    def test_sen_mse_add_mask(self):
        data_generator = self.load_data_mask(self.config.test_set_mask, self.config.test_batch, shuffle=False)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

            if not os.path.exists(self.config.pred_result_dir):
                os.mkdir(self.config.pred_result_dir)

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
                        input_text_ids = self.get_ids(input_text, tokenizer)
                        style = torch.ones(self.config.test_batch, dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=input_text_ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)



    def train_sen_mse_add_mask_in_input(self):
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
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id)
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

                loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
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
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
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


    def test_sen_mse_add_mask_in_input(self):
        data_generator = self.load_data_mask_in_input(self.config.test_set_mask, self.config.test_batch, shuffle=False)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST_Sen_Sty.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

            if not os.path.exists(self.config.pred_result_dir):
                os.mkdir(self.config.pred_result_dir)

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
                        input_text_ids = self.get_ids(input_text, tokenizer)
                        style = torch.ones(self.config.test_batch, dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=input_text_ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    def train_sen_mse_add_mask_in_input_project_sen(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST_Sen_Sty_ProSen.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
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
                sen_loss = self.get_batch_distance_simple(self_output.content_representation)
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

                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss
                loss = cross_entro_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
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


    def test_sen_mse_add_mask_in_input_project_sen(self):
        data_generator = self.load_data_mask_in_input(self.config.test_set_mask, self.config.test_batch, shuffle=False)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST_Sen_Sty_ProSen.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

            if not os.path.exists(self.config.pred_result_dir):
                os.mkdir(self.config.pred_result_dir)

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
                        input_text_ids = self.get_ids(input_text, tokenizer)
                        style = torch.ones(self.config.test_batch, dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=input_text_ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    def train_sen_mse_add_mask_in_input_project_sen_att(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST_Sen_Sty_ProSen_Att.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
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
                sen_loss = self.get_batch_distance_simple(self_output.content_representation)
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

                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss
                loss = cross_entro_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
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


    def test_sen_mse_add_mask_in_input_project_sen_att(self):
        data_generator = self.load_data_mask_in_input(self.config.test_set_mask, self.config.test_batch, shuffle=False)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST_Sen_Sty_ProSen_Att.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

            if not os.path.exists(self.config.pred_result_dir):
                os.mkdir(self.config.pred_result_dir)

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
                        input_text_ids = self.get_ids(input_text, tokenizer)
                        style = torch.ones(self.config.test_batch, dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=input_text_ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)


    def train_sen_mse_add_mask_in_input_project_in_fix_len(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST_Sen_Sty_ProSen_Att_Fix_Len.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
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
                sen_loss = self.get_batch_distance_simple(self_output.content_representation)
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

                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss
                loss = cross_entro_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
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


    def test_sen_mse_add_mask_in_input_project_in_fix_len(self):
        data_generator = self.load_data_mask_in_input(self.config.test_set_mask, self.config.test_batch, shuffle=False)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST_Sen_Sty_ProSen_Att_Fix_Len.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

            if not os.path.exists(self.config.pred_result_dir):
                os.mkdir(self.config.pred_result_dir)

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
                        input_text_ids = self.get_ids(input_text, tokenizer)
                        style = torch.ones(self.config.test_batch, dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=input_text_ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
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
    def train_sen_mse_add_mask_in_input_ablation_1(self):
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
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, ids, margin=True)
                # cross_entro_loss = self.get_cross_entropy_loss(self_output.logits, label_text_ids)

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

                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss
                loss = style_contrastive_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                # writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
                writer.add_scalar('style_classifier_loss', style_classifier_loss, global_step=step)
                # writer.add_scalar('content_distribution', content_distribution, global_step=step)
                # writer.add_scalar('sen_emb_loss', sen_emb_loss, global_step=step)
                # writer.add_scalar('style_loss', style_loss, global_step=step)

                opt.zero_grad()
                loss.backward()
                opt.step()

                print('Training at Epoch ' + str(epoch + 1) + ' step ' + str(step) +
                      ' total loss ' + str(loss.cpu().detach().numpy())
                      # + ' cross_entro_loss ' + str(cross_entro_loss.cpu().detach().numpy())
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
                      + ' style_classifier_loss ' + str(style_classifier_loss.cpu().detach().numpy())
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


    def train_sen_mse_add_mask_in_input_ablation_2(self):
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

                loss = cross_entro_loss + sen_loss + style_classifier_loss
                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
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


    def train_sen_mse_add_mask_in_input_ablation_3(self):
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
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
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

                loss = cross_entro_loss + style_contrastive_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
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
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
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


    def train_sen_mse_add_mask_in_input_ablation_4(self):
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
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
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

                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss
                loss = cross_entro_loss + style_contrastive_loss + sen_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
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
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
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


    def train_sen_mse_add_mask_in_input_ablation_5(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST_Sen_Sty_Ablation_Sen.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
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
                style_contrastive_loss = self.get_style_contrastive_loss(self_output.style_representation, model, style)
                # sen_loss = self.get_sen_mse_loss(self_output.content_representation, self_output.content_label)
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id, no_pading=True)
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

                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss
                loss = cross_entro_loss + style_contrastive_loss + sen_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
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
                      + ' style_contrastive_loss ' + str(style_contrastive_loss.cpu().detach().numpy())
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

    def test_sen_mse_add_mask_in_input_ablation_5(self):
        data_generator = self.load_data_mask_in_input(self.config.test_set_mask, self.config.test_batch, shuffle=False)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST_Sen_Sty_Ablation_Sen.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

            if not os.path.exists(self.config.pred_result_dir):
                os.mkdir(self.config.pred_result_dir)

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
                        input_text_ids = self.get_ids(input_text, tokenizer)
                        style = torch.ones(self.config.test_batch, dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=input_text_ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
                                              max_length=self.config.max_length, transfer_to=style,
                                              eos_id=tokenizer.eos_token_id, sen_id=sen_id)
                        pre = pre.cpu().numpy().tolist()
                        transfer_to = style.cpu().numpy().tolist()
                        self.write2text(f, pre, tokenizer, transfer_to)

    # new ablation
    # 1) token mean ：使用sen的mean作为每个的输出 句子的输出
    # 2) style classifier loss
    # 3) sen loss

    def train_sen_mse_add_mask_in_input_ablation_token_mean(self):
        # data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)
        data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
                                                        self.config.batch_size)
        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST_Sen_token_mean.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
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
            # for i, (input_text, label_text, style, index) in enumerate(data_generator):
            for i, (batch_text, batch_text_add_sen, style, index) in enumerate(data_generator):
                style = style.to(self.config.device)
                label_text_ids = self.get_ids(batch_text, tokenizer)
                input_text_ids = self.get_ids(batch_text_add_sen, tokenizer)

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
        # data_generator = self.load_data_mask_in_input(self.config.test_set_mask, self.config.test_batch, shuffle=False)
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST_Sen_token_mean.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

            if not os.path.exists(self.config.pred_result_dir):
                os.mkdir(self.config.pred_result_dir)

            transfer_label_list = [0, 1]
            result = self.config.pred_result_dir + '/' + '{}.'.format(self.config.task_name)
            out_file_list = [result + str(i) for i in transfer_label_list]
            print('begin predicting')

            for rev_label, out_file in zip(transfer_label_list, out_file_list):
                with open(out_file, 'w') as f:
                    # for i, (input_text, label_text, style, index) in enumerate(tqdm(data_generator)):
                    for i, (batch_text, batch_text_add_sen, style, index) in enumerate(tqdm(data_generator)):
                        # ids = self.get_ids(batch_text, tokenizer)
                        # style = style.to(self.config.device)
                        # label_text_ids = self.get_ids(label_text, tokenizer)
                        input_text_ids = self.get_ids(batch_text_add_sen, tokenizer)
                        style = torch.ones(self.config.test_batch, dtype=torch.long).to(self.config.device) * rev_label
                        # sen_id = tokenizer("<SEN>")

                        pre = model.inference(input_ids=input_text_ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
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

    # 消融实验，获得所有的token
    def train_sen_mse_add_mask_in_input_ablation_sen(self):
        # data_generator, sen_bt_list = self.load_data_bt(self.config.train_set, self.config.sen_bt,
        #                                                 self.config.batch_size)
        data_generator = self.load_data_mask_in_input(self.config.train_set_mask, self.config.batch_size)

        tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
        model = T5ForLongText_ST_Sen_Sty_get_all_token.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
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
                sen_loss = self.get_batch_distance(self_output.content_representation, input_text_ids, sen_id, all_token=True)
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

                loss = cross_entro_loss + sen_loss + style_classifier_loss
                # loss = cross_entro_loss + style_contrastive_loss + sen_loss + style_classifier_loss

                step += 1
                writer.add_scalar('total loss', loss, global_step=step)
                writer.add_scalar('cross_entro_loss', cross_entro_loss, global_step=step)
                # writer.add_scalar('style_contrastive_loss', style_contrastive_loss, global_step=step)
                writer.add_scalar('sen_loss', sen_loss, global_step=step)
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


    def test_sen_mse_ablation_get_all_token(self):
        data_generator = self.load_data_test(self.config.test_set, self.config.test_batch)
        with torch.no_grad():
            tokenizer = T5Tokenizer.from_pretrained(self.config.pre_trained_t5)
            model = T5ForLongText_ST_Sen_Sty_get_all_token.from_pretrained(self.config.pre_trained_t5).to(self.config.device)
            model.load_state_dict(torch.load(self.config.init), strict=True)
            model.eval()
            sen_id = self.get_sen_id(tokenizer)

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

                        pre = model.inference(input_ids=ids, decoder_start_token_id=1, top_p=0.9, temperature=1.0,
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