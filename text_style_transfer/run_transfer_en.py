from argparse import ArgumentParser
from ast import arg
import torch, os
from long_text_style_transfer_en import Fill_Mask_Model, LongTextStyleTrans_En



class Config(object):
    def __init__(self, args):
        # task setting
        self.task_name = args.task
        self.device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
        self.max_length = 512

        # training setting
        self.batch_size = 4
        self.epoch = 80
        self.mask_fill_epoch = 5
        self.learning_rate = 5e-5
        self.mask_fill_learning_rate = 5e-5
        self.model_save_dir = "model" + "/" + args.model
        self.log_dir = "log" + "/" + args.model
        # self.pre_trained_t5 = "pretrained_model/LongLM-base"
        self.pre_trained_t5 = "pretrained_model/t5-base"
        self.sen_token = "<SEN>"
        self.margin = 2

        # test setting
        self.test_batch = 2
        self.init = args.init
        self.pred_result_dir = args.pred
        self.eval_style_file = None

        # data
        # if args.lan == "en":
        # en
        self.train_set_mask = 'data_ours/Longtext_en/sp+story/style_transfer_data/train.mask'
        self.test_set_mask = 'data_ours/Longtext_en/sp+story/style_transfer_data/test.mask'

        # # zh
        #     self.train_set_mask = './data_ours/auxiliary_data/train.sen.add_index.mask'
        #     self.test_set_mask = 'data_ours/Longtext_en/sp+story/style_transfer_data/test.mask'
        
        # self.sen_embs = "./data_ours/auxiliary_data/train.sen.emb.pickle"
        # self.sen_embs = "./data_ours/auxiliary_data/train.sen.bt.emb.pickle"
        # self.sen_bt = "./data_ours/auxiliary_data/train.sen.bt"
        # self.train_set = './data_ours/auxiliary_data/train.sen.add_index'
        # self.train_set_mask = 'data_ours/Longtext_en/sp+story/style_transfer_data/train.mask'
        # self.train_set_mask_jy = './data_ours/auxiliary_data/train.sen.add_index.mask.jy'
        # self.train_set_mask_lx = './data_ours/auxiliary_data/train.sen.add_index.mask.lx'
        # self.valid_set = './data_ours/auxiliary_data/valid.sen.add_index'
        # self.test_set = './data_ours/auxiliary_data/test.sen.add_index'
        # self.test_set_mask = 'data_ours/Longtext_en/sp+story/style_transfer_data/test.mask'
        # self.test_set_mask = 'data_ours/Longtext_en/sp+story/style_transfer_data/train.mask'
        
        self.need_fill = args.fill

        # # classifier setting
        self.in_size = 768
        # self.style_num = 3
        
        #

        # pretrained style classifier
        # self.pretrained_sc = "pretrained_model/pre-trained_style_classifier_en/epoch-2-step-5148-loss-0.00016861417680047452.pth"
        # if args.lan == "en":
        self.pretrained_sc = "pretrained_model/pre-trained_style_classifier_2_style/epoch-2-step-580-loss-2.6538571546552703e-05.pth"
        self.style_num = 2
            
        # elif args.lan == "zh":
        #     self.pretrained_sc = "pretrained_model/pre-trained_style_classifier/epoch-9-step-4660-loss-1.873672408692073e-05.pth"
        #     self.style_num = 3
            #
        
        # # tokenizer setting
        self.sen_id = 32003
        self.pad_token_id = 5
        self.decoder_start_token_id = 1
        self.eos_token_id = 2
        # self.special_token = ['<MT>', '<JK>', '<St>', '<SEN>', '<mask>']
        self.special_token = ['<Sp>', '<St>', '<SEN>', '<mask>']
        self.special_token_fill = ["<KEY>", '<mask>']
        




def main(args):
    config = Config(args)
    print('task name : {}'.format(args.task))

    # only one stage
    if config.task_name in ["train_sen_mse"]:
        task = LongTextStyleTrans_En(config)
        task.train_sen_mse()
    elif config.task_name in ["test_sen_mse"]:
        task = LongTextStyleTrans_En(config)
        task.test_sen_mse()




    # stage 1
    # train
    elif config.task_name in ["tran_transfer_stage_1"]:
        task = LongTextStyleTrans_En(config)
        task.tran_transfer_stage_1()
    # test
    elif config.task_name in ["test_transfer_stage_1"]:
        task = LongTextStyleTrans_En(config)
        task.test_transfer_stage_1()



    # stage 2
    elif config.task_name in ["train_fill"]:
        task = Fill_Mask_Model(config)
        task.train_fill()
    elif config.task_name in ["train_fill_base"]:
        task = Fill_Mask_Model(config)
        task.train_fill_base()

    # train
    elif config.task_name in ["train_fill_LongLM"]:
        task = Fill_Mask_Model(config)
        task.train_fill_LongLM()

    # elif config.task_name in ["test_fill"]:
    #     task = Fill_Mask_Model(config)
    #     task.test()

    # test
    elif config.task_name in ["test_fill"]:
        task = Fill_Mask_Model(config)
        task.test_base()

    # generation
    elif config.task_name in ["fill_mask"]:
        task = Fill_Mask_Model(config)
        task.fill_mask()



    # nablation
    elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_token_mean"]:
        task = LongTextStyleTrans_En(config)
        task.train_sen_mse_add_mask_in_input_ablation_token_mean()
    elif config.task_name in ["test_sen_mse_add_mask_in_input_ablation_token_mean"]:
        task = LongTextStyleTrans_En(config)
        task.test_sen_mse_add_mask_in_input_ablation_token_mean()
    # elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_style_classifier"]:
    #     task = LongTextStyleTrans_En(config)
    #     task.train_sen_mse_add_mask_in_input_ablation_style_classifier()
    # elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_sen_loss"]:
    #     task = LongTextStyleTrans_En(config)
    #     task.train_sen_mse_add_mask_in_input_ablation_sen_loss()


    elif config.task_name in ["train_reverse_style_attention"]:
        task = LongTextStyleTrans_En(config)
        task.train_reverse_style_attention()
    elif config.task_name in ["test_reverse_style_attention"]:
        task = LongTextStyleTrans_En(config)
        task.test_reverse_style_attention()

    elif config.task_name in ["train_ablation_con_enh"]:
        task = LongTextStyleTrans_En(config)
        task.train_ablation_con_enh()
    elif config.task_name in ["test_ablation_con_enh"]:
        task = LongTextStyleTrans_En(config)
        task.test_ablation_con_enh()
        
        
    # ablation 1129
    elif config.task_name in ["ablation_sen_loss"]:
        task = LongTextStyleTrans_En(config)
        task.ablation_sen_loss()
    
    elif config.task_name in ["ablation_style_classifier_loss"]:
        task = LongTextStyleTrans_En(config)
        task.ablation_style_classifier_loss()
    
    elif config.task_name in ["ablation_sen_order_loss"]:
        task = LongTextStyleTrans_En(config)
        task.ablation_sen_order_loss()
        
    # use 1 stage tranfer
    elif config.task_name in ["train_use_1_stage"]:
        task = LongTextStyleTrans_En(config)
        task.tran_transfer_use1_stage()
    elif config.task_name in ["test_use_1_stage"]:
        task = LongTextStyleTrans_En(config)
        task.test_transfer_use1_stage()
    
if __name__ == '__main__':
    # get parameter
    parser = ArgumentParser()
    # parser.add_argument('-b', '--batch_size', default=32, type=int,
    #                     metavar='N',
    #                     help='mini-batch size (default: 32), this is the total '
    #                          'batch size of all GPUs on the current node when '
    #                          'using Data Parallel or Distributed Data Parallel')
    # parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
    #                     help='number of data loading workers (default: 0)')
    # parser.add_argument('--epochs', default=50, type=int, metavar='N',
    #                     help='number of total epochs to run')
    parser.add_argument('--task', default='train', type=str, metavar='TASK',
                        help='Task name. Could be train, test, or else.')
    # parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
    #                     metavar='LR', help='initial learning rate', dest='lr')
    # parser.add_argument('--config', default='./config.json', type=str, help='model config file path')
    # # parser.add_argument('--log', default='training_log', type=str, help='training log')
    parser.add_argument('--model', default='train', type=str, help='model save path')
    # # parser.add_argument('--device', default='0', type=str, help='name of GPU')
    parser.add_argument('--init', default=None, type=str, help='init checkpoint')
    parser.add_argument('--pred', default='predict', type=str, help='result save path')
    parser.add_argument('--fill', default='predict', type=str, help='file needed fill')
    parser.add_argument('--lan', default='en', type=str, help=' which language dataset to use')
    # parser.add_argument('--file', default=None, type=str, help='file needed eval style')
    # parser.add_argument('--do_eval', default=False, type=bool, help='do eval')
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # # # train_sc
    # args.task = "train_use_1_stage"
    # args.lan = "zh"

    # # test


    # # # test
    # args.task = "test_ablation_con_enh"
    # args.init = "model/train_ablation_con_enh_0301/epoch-23-step-13920-loss-1.3085774183273315.pth"
    # args.pred = "pred_stage_1/test_ablation_con_enh_0301-s-13920"
    # # # # #
    # # args.task = "test_reverse_style_attention"
    # args.init = "model/tran_transfer_stage_1_cp_0.25_1224/epoch-39-step-23200-loss-0.726675808429718.pth"




    # fill mask
    # args.task = "fill_mask"
    # # args.task = "train_fill_LongLM"
    # args.task = "test_fill"
    # args.init = "model/train_fill_LongLM-en-2-style-1222/epoch-3-step-2320-loss-0.08412856608629227.pth"
    # args.pred = "pred_fill/train_fill_LongLM-en-2-style-1222-s-2320"
    # args.pred = "pred_stage_2/tran_transfer_stage_1_style_2_1222-s-18560-t5-base-2320"
    # args.fill = "pred_stage_1/tran_transfer_stage_1_1224-s-23200/test_transfer_stage_1.0"

    # 此文件为代码最终版本文件！！！！！！！！
    main(args)