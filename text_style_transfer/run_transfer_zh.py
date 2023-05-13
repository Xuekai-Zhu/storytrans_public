from argparse import ArgumentParser
from ast import arg
import torch, os
from long_text_style_transfer_zh import LongTextStyleTrans, Fill_Mask_Model



class Config(object):
    def __init__(self, args):
        # task setting
        self.task_name = args.task
        self.device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
        self.max_length = 512

        # training setting
        self.batch_size = 4
        self.epoch = 70
        self.mask_fill_epoch = 5
        self.learning_rate = 5e-5
        self.mask_fill_learning_rate = 5e-5
        self.model_save_dir = "model_add_sen_type" + "/" + args.model
        self.log_dir = "log" + "/" + args.model
        self.pre_trained_t5 = "pretrained_model/LongLM-base"
        self.sen_token = "<SEN>"
        self.margin = 2

        # test setting
        self.test_batch = 9
        self.init = args.init
        self.pred_result_dir = args.pred
        self.eval_style_file = None

        # data
        # self.sen_embs = "./data_ours/auxiliary_data/train.sen.emb.pickle"
        self.sen_embs = "./data_ours/auxiliary_data/train.sen.bt.emb.pickle"
        self.sen_bt = "./data_ours/auxiliary_data/train.sen.bt"
        self.train_set = './data_ours/auxiliary_data/train.sen.add_index'
        self.train_set_mask = './data_ours/auxiliary_data/train.sen.add_index.mask'
        self.train_set_mask_jy = './data_ours/auxiliary_data/train.sen.add_index.mask.jy'
        self.train_set_mask_lx = './data_ours/auxiliary_data/train.sen.add_index.mask.lx'
        self.valid_set = './data_ours/auxiliary_data/valid.sen.add_index'
        self.test_set = './data_ours/auxiliary_data/test.sen.add_index'
        self.test_set_mask = './data_ours/auxiliary_data/test.sen.add_index.mask'
        self.need_fill = args.fill

        # # classifier setting
        self.in_size = 768
        self.style_num = 3
        #

        # pretrained style classifier
        self.pretrained_sc = "pretrained_model/pre-trained_style_classifier/epoch-9-step-4660-loss-1.873672408692073e-05.pth"
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
        task = LongTextStyleTrans(config)
        task.train_sen_mse()
        
    elif config.task_name in ["test_sen_mse"]:
        task = LongTextStyleTrans(config)
        task.test_sen_mse()

    # stage 1
    # train
    elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_2"]:
        task = LongTextStyleTrans(config)
        task.train_sen_mse_add_mask_in_input_ablation_2()
    # test
    elif config.task_name in ["test_sen_mse_add_mask_in_input"]:
        task = LongTextStyleTrans(config)
        task.test_sen_mse_add_mask_in_input()

    # ablation 11.29
    elif config.task_name in ["ablation_sen_loss"]:
        task = LongTextStyleTrans(config)
        task.ablation_sen_loss()
    
    elif config.task_name in ["ablation_style_classifier_loss"]:
        task = LongTextStyleTrans(config)
        task.ablation_style_classifier_loss()
    
    elif config.task_name in ["ablation_sen_order_loss"]:
        task = LongTextStyleTrans(config)
        task.ablation_sen_order_loss()

    # stage 2
    elif config.task_name in ["train_fill"]:
        task = Fill_Mask_Model(config)
        task.train_fill()
    elif config.task_name in ["train_fill_base"]:
        task = Fill_Mask_Model(config)
        task.train_fill_base()

    # one stage tranfer
    elif config.task_name in ["train_one_stage_transfer"]:
        task = LongTextStyleTrans(config)
        task.train_1119()
    # test 
    elif config.task_name in ["test_one_stage_transfer"]:
        task = LongTextStyleTrans(config)
        task.test_1119()
        
        
    # train
    elif config.task_name in ["train_fill_LongLM"]:
        task = Fill_Mask_Model(config)
        task.train_fill_LongLM()

    elif config.task_name in ["train_fill_LongLM_jy"]:
        task = Fill_Mask_Model(config)
        task.train_fill_LongLM_jy()
    elif config.task_name in ["train_fill_LongLM_lx"]:
        task = Fill_Mask_Model(config)
        task.train_fill_LongLM_lx()
    elif config.task_name in ["test_fill"]:
        task = Fill_Mask_Model(config)
        task.test()

    # test
    elif config.task_name in ["test_fill_base"]:
        task = Fill_Mask_Model(config)
        task.test_base()

    # generation
    elif config.task_name in ["fill_mask"]:
        task = Fill_Mask_Model(config)
        task.fill_mask()


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
    # parser.add_argument('--file', default=None, type=str, help='file needed eval style')
    # parser.add_argument('--do_eval', default=False, type=bool, help='do eval')
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # # # # # # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # # args.task = "train_one_stage_transfer"
    # args.task = "train_sen_mse_add_mask_in_input_ablation_2"
    # args.model = "model_1102/test"
    
    
    # test
    # args.task = "test_sen_mse_add_mask_in_input"
    # args.pred = "pred_add_sentype/test"
    # args.init = "model_add_sen_type/orgin_version_only_cross_entropy_1103/epoch-69-step-130480-loss-0.1955500841140747.pth"
    
    
    # fill mask
    # args.task = "fill_mask"
    # args.init = "model/train_fill_LongLM-5e-5-1028/epoch-3-step-7456-loss-0.13943903148174286.pth"
    # args.fill = "pred_add_sentype/add_sentype_1007_v1/stage_1/test_sen_mse_add_mask_in_input.1"
    # args.pred = "pred_add_sentype/add_sentype_1007_v1/stage_2"
    
    main(args)
    