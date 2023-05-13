from argparse import ArgumentParser
import torch, os
from long_text_style_transfer import LongTextStyleTrans, Fill_Mask_Model



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


def main(args):
    config = Config(args)
    print('task name : {}'.format(args.task))

    if config.task_name in ["train"]:
        task = LongTextStyleTrans(config)
        task.train_transfer()
    elif config.task_name in ["test"]:
        task = LongTextStyleTrans(config)
        task.test()
    elif config.task_name in ["train_encoder"]:
        task = LongTextStyleTrans(config)
        task.train_encoder()
    elif config.task_name in ["train_transfer_add_cycle"]:
        task = LongTextStyleTrans(config)
        task.train_transfer_add_cycle()
    elif config.task_name in ["train_transfer_bt_sen_and_cycle"]:
        task = LongTextStyleTrans(config)
        task.train_transfer_bt_sen_and_cycle()
    elif config.task_name in ["train_token_mean"]:
        task = LongTextStyleTrans(config)
        task.train_transfer_token_mean()
    elif config.task_name in ["test_for_token_mean"]:
        task = LongTextStyleTrans(config)
        task.test_for_token_mean()
    elif config.task_name in ["train_concat_new_sen"]:
        task = LongTextStyleTrans(config)
        task.train_concat_new_sen()
    elif config.task_name in ["test_concat_new_sen"]:
        task = LongTextStyleTrans(config)
        task.test_concat_new_sen()
    elif config.task_name in ["train_concat_new_sen_add_inter"]:
        task = LongTextStyleTrans(config)
        task.train_concat_new_sen_add_inter()
    elif config.task_name in ["train_hidden_disturb"]:
        task = LongTextStyleTrans(config)
        task.train_hidden_disturb()
    elif config.task_name in ["test_disturb"]:
        task = LongTextStyleTrans(config)
        task.test_disturb()
    elif config.task_name in ["train_hidden_attention"]:
        task = LongTextStyleTrans(config)
        task.train_hidden_attention()
    elif config.task_name in ["test_hidden_attention"]:
        task = LongTextStyleTrans(config)
        task.test_hidden_attention()
    elif config.task_name in ["train_hidden_test"]:
        task = LongTextStyleTrans(config)
        task.train_hidden_test()
    elif config.task_name in ["train_content_attention"]:
        task = LongTextStyleTrans(config)
        task.train_content_attention()
    elif config.task_name in ["test_hidden_test"]:
        task = LongTextStyleTrans(config)
        task.test_hidden_test()
    elif config.task_name in ["test_content_attention"]:
        task = LongTextStyleTrans(config)
        task.test_content_attention()
    elif config.task_name in ["train_reverse_style_attention"]:
        task = LongTextStyleTrans(config)
        task.train_reverse_style_attention()
    elif config.task_name in ["test_reverse_style_attention"]:
        task = LongTextStyleTrans(config)
        task.test_reverse_style_attention()
    elif config.task_name in ["train_content_distribution"]:
        task = LongTextStyleTrans(config)
        task.train_content_distribution()
    elif config.task_name in ["test_content_distribution"]:
        task = LongTextStyleTrans(config)
        task.test_content_distribution()
    elif config.task_name in ["train_content_distribution_on_sen"]:
        task = LongTextStyleTrans(config)
        task.train_content_distribution_on_sen()
    elif config.task_name in ["test_content_distribution_on_sen"]:
        task = LongTextStyleTrans(config)
        task.test_content_distribution_on_sen()
    elif config.task_name in ["train_sen_generate"]:
        task = LongTextStyleTrans(config)
        task.train_sen_generate()
    elif config.task_name in ["test_sen_generate"]:
        task = LongTextStyleTrans(config)
        task.test_sen_generate()
    elif config.task_name in ["train_toekn_mean_generate"]:
        task = LongTextStyleTrans(config)
        task.train_toekn_mean_generate()
    elif config.task_name in ["test_toekn_mean_generate"]:
        task = LongTextStyleTrans(config)
        task.test_toekn_mean_generate()
    elif config.task_name in ["train_toekn_mean_generate_without_sty"]:
        task = LongTextStyleTrans(config)
        task.train_toekn_mean_generate_without_sty()
    elif config.task_name in ["test_toekn_mean_generate_without_sty"]:
        task = LongTextStyleTrans(config)
        task.test_toekn_mean_generate_without_sty()
    elif config.task_name in ["train_sen_generate_add_mask"]:
        task = LongTextStyleTrans(config)
        task.train_sen_generate_add_mask()
    elif config.task_name in ["test_sen_generate_add_mask"]:
        task = LongTextStyleTrans(config)
        task.test_sen_generate_add_mask()
    elif config.task_name in ["train_sen_generate_with_kl_loss"]:
        task = LongTextStyleTrans(config)
        task.train_sen_generate_with_kl_loss()
    elif config.task_name in ["test_sen_generate_with_kl_loss"]:
        task = LongTextStyleTrans(config)
        task.test_sen_generate_with_kl_loss()
    elif config.task_name in ["train_sen_mse_add_mask"]:
        task = LongTextStyleTrans(config)
        task.train_sen_mse_add_mask()
    elif config.task_name in ["train_sen_mse_add_mask_in_input"]:
        task = LongTextStyleTrans(config)
        task.train_sen_mse_add_mask_in_input()
    elif config.task_name in ["test_sen_mse_add_mask"]:
        task = LongTextStyleTrans(config)
        task.test_sen_mse_add_mask()
    elif config.task_name in ["test_sen_mse_add_mask_in_input"]:
        task = LongTextStyleTrans(config)
        task.test_sen_mse_add_mask_in_input()
    elif config.task_name in ["train_sen_mse_add_mask_in_input_project_sen"]:
        task = LongTextStyleTrans(config)
        task.train_sen_mse_add_mask_in_input_project_sen()
    elif config.task_name in ["test_sen_mse_add_mask_in_input_project_sen"]:
        task = LongTextStyleTrans(config)
        task.test_sen_mse_add_mask_in_input_project_sen()
    elif config.task_name in ["train_sen_mse_add_mask_in_input_project_sen_att"]:
        task = LongTextStyleTrans(config)
        task.train_sen_mse_add_mask_in_input_project_sen_att()
    elif config.task_name in ["test_sen_mse_add_mask_in_input_project_sen_att"]:
        task = LongTextStyleTrans(config)
        task.test_sen_mse_add_mask_in_input_project_sen_att()
    elif config.task_name in ["train_sen_mse_add_mask_in_input_project_in_fix_len"]:
        task = LongTextStyleTrans(config)
        task.train_sen_mse_add_mask_in_input_project_in_fix_len()
    elif config.task_name in ["test_sen_mse_add_mask_in_input_project_in_fix_len"]:
        task = LongTextStyleTrans(config)
        task.test_sen_mse_add_mask_in_input_project_in_fix_len()




    # only one stage

    elif config.task_name in ["train_sen_mse"]:
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




    # # ablation
    # elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_1"]:
    #     task = LongTextStyleTrans(config)
    #     task.train_sen_mse_add_mask_in_input_ablation_1()
    # elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_2"]:
    #     task = LongTextStyleTrans(config)
    #     task.train_sen_mse_add_mask_in_input_ablation_2()
    # elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_3"]:
    #     task = LongTextStyleTrans(config)
    #     task.train_sen_mse_add_mask_in_input_ablation_3()
    # elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_4"]:
    #     task = LongTextStyleTrans(config)
    #     task.train_sen_mse_add_mask_in_input_ablation_4()
    # elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_5"]:
    #     task = LongTextStyleTrans(config)
    #     task.train_sen_mse_add_mask_in_input_ablation_5()
    # elif config.task_name in ["test_sen_mse_add_mask_in_input_ablation_5"]:
    #     task = LongTextStyleTrans(config)
    #     task.test_sen_mse_add_mask_in_input_ablation_5()
    # elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_token_mean_6"]:
    #     task = LongTextStyleTrans(config)
    #     task.train_sen_mse_add_mask_in_input_ablation_6()
    # elif config.task_name in ["test_sen_mse_add_mask_in_input_ablation_token_mean_6"]:
    #     task = LongTextStyleTrans(config)
    #     task.test_sen_mse_add_mask_in_input_ablation_6()

    # new ablation
    elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_token_mean"]:
        task = LongTextStyleTrans(config)
        task.train_sen_mse_add_mask_in_input_ablation_token_mean()
    elif config.task_name in ["test_sen_mse_add_mask_in_input_ablation_token_mean"]:
        task = LongTextStyleTrans(config)
        task.test_sen_mse_add_mask_in_input_ablation_token_mean()
    elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_style_classifier"]:
        task = LongTextStyleTrans(config)
        task.train_sen_mse_add_mask_in_input_ablation_style_classifier()
    elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_sen_loss"]:
        task = LongTextStyleTrans(config)
        task.train_sen_mse_add_mask_in_input_ablation_sen_loss()
        

    # rebuttal ablation 不使用sen，直接采用hidden
    elif config.task_name in ["train_sen_mse_add_mask_in_input_ablation_sen"]:
        task = LongTextStyleTrans(config)
        task.train_sen_mse_add_mask_in_input_ablation_sen()
    
    elif config.task_name in ["test_sen_mse_add_mask_in_input_ablation_sen"]:
        task = LongTextStyleTrans(config)
        task.test_sen_mse_ablation_get_all_token()


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

    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    # # train_sc
    # args.task = "train_sen_mse_add_mask_in_input_ablation_sen"
    # args.model = "train_sen_mse_add_mask_in_input_ablation_sen_0816"
    # args.task = "train_transfer_add_cycle"

    # test
    # args.task = "test"
    # args.init = "model/encoder_bt_sen_lr_5e-5_1002/epoch-4-step-18640-loss-3.4503655433654785.pth"
    # args.pred = "predict/encoder_bt_sen_lr_5e-5_1002-s-18640.v2"



    # tran encoder
    # args.task = "train_encoder"
    #
    # train_transfer_bt_sen_and_cycle
    # args.task = "train_transfer_bt_sen_and_cycle"
    # train mean
    # args.task = "train_token_mean"
    # args.task = "train_concat_new_sen"
    # args.task = "test_concat_new_sen"
    # args.task = "train_concat_new_sen_add_inter"
    # args.task = "train_hidden_attention"
    # args.task = "test_reverse_style_attention"
    # args.init = "model/train_sen_generate-1019/epoch-15-step-29824-loss-1.3375431299209595.pth"
    # args.pred = "predict/train_sen_generate-1019-s-29824"
    # args.task = "train_hidden_attention_style"
    # args.task = "train_reverse_style_attention"
    # args.task = "train_content_distribution"
    # args.task = "train_content_distribution_on_sen"
    # args.task = "test_content_distribution_on_sen"
    # args.task = "train_sen_generate"
    # args.task = "test_sen_generate"
    # args.task = "train_toekn_mean_generate"
    # args.task = "test_toekn_mean_generate"
    # args.task = "train_toekn_mean_generate_without_sty"
    # args.task = "train_sen_generate_add_mask"
    # args.task = "train_fill"
    # args.task = "train_fill_base"
    # args.task = "train_sen_generate_with_kl_loss"
    # args.task = "train_sen_mse"
    # args.task = "train_sen_mse_add_mask"
    # args.task = "train_sen_mse_add_mask"
    # args.task = "train_sen_mse_add_mask_in_input"
    # args.task = "train_sen_mse_add_mask_in_input_ablation_5"
    # args.task = "train_sen_mse_add_mask_in_input_project_sen"
    # args.task = "train_sen_mse_add_mask_in_input_project_in_fix_len"
    # args.task = "train_sen_mse_add_mask_in_input_ablation_token_mean"
    # args.task = "train_sen_mse"
    
    # 主要任务 待确认
    # args.task = "train_sen_mse_add_mask_in_input"

    

    # test
    # args.task = "test_toekn_mean_generate"
    # args.task = "test_sen_mse"
    # args.init = "model/train_sen_mse_real-1102/epoch-39-step-74560-loss-1.158427119255066.pth"
    # args.pred = "predict/train_sen_mse_real-1102-s-74560_test"


    # fill mask
    # args.task = "fill_mask"
    # args.task = "test_fill"
    # args.task = "train_fill_LongLM"
    # args.init = "model/train_fill_easy_data-5e-5-1029/epoch-79-step-149120-loss-1.462022066116333.pth"
    # args.pred = "pred_fill/train_fill_easy_data-5e-5-1029-s-149120"
    # args.pred = "predict/train_fill_base-5e-5-1028-s-149120_test"
    # args.fill = "predict/mask_gen-1026-s-74560/test_sen_generate_add_mask.0"
    # args.fill = "predict/mask_gen-1026-s-74560/test_sen_generate_add_mask.1"


    main(args)