CUDA_VISIBLE_DEVICES=1 python run_transfer_zh.py \
--task=test_one_stage_transfer \
--pred=pred_add_sentype/one_stage_transfer_1119_e_13/stage_1 \
--init=model_add_sen_type/one_stage_transfer_1119/epoch-13-step-26096-loss-2.577995538711548.pth