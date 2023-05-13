CUDA_VISIBLE_DEVICES=1 python run_transfer_en.py \
--task=test_use_1_stage \
--pred=pred_add_sentype/en_train_use_1_stage_1224_e_23/stage_1 \
--init=model/en_train_use_1_stage_1224/epoch-23-step-13920-loss-2.577448606491089.pth
# test_use_1_stage