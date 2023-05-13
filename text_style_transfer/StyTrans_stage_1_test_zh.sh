CUDA_VISIBLE_DEVICES=1 python run_transfer_zh.py \
--task=test_sen_mse_add_mask_in_input \
--pred=pred_add_sentype/ablation_sen_loss_1223_e_4/stage_1 \
--init=model_add_sen_type/ablation_sen_loss_1223/epoch-4-step-9320-loss-6.048954486846924.pth