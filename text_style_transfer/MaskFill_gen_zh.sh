CUDA_VISIBLE_DEVICES=1 python run_transfer_zh.py \
--task=fill_mask \
--pred=pred_add_sentype/ablation_sen_loss_1223_e_2/stage_2 \
--init=model/train_fill_LongLM-5e-5-1028/epoch-3-step-7456-loss-0.13943903148174286.pth \
--fill=pred_add_sentype/ablation_sen_loss_1223_e_2/stage_1/test_sen_mse_add_mask_in_input.1
