CUDA_VISIBLE_DEVICES=0 python run_transfer_en.py \
--task=fill_mask \
--pred=pred_add_sentype/en_ablation_style_classifier_loss_1129_e_79/stage_2 \
--init=model/train_fill_LongLM-en-2-style-1222/epoch-3-step-2320-loss-0.08412856608629227.pth \
--fill=pred_add_sentype/en_ablation_style_classifier_loss_1129_e_79/stage_1/test_transfer_stage_1.0
#--init=model/train_fill_LongLM-5e-5-1028/epoch-3-step-7456-loss-0.13943903148174286.pth