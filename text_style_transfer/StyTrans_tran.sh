CUDA_VISIBLE_DEVICES=0 python run_transfer_zh.py \
--task=ablation_sen_loss \
--model=ablation_sen_loss_1223 \
# --init=model_add_sen_type/cross_entropy+style_class_loss+distangle_loss+sen_order_loss_1115/epoch-69-step-130480-loss-1.0523488521575928.pth
# ablation
# task= ablation_sen_order_loss
# task= ablation_style_classifier_loss
# task= ablation_sen_loss
# main_task=train_sen_mse_add_mask_in_input_ablation_2