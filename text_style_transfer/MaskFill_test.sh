CUDA_VISIBLE_DEVICES=3 python run_transfer_en.py \
--task=test_fill \
--pred=pred_fill/train_fill_LongLM-en-2-style-1222s-2320 \
--init=model/train_fill_LongLM-en-2-style-1222/epoch-3-step-2320-loss-0.08412856608629227.pth