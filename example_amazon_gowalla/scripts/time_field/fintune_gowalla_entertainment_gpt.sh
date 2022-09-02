cd "$(dirname $0)"

python3 ../../finetune_gpt.py \
-d gowalla_Entertainment \
--data_type gowalla \
--task_type tf_trans \
--use_pretrain \
--n_epoch 40 \
--cuda 1 \
--seed 0