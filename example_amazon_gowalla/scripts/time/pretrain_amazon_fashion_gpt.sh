cd "$(dirname $0)"

python3 ../../pretrain_gpt.py \
-d amazon_fashion \
--data_type amazon \
--task_type time_trans \
--n_epoch 10 \
--cuda 0 \
--seed 0