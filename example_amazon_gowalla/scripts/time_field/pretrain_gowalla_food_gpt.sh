cd "$(dirname $0)"

python3 ../../pretrain_gpt.py \
-d gowalla_Food \
--data_type gowalla \
--task_type tf_trans \
--n_epoch 10 \
--cuda 0 \
--seed 0