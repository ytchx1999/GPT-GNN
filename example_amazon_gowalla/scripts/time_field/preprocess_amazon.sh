cd "$(dirname $0)"

echo "process time_transfer on amazon"
echo "---------------------------------------------------------------------------------------------"
echo "process amazon_acs for pretrain"

python3 ../../preprocess_dataset.py \
-d amazon_acs \
--data_type amazon \
--task_type tf_trans \
--mode pretrain \
--seed 0

echo "process amazon_fashion for downstream"

python3 ../../preprocess_dataset.py \
-d amazon_fashion \
--data_type amazon \
--task_type tf_trans \
--mode downstream \
--seed 0

echo "process amazon_luxury for downstream"

python3 ../../preprocess_dataset.py \
-d amazon_luxury \
--data_type amazon \
--task_type tf_trans \
--mode downstream \
--seed 0

echo "process amazon_beauty for downstream"

python3 ../../preprocess_dataset.py \
-d amazon_beauty \
--data_type amazon \
--task_type tf_trans \
--mode downstream \
--seed 0