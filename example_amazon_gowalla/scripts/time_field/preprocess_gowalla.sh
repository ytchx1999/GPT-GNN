cd "$(dirname $0)"

echo "process time_transfer on amazon"
echo "---------------------------------------------------------------------------------------------"
echo "process gowalla_Food for pretrain"

python3 ../../preprocess_dataset.py \
-d gowalla_Food \
--data_type gowalla \
--task_type tf_trans \
--mode pretrain \
--seed 0

echo "process gowalla_Nightlife for downstream"

python3 ../../preprocess_dataset.py \
-d gowalla_Nightlife \
--data_type gowalla \
--task_type tf_trans \
--mode downstream \
--seed 0

echo "process gowalla_Outdoors for downstream"

python3 ../../preprocess_dataset.py \
-d gowalla_Outdoors \
--data_type gowalla \
--task_type tf_trans \
--mode downstream \
--seed 0

echo "process gowalla_Entertainment for downstream"

python3 ../../preprocess_dataset.py \
-d gowalla_Entertainment \
--data_type gowalla \
--task_type tf_trans \
--mode downstream \
--seed 0