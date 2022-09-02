cd "$(dirname $0)"

echo "process time_transfer on gowalla"
echo "---------------------------------------------------------------------------------------------"
echo "process gowalla_Entertainment for pretrain"

python3 ../../preprocess_dataset.py \
-d gowalla_Entertainment \
--data_type gowalla \
--task_type time_trans \
--mode pretrain \
--seed 0

echo "process gowalla_Entertainment for downstream"

python3 ../../preprocess_dataset.py \
-d gowalla_Entertainment \
--data_type gowalla \
--task_type time_trans \
--mode downstream \
--seed 0

echo "process gowalla_Nightlife for pretrain"

python3 ../../preprocess_dataset.py \
-d gowalla_Nightlife \
--data_type gowalla \
--task_type time_trans \
--mode pretrain \
--seed 0

echo "process gowalla_Nightlife for downstream"

python3 ../../preprocess_dataset.py \
-d gowalla_Nightlife \
--data_type gowalla \
--task_type time_trans \
--mode downstream \
--seed 0

echo "process gowalla_Outdoors for pretrain"

python3 ../../preprocess_dataset.py \
-d gowalla_Outdoors \
--data_type gowalla \
--task_type time_trans \
--mode pretrain \
--seed 0

echo "process gowalla_Outdoors for downstream"

python3 ../../preprocess_dataset.py \
-d gowalla_Outdoors \
--data_type gowalla \
--task_type time_trans \
--mode downstream \
--seed 0