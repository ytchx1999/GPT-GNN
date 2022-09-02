# Code for GPT-GNN on amazon and gowalla datasets

## Environment

```bash
python==3.8
torch==1.10.1+cu102
torch-geometric==2.0.4
tqdm
```

## Data

```bash
./base_dir/
|-- processed_data
   |-- amazon
   |--|-- field_trans
   |--|--|-- ...
   |--|-- time_trans
   |--|--|-- ...
   |-- gowalla
   |--|-- field_trans
   |--|--|-- ...
   |--|-- time_trans
   |--|--|-- ml_gowalla_Entertainment_pretrain.csv
   |--|--|-- ml_gowalla_Entertainment_pretrain_node.npy
   |--|--|-- ml_gowalla_Entertainment_downstream.csv
   |--|--|-- ml_gowalla_Entertainment_downstream_node.npy
   |--|--|-- ...
```

## Run

### Time transfer

```bash
cd scripts/time/
# amazon
# preprocess
bash preprocess_amazon.sh
# amazon_beauty
bash pretrain_amazon_beauty_gpt.sh
bash finetune_amazon_beauty_gpt.sh
# amazon_fashion
bash pretrain_amazon_fashion_gpt.sh
bash finetune_amazon_fashion_gpt.sh
# amazon_luxury
bash pretrain_amazon_luxury_gpt.sh
bash finetune_amazon_luxury_gpt.sh

# gowalla
# preprocess
bash preprocess_gowalla.sh
# gowalla_Entertainment
bash pretrain_gowalla_entertainment_gpt.sh
bash finetune_gowalla_entertainment_gpt.sh
# gowalla_Outdoors
bash pretrain_gowalla_outdoors_gpt.sh
bash finetune_gowalla_outdoors_gpt.sh
# gowalla_Nightlife
bash pretrain_gowalla_nightlife_gpt.sh
bash finetune_gowalla_nightlife_gpt.sh
```

<!-- # pretrain + amazon_beauty + time transfer: model VGAE
bash pretrain_amazon_beauty_time_vgae.sh
# downstream + amazon_beauty + time transfer: model VGAE
bash downstream_amazon_beauty_time_vgae.sh -->

### Field transfer

```bash
cd scripts/field/
# amazon
# preprocess
bash preprocess_amazon.sh
bash pretrain_amazon_acs_field_gpt.sh
bash downstream_amazon_beauty_field_gpt.sh
bash downstream_amazon_fashion_field_gpt.sh
bash downstream_amazon_luxury_field_gpt.sh

# gowalla
# preprocess
bash preprocess_gowalla.sh
bash pretrain_gowalla_food_gpt.sh
bash downstream_gowalla_entertainment_gpt.sh
bash downstream_gowalla_outdoors_gpt.sh
bash downstream_gowalla_nightlife_gpt.sh
```

### Time+Field transfer

```bash
cd scripts/time_field/
# amazon
# preprocess
bash preprocess_amazon.sh
bash pretrain_amazon_acs_field_gpt.sh
bash downstream_amazon_beauty_field_gpt.sh
bash downstream_amazon_fashion_field_gpt.sh
bash downstream_amazon_luxury_field_gpt.sh

# gowalla
# preprocess
bash preprocess_gowalla.sh
bash pretrain_gowalla_food_gpt.sh
bash downstream_gowalla_entertainment_gpt.sh
bash downstream_gowalla_outdoors_gpt.sh
bash downstream_gowalla_nightlife_gpt.sh
```
