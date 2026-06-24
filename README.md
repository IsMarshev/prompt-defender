# prompt-defender
## Data Preprocess

Convert to jsonl 

### CSV:
```
python3 convert_to_jsonl.py \
  --input data/toxicchat/toxic-chat_annotation_train.csv \
  --output data/toxicchat/toxic-chat_annotation_train.jsonl
```
### Parquet:
```
python3 convert_to_jsonl.py \
  --input data/wildguard/wildguard_train.parquet \
  --output data/wildguard/wildguard_train.jsonl
```
### JSON
```
python3 convert_to_jsonl.py \
  --input data/aegis/train.json \
  --output data/aegis/aegis_train.jsonl
```


## Convert to dataset 

### BeaverTails
```
python prepare_data.py --source beavertails --input data/beavertails.jsonl \
    --output datasets/train.jsonl --split 0.9
```

### ToxicChat
```
python prepare_data.py --source toxicchat --input data/toxicchat.jsonl \
    --output datasets/train.jsonl --split 0.9
```

### WildGuard
```
python prepare_data.py --source wildguard --input data/wildguard/ \
    --output datasets/train.jsonl --split 0.9
```

### Aegis
```
python prepare_data.py --source aegis --input data/aegis.jsonl \
    --output datasets/train.jsonl
```

### Свой CSV/JSONL
```
python prepare_data.py --source custom --input data/my_data.csv \
    --text_col prompt --label_col safety --category_col category \
    --response_col response --output datasets/train.jsonl --split 0.9
```
