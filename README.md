# prompt-defender
# 1 GPU
python train.py --config config.yaml

# Multi-GPU (DDP)
python train.py --config config.yaml --devices 4 --strategy ddp

# Multi-node
python train.py --config config.yaml --devices 4 --strategy ddp --num_nodes 2

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/guard-epoch=0-step=500-val/loss=0.1234.ckpt
