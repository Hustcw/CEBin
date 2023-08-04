#!/bin/bash -e
accelerate launch --config_file config.yaml --machine_rank $RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT pretrain.py --data_dir $DATADIR --output_dir $AMLT_OUTPUT_DIR --job_name pretraining --gradient_accumulation_steps 4 --train_batchsize 16 --epochs 1 --ckpt_steps 4000 --eval_steps 160000 2>&1
