run_args="--nproc_per_node 8 \
          --master_addr $MASTER_ADDR \
          --node_rank 0 \
          --master_port 39999" 
export PYTHONPATH=$PYTHONPATH:/root/BAGEL
torchrun $run_args /root/BAGEL/train/pretrain_unified_navit.py \
  --dataset_config_file /root/BAGEL/data/configs/example.yaml\
  --model_path /root/BAGEL-7B-MoT \
  --max_latent_size 64 \
  --resume-from /root/BAGEL-7B-MoT\
  --finetune_from_hf True \
  --finetune-from-ema True \
  --resume_model_only True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --visual_und False \
  --save_every 1000