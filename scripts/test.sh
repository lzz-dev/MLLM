run_args="--nproc_per_node 8 \
          --master_addr $MASTER_ADDR \
          --node_rank 0 \
          --master_port 39999" 
export PYTHONPATH=$PYTHONPATH:/root/BAGEL
torchrun $run_args /root/BAGEL/train/test.py
  