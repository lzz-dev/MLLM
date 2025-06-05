import torch
import torch.distributed as dist
import os

def main():
    dist.init_process_group(backend='nccl')  # 或 'gloo' 如果没用 GPU
    print(f"Hello from rank {dist.get_rank()} / {dist.get_world_size()}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
