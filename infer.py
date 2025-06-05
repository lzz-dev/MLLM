import os
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO
import random
import numpy as np
from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
from inferencer import InterleaveInferencer
from IPython.display import display

## model initialization
model_path = "/root/.cache/modelscope/hub/models/ByteDance-Seed/BAGEL-7B-MoT"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

## model loading and multiple gpu preparing
max_mem_per_gpu = "40GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
print(device_map)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

# Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=True,
    dtype=torch.bfloat16,
    force_hooks=True,
    offload_folder="/tmp/offload"
)

model = model.eval()
print('Model loaded')

## infer preparing 
inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def img_generation(inferencer, save_path='/root/BAGEL/output',prompt="a young couple, Ghibli style"):
    inference_hyper=dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
    )

    # prompt = "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere."
    # prompt = "A poster showing that a film titled Good Things Happen will be released on May 5. Do not show anything not mentioned"
    # prompt = "A teacher standing in front of a blackboard is giving a lecture at the podium, and the blackboard is full of mathematical formulas. "
    # prompt = "a young couple, Ghibli style"
    # prompt = "Two people are walking, medium shot, low-angle. Anime style"
    prompt = prompt
    print(prompt)
    print('-' * 10)
    output_dict = inferencer(text=prompt, **inference_hyper)
    output_dict['image'].save(os.path.join(save_path,'img_generation.png'))
    display(output_dict['image'])

def img_generation_think(inferencer, save_path='/root/BAGEL/output', prompt="a couple walking, anime style"):
    inference_hyper=dict(
        max_think_token_n=1000,
        do_sample=False,
        # text_temperature=0.3,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
    )

    # prompt = 'a car made of small cars'
    # prompt = "A teacher standing in front of a blackboard is giving a lecture at the podium, and the blackboard is full of mathematical formulas."
    # prompt = "Two people are walking, medium shot, low-angle. Anime style"
    prompt = prompt
    print(prompt)
    print('-' * 10)
    output_dict = inferencer(text=prompt, think=True, **inference_hyper)
    print(output_dict['text'])
    output_dict['image'].save(os.path.join(save_path,'img_generation_think.png'))
    display(output_dict['image'])

def img_edit(inferencer, 
             save_path='/root/BAGEL/output', 
             input_file='/root/BAGEL/test_images/women.jpg', 
             input_prompt='She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes.'):
    inference_hyper=dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )

    image = Image.open(input_file)
    prompt = input_prompt

    display(image)
    print(prompt)
    print('-'*10)
    output_dict = inferencer(image=image, text=prompt, **inference_hyper)
    output_dict['image'].save(os.path.join(save_path,'img_edit.png'))
    display(output_dict['image'])


def img_edit_think(inferencer, 
                   save_path='/root/BAGEL/output', 
                   input_file='test_images/octupusy.jpg', 
                input_prompt='Could you display the sculpture that takes after this design?'):
    inference_hyper=dict(
        max_think_token_n=1000,
        do_sample=False,
        # text_temperature=0.3,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )

    image = Image.open(input_file)
    prompt = input_prompt

    display(image)
    print('-'*10)
    output_dict = inferencer(image=image, text=prompt, think=True, **inference_hyper)
    print(output_dict['text'])
    output_dict['image'].save(os.path.join(save_path,'img_edit_think.png'))
    display(output_dict['image'])


def understanding(inferencer):
    inference_hyper=dict(
        max_think_token_n=1000,
        do_sample=False,
        # text_temperature=0.3,
    )

    image = Image.open('test_images/meme.jpg')
    prompt = "Can someone explain what’s funny about this meme??"

    display(image)
    print(prompt)
    print('-'*10)
    output_dict = inferencer(image=image, text=prompt, understanding_output=True, **inference_hyper)
    print(output_dict['text'])

save_path = '/root/BAGEL/output'
img_generation(inferencer=inferencer, save_path=save_path, prompt='A movie poster featuring a film titled Good Things Happen, which will be released on May 5th. Anime style')
img_generation_think(inferencer=inferencer, prompt='A movie poster featuring a film titled Good Things Happen, which will be released on May 5th. Anime style')
# img_edit(inferencer=inferencer, save_path=save_path, input_file='/root/BAGEL/test_images/30.jpeg', input_prompt="remove the cake on the plate")
# img_edit_think(inferencer=inferencer, input_file='/root/BAGEL/test_images/30.jpeg', input_prompt="remove the cake on the plate")
# understanding(inferencer=inferencer)