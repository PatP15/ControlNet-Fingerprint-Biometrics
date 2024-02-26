import os
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


# Assuming the functions from your snippet
def load_model(model_path):
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(model_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler

def process(model, ddim_sampler, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        # Resize the input image to the desired resolution
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        # Directly use the preprocessed (Canny edges) image as control
        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results

# Assuming the rest of your definitions are provided earlier in the script

def process_directory(input_dir, output_dir, model_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    model, ddim_sampler = load_model(model_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + "_processed.png")  # Change output filename to indicate processing
            # Load image
            input_image = cv2.imread(input_path)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV format) to RGB

            # Process the image
            results = process(model, ddim_sampler, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)

            # Save the processed images
            for i, result in enumerate(results):
                result_path = os.path.splitext(output_path)[0] + f"_result_{i}.png"  # Generate unique filenames for each result
                cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))  # Convert RGB back to BGR for saving


# Example usage - adjust these parameters as needed
input_directory = './train_fingers/source/orient/test/00002332'
output_directory = './orient_test_outputs_smudge'
# model_checkpoint_path = './train_fingers/checkpoints/last.ckpt'
model_checkpoint_path = 'models/orient_model-epoch=07-val_loss=0.00.ckpt'
prompt = "a distorted single left index fingerprint by a roll from a Solid-state scanner surrounded by blank space"
a_prompt = "A single finger print from a scanner surrounded by blank space, only finger, clear image centered, photorealistic"
n_prompt = "multiple, mushed, low quality, cropped, worst quality, low quality"
num_samples = 3
image_resolution = 512  # Assuming square images for simplicity
ddim_steps = 50
guess_mode = False
strength = 1.0
scale = 7.5
seed = -1  # Use -1 for random seeds
eta = 0.0

process_directory(input_directory, output_directory, model_checkpoint_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)