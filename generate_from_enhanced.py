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
from train_fingers.NIST302_dicts import id_to_finger

def load_model(model_path):
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(model_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler

def generate_varied_prompt(selected_method, fingertype):
    # Define the options for each component of the prompt
    
    scanner_types = ['Solid-state scanner', 'Optical scanner', 'Capacitive scanner', 'Touch-free scanner']
    
    selected_finger = id_to_finger[fingertype]
    # Randomly select one option from each list
    # selected_finger = random.choice(finger_types)
    # selected_method = random.choice(creation_methods)
    selected_scanner = random.choice(scanner_types)
    selected_method = id_to_finger
    # Combine the selections into a prompt
    prompt = f"a distorted single {selected_finger} by a {selected_method} from a {selected_scanner} surrounded by blank space with random smudging and different ink coloring"
    
    return prompt


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

        
        model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        
        model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        
        model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results


def process_directory(input_dir, output_dir, model_path, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    model, ddim_sampler = load_model(model_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Extract the fingerprint ID from the filename
            person_id = filename.split('_')[0]

            fingerprint_type = filename.split('_')[3].split('.')[0]

            # Define the output directory for this fingerprint ID
            fingerprint_output_dir = os.path.join(output_dir, person_id)


            if os.path.exists(fingerprint_output_dir) or person_id >= 15000:
                print(f"Skipping {person_id} as directory already exists.")
                continue
            
            if not os.path.exists(fingerprint_output_dir):
                print("making ", fingerprint_output_dir)
                os.makedirs(fingerprint_output_dir)
                
            input_path = os.path.join(input_dir, filename)
            # The output filename now indicates the ID and the processing step
            # output_filename = f"{fingerprint_id}_J_palm.png"
            # output_path = os.path.join(fingerprint_output_dir, filename)
            

            #we would like to make 15 variations of each finger so we will do 5 of each of these modalities
            creation_methods = ['roll', 'palm', 'slap']

            # loop through the creation methods
            # results = []
            for selected_method in creation_methods:
                prompt = generate_varied_prompt(selected_method, fingerprint_type)

                # pint
                # Load and process the image as before
                input_image = cv2.imread(input_path)
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                
                #we will do it in a batch of 5 each time
                results = process(model, ddim_sampler, input_image, prompt, a_prompt, n_prompt, 5, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
                
                
                #save the result
                for i, result in enumerate(results):
                    result_filename = f"{person_id}_{selected_method}_{fingerprint_type}_{i}.png"
                    result_path = os.path.join(fingerprint_output_dir, result_filename)
                    cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                

input_directory = './shared/trigo'
output_directory = './shared/control_variations'
# model_checkpoint_path = './train_fingers/checkpoints/last.ckpt'
model_checkpoint_path = 'models/enhanced_model-epoch=07-val_loss=0.00.ckpt'
a_prompt = "A single finger print from a scanner surrounded by blank space, only finger, clear image centered, photorealistic"
n_prompt = "multiple, mushed, low quality, cropped, worst quality"
num_samples = 1
image_resolution = 512  # Assuming square images for simplicity
ddim_steps = 30
guess_mode = False
strength = 1.0
scale = 7.5
seed = -1  # Use -1 for random seeds
eta = 0.0

process_directory(input_directory, output_directory, model_checkpoint_path, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)