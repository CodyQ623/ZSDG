import argparse
import os
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import random
import glob
import json
from tqdm import tqdm
import re
import time
import shutil
import cv2
import pandas as pd
import sys

# Ensure models are imported from the current directory
from models import GeneratorUNet

# --- Add gen_fg.py to Python path for importing ---
# os.path.dirname(__file__) gives the directory of the current script (mydata/pix2pix)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to 'prepare/realiad' directory relative to this script
prepare_realiad_dir = os.path.join(script_dir, '..', '..', 'prepare', 'realiad')
sys.path.insert(0, os.path.abspath(prepare_realiad_dir))

try:
    from gen_fg import extract_foreground_connected as imported_extract_foreground
    print(f"Successfully imported extract_foreground_connected from: {os.path.join(prepare_realiad_dir, 'gen_fg.py')}")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import extract_foreground_connected from gen_fg.py: {e}")
    print(f"Attempted to import from: {os.path.join(prepare_realiad_dir, 'gen_fg.py')}")
    print("Please ensure gen_fg.py exists at that location and is importable.")
    # Define a dummy function to allow script to proceed for structural checks, but it will fail functionally
    def imported_extract_foreground(image_bgr, min_contour_area=100, closing_kernel_size=10):
        print("ERROR: imported_extract_foreground IS A DUMMY DUE TO IMPORT FAILURE!")
        # Return a black mask of the same size as input image
        if image_bgr is None: return None, None
        black_mask = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)
        return image_bgr, black_mask # Return original image and black mask
    # exit(1) # Or, more strictly, exit if the import fails.

# --- Constants ---
# TODO: Check the paths
ADABLDMROOT = "/home/cody/Projects/AnomDetect/anomaly_generation/langcode"
MVTEC_ENHNC_ROOT = os.path.join(ADABLDMROOT, "data/mvtec_reorg")
PROMPT_AND_METADATA_ROOT = os.path.join(ADABLDMROOT, "data/realiad/realiad_fake")

SAMPLES_PER_ITEM_TEST = 2000 
MASK_THRESHOLD = 0.5 
PROMPT_SIGMA = 0.00005 
MIN_MASK_PIXELS = 20 
MAX_RETRIES_PER_BASE = 5 
VECTOR_SCALE_FACTOR = 1.5 
MAX_BASE_VECTOR_ATTEMPTS = 100 
IMG_HEIGHT = 256 
IMG_WIDTH = 256 
CHANNELS = 1 
VECTOR_DIM = 6 

# --- Helper Functions ---

def read_prompts_from_label_json_for_item(item_name, base_prompt_path):
    label_json_path = os.path.join(base_prompt_path, item_name, "target", "label.json")
    item_prompts_list = []
    if not os.path.exists(label_json_path):
        print(f"Warning: label.json not found for item '{item_name}' at {label_json_path}")
        return item_prompts_list
    try:
        with open(label_json_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "prompt" in data and isinstance(data["prompt"], list) and len(data["prompt"]) == 4:
                        item_prompts_list.append(np.array(data["prompt"], dtype=np.float32))
                except json.JSONDecodeError:
                    pass # Ignore lines that are not valid JSON
    except Exception as e:
        print(f"Error reading {label_json_path} for item {item_name}: {e}")
    return item_prompts_list

def save_test_prompt_json(file_data, item_output_dir):
    json_path = os.path.join(item_output_dir, "prompt.json")
    try:
        with open(json_path, 'w') as f:
            for entry in file_data:
                prompt_str = json.dumps(entry["final_prompt"])
                json_line = {
                    "mask": entry["mask_path"],
                    "source": entry["source_path"],
                    "semap": entry["semap_path"],
                    "fg": entry.get("fg_path", ""),
                    "prompt": prompt_str
                }
                f.write(json.dumps(json_line) + '\n')
        print(f"Saved test prompts to {json_path}")
    except Exception as e:
        print(f"Error saving test prompt JSON to {json_path}: {e}")

def generate_perturbed_vector(base_vector, sigma=PROMPT_SIGMA):
    new_vector = np.array(base_vector[:4], dtype=np.float32) # Start with 4D attribute part
    attr_base = base_vector[:4]
    non_zero_indices_attr = np.where(attr_base > 1e-5)[0]
    if len(non_zero_indices_attr) > 0:
        for idx in non_zero_indices_attr:
            base_attr_value = attr_base[idx]
            new_attr_value = np.random.normal(loc=base_attr_value, scale=sigma * base_attr_value if base_attr_value > 1e-5 else sigma) 
            new_vector[idx] = np.clip(new_attr_value, 0, 1)
    
    # Add/perturb positional components to make it 6D
    pos_x = base_vector[4] if len(base_vector) == 6 else random.uniform(0.2, 0.8)
    pos_y = base_vector[5] if len(base_vector) == 6 else random.uniform(0.2, 0.8)
    
    perturbed_pos_x = np.clip(np.random.normal(loc=pos_x, scale=sigma), 0, 1)
    perturbed_pos_y = np.clip(np.random.normal(loc=pos_y, scale=sigma), 0, 1)
        
    return np.concatenate([new_vector, [perturbed_pos_x, perturbed_pos_y]]).astype(np.float32)


def run_test_mode(opt):
    print("Running in TEST mode...")

    if not os.path.isdir(MVTEC_ENHNC_ROOT):
        print(f"Error: MVTEC_ENHNC_ROOT '{MVTEC_ENHNC_ROOT}' not found. Exiting.")
        exit()
    ITEMS = [d for d in os.listdir(MVTEC_ENHNC_ROOT) if os.path.isdir(os.path.join(MVTEC_ENHNC_ROOT, d))]
    print(f"Found {len(ITEMS)} items in {MVTEC_ENHNC_ROOT}: {ITEMS}")

    if not opt.generator_path or not os.path.isfile(opt.generator_path):
         print(f"Error: Generator path '{opt.generator_path}' not provided or not found.")
         exit()
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if cuda else "cpu")
    print(f"Using device: {device}")

    print("Loading Generator...")
    generator = GeneratorUNet(in_channels=opt.channels, out_channels=opt.channels, vector_dim=opt.vector_dim)
    try:
        generator.load_state_dict(torch.load(opt.generator_path, map_location=device))
        generator.to(device)
        generator.eval()
        print(f"Loaded generator weights from {opt.generator_path}")
    except Exception as e:
        print(f"Error loading generator weights: {e}")
        exit()

    base_prompts_all_items = {}
    for item_name in ITEMS:
        prompts_for_item = read_prompts_from_label_json_for_item(item_name, PROMPT_AND_METADATA_ROOT)
        if prompts_for_item:
            base_prompts_all_items[item_name] = prompts_for_item
    
    if not base_prompts_all_items:
        print("Error: Failed to read any prompt vectors from label.json files. Exiting.")
        exit()

    transform_input_mask_for_generator = transforms.Compose([
        transforms.Resize((opt.img_height, opt.img_width), interpolation=Image.NEAREST),
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    transform_save_source = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((opt.img_height, opt.img_width), interpolation=Image.BICUBIC),
    ])

    for item in ITEMS:
        print(f"\nProcessing item: {item}")

        item_base_prompt_pool_4d = base_prompts_all_items.get(item, [])
        if not item_base_prompt_pool_4d:
            print(f"Warning: No prompt vectors found for item '{item}'. Skipping.")
            continue
        
        item_base_prompt_pool_6d = [generate_perturbed_vector(p_4d, 0) for p_4d in item_base_prompt_pool_4d] # Add initial random positions

        item_source_base_dir = os.path.join(MVTEC_ENHNC_ROOT, item)
        if not os.path.isdir(item_source_base_dir): continue
        
        type_dirs = [d for d in os.listdir(item_source_base_dir) if os.path.isdir(os.path.join(item_source_base_dir, d))]
        if not type_dirs: continue
        
        first_type_name = type_dirs[0]
        normal_source_dir = os.path.join(item_source_base_dir, "source")

        if not os.path.isdir(normal_source_dir): continue
        
        item_source_image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']:
            item_source_image_paths.extend(glob.glob(os.path.join(normal_source_dir, ext)))
        
        if not item_source_image_paths: continue
        print(f"Found {len(item_source_image_paths)} source images for {item} from {normal_source_dir}.")
            
        item_output_dir = os.path.join(opt.output_dir, item)
        os.makedirs(os.path.join(item_output_dir, "mask"), exist_ok=True)
        os.makedirs(os.path.join(item_output_dir, "source"), exist_ok=True)
        os.makedirs(os.path.join(item_output_dir, "semap"), exist_ok=True)
        os.makedirs(os.path.join(item_output_dir, "fg"), exist_ok=True)

        item_prompts_for_json = []
        generated_count = 0
        pbar = tqdm(total=SAMPLES_PER_ITEM_TEST, desc=f"Generating {item}", leave=False)
        
        generation_attempts_total = 0 

        while generated_count < SAMPLES_PER_ITEM_TEST and generation_attempts_total < SAMPLES_PER_ITEM_TEST * MAX_BASE_VECTOR_ATTEMPTS * 3 : 
            generation_attempts_total +=1
            if not item_source_image_paths: break 
            selected_source_path = random.choice(item_source_image_paths)
            
            foreground_mask_pil_original_res = None
            input_tensor_for_generator = None
            original_source_pil_for_saving = None

            try:
                source_cv2_img_original = cv2.imread(selected_source_path)
                if source_cv2_img_original is None: continue
                
                original_source_pil_for_saving = Image.open(selected_source_path)

                # Use imported foreground extractor
                # It returns (foreground_bgr, mask_single_channel_numpy_0_255)
                _, fg_mask_np_original_res = imported_extract_foreground(source_cv2_img_original) 
                if fg_mask_np_original_res is None: # Handle case where extraction might fail
                    print(f"Warning: Foreground extraction returned None for {selected_source_path}. Skipping.")
                    continue

                foreground_mask_pil_original_res = Image.fromarray(fg_mask_np_original_res, mode='L')

            except Exception as e:
                print(f"\nError processing source/extracting foreground for {selected_source_path}: {e}. Skipping.")
                time.sleep(0.01)
                continue
            
            if foreground_mask_pil_original_res is None: continue

            try:
                input_tensor_for_generator = transform_input_mask_for_generator(foreground_mask_pil_original_res).unsqueeze(0).to(device)
            except Exception as e:
                print(f"\nError transforming foreground for generator {selected_source_path}: {e}. Skipping.")
                time.sleep(0.01)
                continue

            sample_generated_successfully = False
            final_prompt_vec_np = None
            final_generated_mask_np = None # This will be the confined mask
            
            shuffled_base_prompts = random.sample(item_base_prompt_pool_6d, len(item_base_prompt_pool_6d))

            for base_vector_attempt_idx, base_prompt_vec_6d in enumerate(shuffled_base_prompts):
                if base_vector_attempt_idx >= MAX_BASE_VECTOR_ATTEMPTS: break 

                current_prompt_vec_np = generate_perturbed_vector(base_prompt_vec_6d, sigma=PROMPT_SIGMA)

                for retry in range(MAX_RETRIES_PER_BASE):
                    current_prompt_tensor = torch.from_numpy(current_prompt_vec_np.copy()).unsqueeze(0).to(device)
                    with torch.no_grad():
                        generated_mask_raw_from_gen = generator(input_tensor_for_generator, current_prompt_tensor)
                    
                    # Post-process: output is [-1, 1], scale to [0, 1]
                    generated_mask_np_unconfined = (generated_mask_raw_from_gen.squeeze(0).cpu() * 0.5 + 0.5).numpy().squeeze()

                    # --- BEGIN CONFINEMENT of generated mask ---
                    # Resize original foreground mask to generator's output dimensions (e.g., 256x256)
                    resized_fg_for_confinement_pil = foreground_mask_pil_original_res.resize(
                        (opt.img_width, opt.img_height), Image.NEAREST
                    )
                    # Convert to a binary numpy array (0.0 for background, 1.0 for foreground)
                    generator_input_fg_mask_np_binary = (np.array(resized_fg_for_confinement_pil) > 127).astype(np.float32) 
                    
                    # Confine the generated mask by multiplying with the (resized) input foreground mask
                    confined_generated_mask_np = generated_mask_np_unconfined * generator_input_fg_mask_np_binary
                    # --- END CONFINEMENT ---

                    # Check pixel count on the *confined* mask
                    pixel_count = np.sum(confined_generated_mask_np > MASK_THRESHOLD)

                    if pixel_count >= MIN_MASK_PIXELS:
                        final_prompt_vec_np = current_prompt_vec_np.copy()
                        final_generated_mask_np = confined_generated_mask_np # Use the confined mask
                        sample_generated_successfully = True
                        break 
                    else: # Scale prompt if not enough pixels
                        if retry < MAX_RETRIES_PER_BASE - 1:
                            attr_vec_to_scale = current_prompt_vec_np[:4]
                            non_zero_indices = np.where(attr_vec_to_scale > 1e-5)[0]
                            if len(non_zero_indices) > 0:
                                scaled_values = np.clip(attr_vec_to_scale[non_zero_indices] * VECTOR_SCALE_FACTOR, 0, 1)
                                attr_vec_to_scale[non_zero_indices] = scaled_values
                                current_prompt_vec_np[:4] = attr_vec_to_scale
                if sample_generated_successfully: break

            if not sample_generated_successfully:
                time.sleep(0.01) 
                continue 

            # Calculations based on the *final (confined)* generated mask
            binary_final_mask_np = (final_generated_mask_np > MASK_THRESHOLD).astype(np.float32)
            final_anomaly_pixels_in_mask = binary_final_mask_np > 0
            generated_anomaly_area = np.sum(final_anomaly_pixels_in_mask)
            
            total_image_area_at_gen_res = opt.img_height * opt.img_width 
            area_ratio = np.clip(generated_anomaly_area / total_image_area_at_gen_res if total_image_area_at_gen_res > 0 else 0, 0.0, 1.0)

            attr_vec_np = final_prompt_vec_np[:4] 
            semap_np = np.zeros((opt.img_height, opt.img_width, 4), dtype=np.float32)
            for i in range(4):
                channel = np.zeros_like(binary_final_mask_np)
                if attr_vec_np[i] > 1e-5: 
                    channel[final_anomaly_pixels_in_mask] = area_ratio
                semap_np[:, :, i] = channel
                        
            file_index_str = f"{generated_count:03d}"
            mask_rel_path = f"mask/{file_index_str}.png"
            source_img_ext = os.path.splitext(selected_source_path)[1] if os.path.splitext(selected_source_path)[1] else ".png"
            source_rel_path = f"source/{file_index_str}{source_img_ext}"
            semap_rel_path = f"semap/{file_index_str}.npy"
            fg_rel_path = f"fg/{file_index_str}.png" 
            
            mask_save_path = os.path.join(item_output_dir, mask_rel_path)
            semap_save_path = os.path.join(item_output_dir, semap_rel_path)
            source_save_path = os.path.join(item_output_dir, source_rel_path)
            fg_save_path = os.path.join(item_output_dir, fg_rel_path)

            # Save the final (confined) generated mask
            save_image(torch.from_numpy(final_generated_mask_np).unsqueeze(0), mask_save_path, normalize=False) # Already in 0-1 range
            np.save(semap_save_path, semap_np)
            
            try:
                resized_source_to_save = transform_save_source(original_source_pil_for_saving)
                resized_source_to_save.save(source_save_path)
            except Exception as e:
                print(f"Warning: Failed to save resized source image to {source_save_path}: {e}")

            try: 
                if foreground_mask_pil_original_res: # This is the mask at original source resolution
                    foreground_mask_pil_original_res.save(fg_save_path)
            except Exception as e:
                print(f"Warning: Failed to save foreground image to {fg_save_path}: {e}")

            item_prompts_for_json.append({
                "mask_path": mask_rel_path,
                "source_path": source_rel_path,
                "semap_path": semap_rel_path,
                "fg_path": fg_rel_path,
                "final_prompt": final_prompt_vec_np.tolist()
            })

            generated_count += 1
            pbar.update(1)

        pbar.close()
        if generated_count < SAMPLES_PER_ITEM_TEST:
             print(f"Warning: Only generated {generated_count}/{SAMPLES_PER_ITEM_TEST} samples for item {item}.")

        if item_prompts_for_json:
            save_test_prompt_json(item_prompts_for_json, item_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test anomalies using trained generator.")
    parser.add_argument("--generator_path", type=str, default="/home/cody/Projects/AnomDetect/anomaly_generation/langcode/pix2pix/saved_models/all_items_combined/generator_final.pth", help="Path to the trained generator .pth file")
    parser.add_argument("--output_dir", type=str, default="./output_realiad", help="Base directory for outputs")
    parser.add_argument("--img_height", type=int, default=IMG_HEIGHT, help="Image height for generator")
    parser.add_argument("--img_width", type=int, default=IMG_WIDTH, help="Image width for generator")
    parser.add_argument("--channels", type=int, default=CHANNELS, help="Image channels for generator")
    parser.add_argument("--vector_dim", type=int, default=VECTOR_DIM, help="Control vector dimension for generator")
    parser.add_argument("--adabldm_root", type=str, default=ADABLDMROOT, help="ADABLDM root directory")
    parser.add_argument("--samples_per_item", type=int, default=SAMPLES_PER_ITEM_TEST, help="Number of samples per item")

    opt = parser.parse_args()

    ADABLDMROOT = opt.adabldm_root
    MVTEC_ENHNC_ROOT = os.path.join(ADABLDMROOT, "data/mvtec_reorg")
    PROMPT_AND_METADATA_ROOT = os.path.join(ADABLDMROOT, "data/realiad/realiad_fake")

    IMG_HEIGHT = opt.img_height
    IMG_WIDTH = opt.img_width
    CHANNELS = opt.channels
    VECTOR_DIM = opt.vector_dim
    SAMPLES_PER_ITEM_TEST = opt.samples_per_item

    print("Configuration:")
    print(f"  Output Directory: {opt.output_dir}")
    print(f"  Generator Path: {opt.generator_path}")
    print(f"  ADABLDM Root: {ADABLDMROOT}")
    print(f"  Source Image Root (MVTEC_ENHNC_ROOT): {MVTEC_ENHNC_ROOT}")
    print(f"  Prompt & Metadata Root (PROMPT_AND_METADATA_ROOT): {PROMPT_AND_METADATA_ROOT}")
    print(f"  Samples per item: {SAMPLES_PER_ITEM_TEST}")

    run_test_mode(opt)
    print("\nProcessing finished.")