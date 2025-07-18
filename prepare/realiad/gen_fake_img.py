import cv2
import glob
import os
import random
import numpy as np
import json
import pandas as pd
from math import *
from tqdm import tqdm
import sys
import gc
import torch # For cleanup, if CUDA is used elsewhere

# Configuration
# TODO: Please replace with actual REALIADROOT path
REALIAD_ROOT = "/home/cody/Projects/AnomDetect/anomaly_generation/langcode/data/realiad" # User specified
OUTPUT_BASE_PATH = os.path.join(REALIAD_ROOT, "realiad_fake")
PROMPT_EXCEL_PATH = os.path.join(REALIAD_ROOT, "realiad_reorg", "result.xlsx") # User specified
SCALEFACTOR = 2.0 # Added as per request: scales the anomaly before pasting

random.seed(1003)

def warpAnomaly(img, mask_3channel, useRotate=True, useResize=True):
    """
    Transform an anomaly image and its mask using rotation and resizing.
    Expects a 3-channel mask (e.g., loaded and converted, or already 3-channel).
    Returns transformed image, transformed 3-channel mask, and area ratio.
    """
    # Ensure input mask is binary 0 or 255 before processing, and 3-channel
    mask_binary_3channel = np.where(mask_3channel > 127, 255, 0).astype(np.uint8)

    # Find bounding box from the binary mask
    mask_single_channel_for_bbox = cv2.cvtColor(mask_binary_3channel, cv2.COLOR_BGR2GRAY)
    if np.sum(mask_single_channel_for_bbox) == 0: # Mask is empty
        return None, None, 0
    
    coords = cv2.findNonZero(mask_single_channel_for_bbox)
    if coords is None: 
        return None, None, 0
    x, y, w_bbox, h_bbox = cv2.boundingRect(coords)
    
    img_cropped = img[y : y + h_bbox, x : x + w_bbox]
    # Use the 3-channel binary mask for cropping, ensuring it's (H,W,3)
    mask_cropped_3channel = mask_binary_3channel[y : y + h_bbox, x : x + w_bbox] 

    if img_cropped.size == 0 or mask_cropped_3channel.size == 0:
        return None, None, 0

    height, width = img_cropped.shape[:2]
    # Calculate original mask area from a single channel of the 3-channel cropped mask
    original_mask_area = np.sum(cv2.cvtColor(mask_cropped_3channel, cv2.COLOR_BGR2GRAY) > 0)
    if original_mask_area == 0:
        return None, None, 0

    img_result = img_cropped.copy()
    mask_result_3channel = mask_cropped_3channel.copy() # This is (H,W,3)

    if useRotate:
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        
        cos_val = fabs(cos(radians(angle)))
        sin_val = fabs(sin(radians(angle)))
        
        new_width = int(width * cos_val + height * sin_val)
        new_height = int(width * sin_val + height * cos_val)

        M[0, 2] += (new_width - width) / 2
        M[1, 2] += (new_height - height) / 2
        
        img_result = cv2.warpAffine(img_result, M, (new_width, new_height), borderValue=(0, 0, 0))
        mask_result_3channel = cv2.warpAffine(mask_result_3channel, M, (new_width, new_height), borderValue=(0, 0, 0))
        height, width = new_height, new_width 

    if useResize:
        resize_factor = random.uniform(0.85, 1.15) 
        new_height_r = int(height * resize_factor)
        new_width_r = int(width * resize_factor)
        
        if new_height_r <= 0 or new_width_r <= 0:
            return None, None, 0
        img_result = cv2.resize(img_result, (new_width_r, new_height_r), interpolation=cv2.INTER_LINEAR)
        mask_result_3channel = cv2.resize(mask_result_3channel, (new_width_r, new_height_r), interpolation=cv2.INTER_NEAREST)

    # Final crop to content after transformations
    # Convert current mask_result_3channel to single channel for finding bounding box
    mask_single_channel_final_crop = cv2.cvtColor(mask_result_3channel, cv2.COLOR_BGR2GRAY)
    if np.sum(mask_single_channel_final_crop) == 0:
        return None, None, 0
        
    coords_final = cv2.findNonZero(mask_single_channel_final_crop)
    if coords_final is None: return None, None, 0
    x_f, y_f, w_f, h_f = cv2.boundingRect(coords_final)

    # Ensure width and height are positive before cropping
    if w_f <= 0 or h_f <= 0:
        return None, None, 0

    img_result = img_result[y_f : y_f + h_f, x_f : x_f + w_f]
    mask_result_3channel = mask_result_3channel[y_f : y_f + h_f, x_f : x_f + w_f]
    
    # Check if cropping resulted in empty arrays
    if img_result.size == 0 or mask_result_3channel.size == 0:
        return None, None, 0

    # Ensure mask is strictly binary 0 or 255 after all operations, AND REMAINS 3-CHANNEL
    # Convert to grayscale, threshold, then convert back to 3-channel BGR
    temp_mask_gray = cv2.cvtColor(mask_result_3channel, cv2.COLOR_BGR2GRAY)
    _, temp_mask_binary_single_channel = cv2.threshold(temp_mask_gray, 127, 255, cv2.THRESH_BINARY)
    mask_result_3channel = cv2.cvtColor(temp_mask_binary_single_channel, cv2.COLOR_GRAY2BGR)

    # Apply mask to image (make non-anomaly parts black)
    # Now, img_result (H,W,3) and mask_result_3channel (H,W,3) should have same dimensions.
    if img_result.shape != mask_result_3channel.shape:
        return None, None, 0
        
    img_result = cv2.bitwise_and(img_result, mask_result_3channel)
    
    # Calculate transformed_mask_area from the final masked 3-channel mask
    transformed_mask_area = np.sum(cv2.cvtColor(mask_result_3channel, cv2.COLOR_BGR2GRAY) > 0)
    
    # Modified: Calculate absolute ratio (anomaly area occupies the proportion of the entire image), always less than 1
    total_area = img_result.shape[0] * img_result.shape[1]
    area_ratio = transformed_mask_area / total_area if total_area > 0 else 0
    
    if img_result.size == 0 or mask_result_3channel.size == 0 or transformed_mask_area == 0:
        return None, None, 0
        
    return img_result, mask_result_3channel, area_ratio

def find_valid_placement_in_foreground(template_fg_mask_single_channel, warped_anomaly_img, warped_anomaly_mask_single_channel, max_tries=50, min_overlap_ratio=0.7):
    """
    Finds a valid top-left (h, w) for placing the warped anomaly such that
    it significantly overlaps with the template's foreground mask.
    All masks here are expected to be single-channel binary (0 or 255).
    Returns potentially resized anomaly and mask if they were too large initially.
    """
    H_t, W_t = template_fg_mask_single_channel.shape[:2]
    H_a, W_a = warped_anomaly_img.shape[:2]

    if H_a == 0 or W_a == 0: return None # Warped anomaly is empty

    # If anomaly is larger than template, try to resize it down.
    current_warped_anomaly_img = warped_anomaly_img
    current_warped_anomaly_mask_single_channel = warped_anomaly_mask_single_channel
    
    scale_factor_internal = 1.0
    if H_a > H_t : scale_factor_internal = min(scale_factor_internal, H_t / H_a)
    if W_a > W_t : scale_factor_internal = min(scale_factor_internal, W_t / W_a)
    
    if scale_factor_internal < 1.0:
        scale_factor_internal *= 0.95 # Ensure it's a bit smaller
        new_H_a_internal = int(H_a * scale_factor_internal)
        new_W_a_internal = int(W_a * scale_factor_internal)
        if new_H_a_internal <=0 or new_W_a_internal <=0: return None
        
        current_warped_anomaly_img = cv2.resize(warped_anomaly_img, (new_W_a_internal, new_H_a_internal), interpolation=cv2.INTER_LINEAR)
        current_warped_anomaly_mask_single_channel = cv2.resize(warped_anomaly_mask_single_channel, (new_W_a_internal, new_H_a_internal), interpolation=cv2.INTER_NEAREST)
        H_a, W_a = new_H_a_internal, new_W_a_internal # Update dimensions for placement logic
        if H_a == 0 or W_a == 0: return None


    possible_hs = list(range(H_t - H_a + 1))
    possible_ws = list(range(W_t - W_a + 1))

    if not possible_hs or not possible_ws:
        return None

    anomaly_pixel_area = np.sum(current_warped_anomaly_mask_single_channel > 0)
    if anomaly_pixel_area == 0: return None

    for _ in range(max_tries):
        h = random.choice(possible_hs)
        w = random.choice(possible_ws)

        roi_template_fg = template_fg_mask_single_channel[h : h + H_a, w : w + W_a]
        
        intersection = cv2.bitwise_and(roi_template_fg, current_warped_anomaly_mask_single_channel)
        intersection_area = np.sum(intersection > 0)
        
        if intersection_area / anomaly_pixel_area >= min_overlap_ratio:
            return h, w, current_warped_anomaly_img, current_warped_anomaly_mask_single_channel # Return potentially resized versions
            
    return None

def load_prompt_vectors_from_excel(excel_path, item_name_for_sheet):
    prompt_vectors = {}
    if not os.path.exists(excel_path):
        print(f"Warning: Excel file path does not exist: {excel_path}")
        return prompt_vectors
    try:
        xl = pd.ExcelFile(excel_path)
        target_sheet_name = None
        for sheet_name in xl.sheet_names:
            if sheet_name.lower() == item_name_for_sheet.lower():
                target_sheet_name = sheet_name
                break
        
        if target_sheet_name is None:
            print(f"Warning: Sheet named '{item_name_for_sheet}' not found in Excel file. Available sheets: {xl.sheet_names}")
            return prompt_vectors

        df = xl.parse(target_sheet_name, dtype=str)
        
        for _, row in df.iterrows():
            try:
                file_id_raw = row['File']
                file_id = file_id_raw.strip()
                
                if '.' in file_id:
                    file_id_base = file_id.split('.')[0]
                else:
                    file_id_base = file_id
                
                try:
                    vector = [
                        float(row['Value_1']), float(row['Value_2']),
                        float(row['Value_3']), float(row['Value_4'])
                    ]
                except ValueError:
                    # print(f"Warning: Cannot convert values: {row['Value_1']}, {row['Value_2']}, {row['Value_3']}, {row['Value_4']}")
                    continue
                
                prompt_vectors[file_id] = vector
                if file_id != file_id_base:
                    prompt_vectors[file_id_base] = vector
                if file_id_base.isdigit():
                    int_key = str(int(file_id_base))
                    prompt_vectors[int_key] = vector
                
            except KeyError as e:
                # print(f"Warning: Excel worksheet '{target_sheet_name}' missing expected columns: {e}")
                continue
            except Exception as e:
                # print(f"Warning: Error processing Excel row: {e}, row data: {row}")
                continue
        
        # print(f"Loaded {len(prompt_vectors)} prompt vectors from worksheet '{target_sheet_name}'")
        # print(f"Available key examples: {list(prompt_vectors.keys())[:10]}")
        
    except Exception as e:
        print(f"Error reading Excel file '{excel_path}': {e}")
    
    return prompt_vectors

def modify_prompt_vector_with_area_ratio(original_vector, area_ratio):
    if original_vector is None: return [0.0,0.0,0.0,0.0] # Default if no original
    modified_vector = []
    for value in original_vector:
        if abs(value) > 1e-9:
            modified_vector.append(value * area_ratio)
        else:
            modified_vector.append(0.0)
    return modified_vector

def extract_image_number(filename):
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    digits = ''.join(filter(str.isdigit, name_without_ext))
    return digits if digits else name_without_ext # Fallback to name if no digits

def generate_synthetic_defects(realiad_root, item, defect_type="ng", num_images=200):
    print(f"Generating synthetic defect images for {item}/{defect_type} (foreground restriction, paste scale factor: {SCALEFACTOR})...")

    # Input paths
    real_anomaly_dir = os.path.join(realiad_root, "realiad_reorg", item, "test", defect_type)
    mask_anomaly_dir = os.path.join(realiad_root, "realiad_reorg", item, "ground_truth", defect_type)
    
    template_source_dir = os.path.join(realiad_root, "realiad_fg", item, "source")
    template_fg_mask_dir = os.path.join(realiad_root, "realiad_fg", item, "mask") # Foreground restriction masks

    # Output paths
    save_item_path = os.path.join(OUTPUT_BASE_PATH, item)
    save_target_dir = os.path.join(save_item_path, "target") # Synthetic defective images
    save_mask_dir = os.path.join(save_item_path, "mask")     # Masks for synthetic defects
    save_source_dir = os.path.join(save_item_path, "source") # Original templates used

    os.makedirs(save_target_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)
    os.makedirs(save_source_dir, exist_ok=True)

    anomaly_img_paths = sorted(glob.glob(os.path.join(real_anomaly_dir, "*.png")) + \
                               glob.glob(os.path.join(real_anomaly_dir, "*.jpg")) + \
                               glob.glob(os.path.join(real_anomaly_dir, "*.bmp")))
    
    template_img_paths = sorted(glob.glob(os.path.join(template_source_dir, "*.png")) + \
                                glob.glob(os.path.join(template_source_dir, "*.jpg")) + \
                                glob.glob(os.path.join(template_source_dir, "*.bmp")))

    if not anomaly_img_paths: print(f"Error: No anomaly images found in {real_anomaly_dir}."); return
    if not template_img_paths: print(f"Error: No template images found in {template_source_dir}."); return

    item_prompt_vectors = load_prompt_vectors_from_excel(PROMPT_EXCEL_PATH, item)
    if not item_prompt_vectors:
        print(f"Warning: Failed to load any prompt vectors for item '{item}'. Will use default vectors.")

    label_data = []
    count = 0
    
    pbar = tqdm(total=num_images, desc=f"Generating {item}/{defect_type}")

    while count < num_images:
        if not anomaly_img_paths or not template_img_paths: break 

        template_path = random.choice(template_img_paths)
        anomaly_path = random.choice(anomaly_img_paths)
        
        base_anomaly_name = os.path.basename(anomaly_path)
        name_anomaly, ext_anomaly = os.path.splitext(base_anomaly_name)

        anomaly_gt_mask_path_variants = [
            os.path.join(mask_anomaly_dir, base_anomaly_name),
            os.path.join(mask_anomaly_dir, f"{name_anomaly}_mask{ext_anomaly}"),
            os.path.join(mask_anomaly_dir, f"{name_anomaly}.png"),
        ]
        anomaly_gt_mask_path = None
        for variant in anomaly_gt_mask_path_variants:
            if os.path.exists(variant):
                anomaly_gt_mask_path = variant
                break
        
        template_fg_mask_path = os.path.join(template_fg_mask_dir, os.path.basename(template_path))

        if not anomaly_gt_mask_path: continue
        if not os.path.exists(template_fg_mask_path): continue

        try:
            template_img = cv2.imread(template_path)
            template_fg_mask_img = cv2.imread(template_fg_mask_path, cv2.IMREAD_GRAYSCALE) 
            
            anomaly_img = cv2.imread(anomaly_path)
            anomaly_gt_mask_img = cv2.imread(anomaly_gt_mask_path, cv2.IMREAD_GRAYSCALE)

            if template_img is None or template_fg_mask_img is None or \
               anomaly_img is None or anomaly_gt_mask_img is None:
                continue
            
            _, template_fg_mask_img = cv2.threshold(template_fg_mask_img, 127, 255, cv2.THRESH_BINARY)
            _, anomaly_gt_mask_img = cv2.threshold(anomaly_gt_mask_img, 127, 255, cv2.THRESH_BINARY)

        except Exception:
            continue
        
        anomaly_gt_mask_3channel = cv2.cvtColor(anomaly_gt_mask_img, cv2.COLOR_GRAY2BGR)
        warped_anomaly_orig, warped_anomaly_mask_3channel, area_ratio = warpAnomaly(anomaly_img, anomaly_gt_mask_3channel)

        if warped_anomaly_orig is None or warped_anomaly_orig.size == 0 or area_ratio <= 0:
            continue
        
        warped_anomaly_mask_single_channel_orig = cv2.cvtColor(warped_anomaly_mask_3channel, cv2.COLOR_BGR2GRAY)

        placement_result = find_valid_placement_in_foreground(
            template_fg_mask_img, 
            warped_anomaly_orig, 
            warped_anomaly_mask_single_channel_orig
        )

        if placement_result is None:
            continue
        
        place_h_orig, place_w_orig, current_warped_anomaly, current_warped_anomaly_mask_single = placement_result

        # --- START OF MODIFICATION FOR SCALING ANOMALY BEFORE PASTING ---
        H_a, W_a = current_warped_anomaly.shape[:2]
        if H_a == 0 or W_a == 0:
            continue

        # 1. Scale the anomaly and its mask by SCALEFACTOR
        new_H_a_scaled = int(round(H_a * SCALEFACTOR))
        new_W_a_scaled = int(round(W_a * SCALEFACTOR))

        if new_H_a_scaled <= 0 or new_W_a_scaled <= 0:
            continue

        scaled_pasted_anomaly = cv2.resize(current_warped_anomaly, (new_W_a_scaled, new_H_a_scaled), interpolation=cv2.INTER_LINEAR)
        scaled_pasted_mask = cv2.resize(current_warped_anomaly_mask_single, (new_W_a_scaled, new_H_a_scaled), interpolation=cv2.INTER_NEAREST)
        _, scaled_pasted_mask = cv2.threshold(scaled_pasted_mask, 127, 255, cv2.THRESH_BINARY) # Ensure binary

        # 2. Calculate new top-left position for the scaled anomaly to maintain center
        center_h_on_template = place_h_orig + H_a / 2.0
        center_w_on_template = place_w_orig + W_a / 2.0

        place_h_scaled = int(round(center_h_on_template - new_H_a_scaled / 2.0))
        place_w_scaled = int(round(center_w_on_template - new_W_a_scaled / 2.0))

        # 3. Composite the scaled anomaly onto the template with clipping
        target_img = template_img.copy()
        H_t, W_t = target_img.shape[:2]

        # Determine the actual portion of scaled anomaly and target to use (clipping)
        paste_y_start_target = max(0, place_h_scaled)
        paste_x_start_target = max(0, place_w_scaled)
        paste_y_end_target = min(H_t, place_h_scaled + new_H_a_scaled)
        paste_x_end_target = min(W_t, place_w_scaled + new_W_a_scaled)
        
        eff_h_paste = paste_y_end_target - paste_y_start_target
        eff_w_paste = paste_x_end_target - paste_x_start_target

        if eff_h_paste <= 0 or eff_w_paste <= 0:
            continue 

        crop_y_start_anomaly = max(0, -place_h_scaled)
        crop_x_start_anomaly = max(0, -place_w_scaled)
        crop_y_end_anomaly = crop_y_start_anomaly + eff_h_paste
        crop_x_end_anomaly = crop_x_start_anomaly + eff_w_paste
        
        anomaly_patch_to_paste = scaled_pasted_anomaly[crop_y_start_anomaly:crop_y_end_anomaly, crop_x_start_anomaly:crop_x_end_anomaly]
        mask_patch_to_paste = scaled_pasted_mask[crop_y_start_anomaly:crop_y_end_anomaly, crop_x_start_anomaly:crop_x_end_anomaly]

        if anomaly_patch_to_paste.size == 0 or mask_patch_to_paste.size == 0:
            continue

        roi_on_target = target_img[paste_y_start_target:paste_y_end_target, paste_x_start_target:paste_x_end_target]
        
        if roi_on_target.shape[:2] != anomaly_patch_to_paste.shape[:2] or \
           roi_on_target.shape[:2] != mask_patch_to_paste.shape[:2]:
            continue # Shape mismatch after clipping

        active_pixels_bool_mask_scaled = mask_patch_to_paste > 0
        np.copyto(roi_on_target, anomaly_patch_to_paste, where=active_pixels_bool_mask_scaled[:,:,np.newaxis])

        # 4. Create final synthetic defect mask (single channel) using the scaled and placed mask
        synthetic_defect_mask = np.zeros_like(template_fg_mask_img) 
        synthetic_defect_mask[paste_y_start_target:paste_y_end_target, paste_x_start_target:paste_x_end_target] = \
            np.where(mask_patch_to_paste > 0, 255, 0)
        # --- END OF MODIFICATION FOR SCALING ---

        original_anomaly_filenumber = extract_image_number(anomaly_path)
        potential_keys = [
            original_anomaly_filenumber,
            f"{original_anomaly_filenumber}.0",
        ]
        if original_anomaly_filenumber.isdigit():
            potential_keys.append(str(int(original_anomaly_filenumber)))
        
        original_prompt_vector = None
        for key_try in potential_keys:
            if key_try in item_prompt_vectors:
                original_prompt_vector = item_prompt_vectors[key_try]
                break
        
        if original_prompt_vector is None:
            pbar.set_postfix_str(f"Prompt not found for {original_anomaly_filenumber}, skipping")
            continue # Or assign a default prompt vector if preferred
            # original_prompt_vector = [0.0, 0.0, 0.0, 0.0] # Example default
        
        scaled_area_ratio = min(area_ratio * (SCALEFACTOR * SCALEFACTOR), 1.0)  
        modified_prompt = modify_prompt_vector_with_area_ratio(original_prompt_vector, scaled_area_ratio)

        out_filename = f"{count:03d}.png"
        cv2.imwrite(os.path.join(save_target_dir, out_filename), target_img)
        cv2.imwrite(os.path.join(save_mask_dir, out_filename), synthetic_defect_mask)
        cv2.imwrite(os.path.join(save_source_dir, out_filename), template_img)

        label_data.append({"image": out_filename, "prompt": modified_prompt})
        
        count += 1
        pbar.update(1)
        pbar.set_postfix_str("")


    pbar.close()

    if count < num_images:
        print(f"Warning: Only generated {count} images, less than requested {num_images}.")

    label_json_path = os.path.join(save_target_dir, "label.json")
    with open(label_json_path, 'w') as f:
        for entry in label_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Generated {count} synthetic defect images for {item}/{defect_type}.")


if __name__ == "__main__":
    if not os.path.isdir(REALIAD_ROOT):
        print(f"Warning: The specified REALIAD_ROOT '{REALIAD_ROOT}' does not exist or is not a directory.")
        sys.exit(1)

    reorg_path = os.path.join(REALIAD_ROOT, "realiad_reorg")
    if not os.path.isdir(reorg_path):
        print(f"Warning: The specified reorg path '{reorg_path}' does not exist or is not a directory.")
        sys.exit(1)

    items = [d for d in os.listdir(reorg_path) if os.path.isdir(os.path.join(reorg_path, d))]
    
    if not items:
        print("Warning: No items found in the REALIAD reorg directory. Please check the directory structure.")
        sys.exit(1)

    for item_idx, item_name in enumerate(items):
        print(f"\nProcessing item {item_idx + 1}/{len(items)}: {item_name}")

        required_paths_exist = True
        paths_to_check = {
            "Real Anomaly Images": os.path.join(REALIAD_ROOT, "realiad_reorg", item_name, "test", "ng"),
            "Real Anomaly Masks": os.path.join(REALIAD_ROOT, "realiad_reorg", item_name, "ground_truth", "ng"),
            "Template Images (FG)": os.path.join(REALIAD_ROOT, "realiad_fg", item_name, "source"),
            "Template FG Masks": os.path.join(REALIAD_ROOT, "realiad_fg", item_name, "mask")
        }
        for desc, path in paths_to_check.items():
            if not os.path.isdir(path):
                print(f"Warning: Required path '{desc}' does not exist: {path}")
                required_paths_exist = False
                break
        
        if not required_paths_exist:
            continue
        
        defect_type_to_process = "ng" 
        
        try:
            generate_synthetic_defects(REALIAD_ROOT, item_name, defect_type_to_process, num_images=2000)
            print(f"Successfully processed item {item_name}/{defect_type_to_process}.")
        except Exception as e:
            print(f"Error processing item {item_name}/{defect_type_to_process}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Finished processing item {item_name}/{defect_type_to_process}.")