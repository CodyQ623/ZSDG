import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import glob

# Removed DATAROOTPATH and MVTECPATH from config as paths are now explicit

def generate_semantic_maps_for_item(base_item_path, item_name):
    """
    Generate semantic maps for a given item from the pcb_dataset_fake_normed_black structure.
    
    Args:
        base_item_path: The root path to the item's directory 
                        (e.g., /path/to/pcb_dataset_fake_normed_black/item_name)
        item_name: The name of the item (used for print statements)
    """
    print(f"Generating semantic maps for item: {item_name}")
    
    # Define input and output paths based on the new structure
    json_path = os.path.join(base_item_path, "target", "label.json")
    mask_dir = os.path.join(base_item_path, "mask")
    # source_dir = os.path.join(base_item_path, "source") # Not directly used for semap generation logic
    # target_dir = os.path.join(base_item_path, "target") # Not directly used for semap generation logic
    output_semap_dir = os.path.join(base_item_path, "semap")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_semap_dir, exist_ok=True)
    
    # Load prompt vectors from the JSON file
    prompt_vectors = {}
    if not os.path.exists(json_path):
        print(f"Error: Label file {json_path} not found for item {item_name}!")
        return False
    
    with open(json_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                image_name_from_json = data["image"] # e.g., "000.png"
                prompt = data["prompt"]
                prompt_vectors[image_name_from_json] = prompt
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON line in {json_path}: {line.strip()}")
            except KeyError:
                print(f"Warning: Missing required keys ('image', 'prompt') in JSON line in {json_path}: {line.strip()}")
    
    if not prompt_vectors:
        print(f"Warning: No prompt vectors loaded from {json_path} for item {item_name}.")
        # Decide if to continue or return False. For now, let's try to process masks.
        # return False 
    else:
        print(f"Loaded {len(prompt_vectors)} prompt vectors from {json_path} for item {item_name}")
    
    # Find all mask files (e.g., 000.png, 001.png)
    # The previous script looked for "*_mask.png", then "*.png".
    # For the new dataset, masks likely have simple names like "000.png".
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    if not mask_files:
        mask_files = glob.glob(os.path.join(mask_dir, "*.jpg")) # Add other extensions if needed
    if not mask_files:
        mask_files = glob.glob(os.path.join(mask_dir, "*.bmp"))

    if not mask_files:
        print(f"Error: No mask files found in {mask_dir} for item {item_name}")
        return False
    
    print(f"Found {len(mask_files)} mask files in {mask_dir}")
    
    # Prepare data for prompt.json (which will be saved in the item's root)
    prompt_json_data_output = []
    
    success_count = 0
    for mask_path in tqdm(mask_files, desc=f"Processing masks for {item_name}"):
        # Get base image name (e.g., "000.png" from "/path/to/mask/000.png")
        image_name = os.path.basename(mask_path) 
        
        if image_name not in prompt_vectors:
            print(f"Warning: No prompt vector found for mask '{image_name}' in item '{item_name}', skipping. (Checked against keys like '{image_name}')")
            continue
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load mask file {mask_path}, skipping")
            continue
        
        height, width = mask.shape
        prompt_vector = prompt_vectors[image_name]
        
        semantic_map = np.zeros((height, width, 4), dtype=np.float32)
        
        for i in range(4): 
            semantic_map[:, :, i] = np.where(mask > 0, prompt_vector[i], 0)
        
        npy_filename = image_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.bmp', '.npy')
        output_npy_path = os.path.join(output_semap_dir, npy_filename)
        np.save(output_npy_path, semantic_map)
        success_count += 1
        
        # Paths in prompt.json are relative to the item directory
        prompt_json_data_output.append({
            "source": f"source/{image_name}",  # Assuming source has same base name
            "target": f"target/{image_name}",  # Assuming target has same base name
            "semap": f"semap/{npy_filename}"
        })
        
        # Visualization of the semantic map
        vis_map = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(3): 
            # Original scaling: int(255 * (prompt_vector[i] / 2)).
            # This assumes prompt_vector values are roughly in 0-2 range for good visualization.
            # If prompt_vector[i] is typically 0-1, then `* 255` is more direct.
            # Keeping original logic for consistency unless specified.
            scaled_value = int(np.clip(prompt_vector[i] / 2.0 * 255, 0, 255))
            vis_map[:, :, i] = np.where(mask > 0, scaled_value, 0)
        
        vis_output_filename = image_name.replace('.png', '_vis.png').replace('.jpg', '_vis.jpg').replace('.bmp', '_vis.bmp')
        vis_output_path = os.path.join(output_semap_dir, vis_output_filename)
        cv2.imwrite(vis_output_path, vis_map)
    
    # Save prompt.json file in the item's base directory
    output_prompt_json_path = os.path.join(base_item_path, "prompt.json")
    with open(output_prompt_json_path, 'w') as f:
        for entry in prompt_json_data_output:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Successfully generated {success_count} semantic maps for item {item_name}")
    print(f"Created prompt.json with {len(prompt_json_data_output)} entries at {output_prompt_json_path}")
    return True

if __name__ == "__main__":
    # TODO: Please replace with actual REALIADROOT path
    ROOT_INPUT_DIR = "/home/cody/Projects/AnomDetect/anomaly_generation/langcode/data/realiad/realiad_fake"

    if not os.path.isdir(ROOT_INPUT_DIR):
        print(f"Error: Root input directory not found: {ROOT_INPUT_DIR}")
        exit()

    item_names = [d for d in os.listdir(ROOT_INPUT_DIR) if os.path.isdir(os.path.join(ROOT_INPUT_DIR, d))]

    if not item_names:
        print(f"No items (subdirectories) found in {ROOT_INPUT_DIR}")
        exit()
    
    print(f"Found {len(item_names)} items to process: {item_names}")

    for item_idx, current_item_name in enumerate(item_names):
        print(f"\n[{item_idx+1}/{len(item_names)}] Processing item: {current_item_name}")
        
        current_item_path = os.path.join(ROOT_INPUT_DIR, current_item_name)
        
        try:
            generate_semantic_maps_for_item(current_item_path, current_item_name)
            print(f"  Completed processing for item: {current_item_name}")
        except Exception as e:
            print(f"  An error occurred while processing item {current_item_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    print("\n--- All items processed. ---")