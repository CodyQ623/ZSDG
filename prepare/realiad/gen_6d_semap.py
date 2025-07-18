import os
import cv2
import numpy as np
import json
import glob
import re
from tqdm import tqdm

# Path definitions
ADABLDM_ROOT = "/home/cody/Projects/AnomDetect/anomaly_generation/langcode"
PCB_DATA_PATH = os.path.join(ADABLDM_ROOT, "data/realiad/realiad_fake")
OUTPUT_ROOT = os.path.join(ADABLDM_ROOT, "data/realiad/realiad_6dsemap")

# Add anomaly region scale factor
ANOMALY_SCALE_FACTOR = 1  # Set the scale factor for anomaly region enlargement, can be adjusted as needed

def extract_file_number(file_path):
    """Extract sequence number from file path"""
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Try to directly parse filename as integer (if it's pure numeric)
    if name_without_ext.isdigit():
        return int(name_without_ext)
    
    # If filename is not pure numeric, try to extract numeric part
    numbers = re.findall(r'\d+', name_without_ext)
    if numbers:
        return int(numbers[0])
    
    # If unable to extract numbers, return None, subsequent code will handle this case
    return None

def calculate_centroid(mask):
    """Calculate the centroid of an anomaly region in the mask"""
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        # Normalize coordinates to [0,1] range
        height, width = mask.shape
        cX_norm = cX / width
        cY_norm = cY / height  # Note: (0,0) is at top-left in original coordinate system
        return cX_norm, cY_norm
    return 0.5, 0.5  # Default value is center point

def calculate_foreground_mask_from_original_dimensions(w_orig, h_orig, current_size=(256, 256)):
    """
    Calculate foreground mask position in current image (usually 256x256) based on original dimensions
    Returns a binary mask, 1 represents foreground area
    """
    w_current, h_current = current_size
    foreground_mask = np.ones((h_current, w_current), dtype=np.uint8)
    
    # If original image is already square, entire current image is foreground
    if w_orig == h_orig:
        return foreground_mask * 255
    
    # If original image is non-square, calculate padding area
    target_dim = max(w_orig, h_orig)
    
    if h_orig < w_orig:  # Original image is wider than tall
        total_pad_h = target_dim - h_orig
        pad_ratio = total_pad_h / (2 * target_dim)  # Ratio of padding to total size
        pad_pixels = int(pad_ratio * h_current)  # Padding pixels on current height
        
        # Set padding areas to 0 (non-foreground)
        foreground_mask[:pad_pixels, :] = 0  # Top padding
        foreground_mask[-pad_pixels:, :] = 0  # Bottom padding
        
    elif w_orig < h_orig:  # Original image is taller than wide
        total_pad_w = target_dim - w_orig
        pad_ratio = total_pad_w / (2 * target_dim)  # Ratio of padding to total size
        pad_pixels = int(pad_ratio * w_current)  # Padding pixels on current width
        
        # Set padding areas to 0 (non-foreground)
        foreground_mask[:, :pad_pixels] = 0  # Left padding
        foreground_mask[:, -pad_pixels:] = 0  # Right padding
    
    return foreground_mask * 255  # Adjust mask values to 0-255 range

def create_semap_from_mask_and_prompt(mask, prompt_vector):
    """
    Create 6D SeMaP from foreground mask and prompt vector, ensuring precise preservation of original vector values
    
    Args:
        mask: Binary foreground mask (256x256)
        prompt_vector: 6-dimensional vector [p1, p2, p3, p4, x, y]
        
    Returns:
        6-channel SeMaP (256x256x6)
    """
    height, width = mask.shape
    # Create 6-channel semantic map
    semantic_map = np.zeros((height, width, 6), dtype=np.float64)  # Use float64 to preserve higher precision
    
    # Create a binary mask to avoid floating point comparison issues
    binary_mask = (mask > 0)
    
    # First 4 channels correspond to the first 4 dimensions of original prompt vector
    for i in range(4):
        # No matter how small the value, preserve the original prompt value
        channel = np.zeros((height, width), dtype=np.float64)
        channel[binary_mask] = prompt_vector[i]
        semantic_map[:, :, i] = channel
    
    # Last 2 channels correspond to position information
    for i in range(4, 6):
        channel = np.zeros((height, width), dtype=np.float64)
        channel[binary_mask] = prompt_vector[i]
        semantic_map[:, :, i] = channel
    
    # Verify if mapping is correct
    if np.any(binary_mask):
        # Take any point in the mask region, check if the first four dimensions are consistent with the provided vector
        y_idx, x_idx = np.where(binary_mask)
        if len(y_idx) > 0:
            sample_y, sample_x = y_idx[0], x_idx[0]
            for i in range(4):
                # Verify if values are consistent (considering precision)
                stored_val = semantic_map[sample_y, sample_x, i]
                expected_val = prompt_vector[i]
                if abs(stored_val - expected_val) > 1e-10:
                    print(f"Warning: Vector values inconsistent! Channel {i}: stored value={stored_val}, expected value={expected_val}")
    
    return semantic_map

def verify_semap(semap_path, original_prompt):
    """Verify if saved SeMaP correctly preserves original prompt values"""
    try:
        semap = np.load(semap_path)
        # Find non-zero regions
        non_zero = np.any(semap > 0, axis=2)
        if np.any(non_zero):
            y_idx, x_idx = np.where(non_zero)
            if len(y_idx) > 0:
                sample_y, sample_x = y_idx[0], x_idx[0]
                stored_vector = semap[sample_y, sample_x, :4]
                # print(f"Verifying SeMaP: Original vector = {original_prompt}")
                # print(f"Verifying SeMaP: Stored vector = {stored_vector}")
                
                # Verify if the first 4 dimensions are consistent
                for i in range(min(4, len(original_prompt))):
                    if abs(stored_vector[i] - original_prompt[i]) > 1e-10:
                        print(f"  !! Error: Channel {i} inconsistent: stored value={stored_vector[i]}, original value={original_prompt[i]}")
                        return False
                return True
        print("  Warning: No non-zero regions found in SeMaP")
        return False
    except Exception as e:
        print(f"  Error verifying SeMaP: {e}")
        return False

def read_prompts_from_json(item_path):
    """Read prompt vectors from a JSON file"""
    prompts = {}
    label_json_path = os.path.join(item_path, "target", "label.json")
    
    if not os.path.exists(label_json_path):
        print(f"Warning: Prompt JSON file not found: {label_json_path}")
        return prompts
    
    try:
        with open(label_json_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                image_name = data.get("image", "")
                prompt = data.get("prompt", [0.5, 0.5, 0.5, 0.5])
                if image_name:
                    # Ensure that values in the vector are not treated as 0
                    prompts[image_name] = [float(v) for v in prompt]  # Explicitly convert to float
        
        print(f"Loaded {len(prompts)} prompt vectors from {label_json_path}")
        
        # Print some sample vectors for debugging
        if prompts:
            samples = list(prompts.items())[:3]  # Take the first 3 samples
            print("Sample vectors:")
            for name, vec in samples:
                print(f"  {name}: {vec}")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
    
    return prompts

def generate_source_files(file_number, foreground_mask, output_dir):

    source_dir = os.path.join(output_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
    
    source_image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    for c in range(3):
        source_image[:, :, c] = np.where(foreground_mask > 0, 255, 0)
    
    source_filename = f"{file_number:03d}.png"
    source_path = os.path.join(source_dir, source_filename)
    cv2.imwrite(source_path, source_image)
    
    return f"source/{source_filename}"

def generate_target_files(file_number, mask_path, output_dir):

    target_dir = os.path.join(output_dir, "target")
    os.makedirs(target_dir, exist_ok=True)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Cannot read mask:{mask_path}")
        mask = np.zeros((256, 256), dtype=np.uint8)
    
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    if ANOMALY_SCALE_FACTOR > 1.0:
        kernel_size = int(max(3, ANOMALY_SCALE_FACTOR * 3))  
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        binary_mask = dilated_mask
    
    target_image = np.stack([binary_mask, binary_mask, binary_mask], axis=-1)
    
    target_filename = f"{file_number:03d}.png"
    target_path = os.path.join(target_dir, target_filename)
    cv2.imwrite(target_path, target_image)
    
    return target_filename, f"target/{target_filename}"

def generate_semap_files(file_number, mask, prompt_vector, output_dir):

    semap_dir = os.path.join(output_dir, "semap")
    os.makedirs(semap_dir, exist_ok=True)
    
    semap = create_semap_from_mask_and_prompt(mask, prompt_vector)
    
    semap_filename = f"{file_number:03d}.npy"
    semap_path = os.path.join(semap_dir, semap_filename)
    
    np.save(semap_path, semap)
    
    verify_success = verify_semap(semap_path, prompt_vector[:4])
    if not verify_success:
        print(f"Warning: verify semap failed{semap_filename}")
    
    vis_map = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(3):
        channel = semap[:, :, i]
        if np.max(channel) > 0:
            normalized = channel / np.max(channel) * 255
        else:
            normalized = channel * 255
        vis_map[:, :, i] = normalized.astype(np.uint8)
    
    vis_path = os.path.join(semap_dir, f"{file_number:03d}_vis.png")
    cv2.imwrite(vis_path, vis_map)
    
    return f"semap/{semap_filename}"

def generate_prompt_json(file_data, output_dir):
    prompt_file = os.path.join(output_dir, "prompt.json")
    
    with open(prompt_file, 'w') as f:
        for data in file_data:
            f.write(json.dumps(data) + '\n')
    
    print(f"Generated prompt.json with {len(file_data)} entries at {prompt_file}")
    

from gen_fg import extract_foreground_connected

def process_pcb_item(item_name):
    print(f"Processing realiad items: {item_name}")
    
    item_path = os.path.join(PCB_DATA_PATH, item_name)
    mask_dir = os.path.join(item_path, "mask")
    source_dir = os.path.join(item_path, "source")  
    image_size_path = os.path.join(item_path, "image_size.txt")
    
    if not os.path.exists(mask_dir):
        print(f"Error: no mask dir found{mask_dir}")
        return False
    
    if os.path.exists(image_size_path):
        try:
            with open(image_size_path, 'r') as f:
                size_data = f.read().strip().split(',')
                original_width = int(size_data[0])
                original_height = int(size_data[1])
                print(f"Read original size: {original_width}x{original_height}")
                
                foreground_mask = calculate_foreground_mask_from_original_dimensions(
                    original_width, original_height, (256, 256)
                )
        except Exception as e:
            print(f"Warning: Cannot read image_size.txt: {e}, will use foreground extraction method.")
            foreground_mask = None  # Initialize to None, will extract foreground for each image individually later
    else:
        print(f"Warning: Image size file not found: {image_size_path}. Will use foreground extraction method.")
        foreground_mask = None  # Initialize to None, will extract foreground for each image individually later
    
    # Read prompts from JSON file
    item_prompts = read_prompts_from_json(item_path)
    
    if not item_prompts:
        print(f"Warning: Prompt vectors for item '{item_name}' not found. Will use default values.")
    
    # Get all mask files
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    if not mask_files:
        mask_files = glob.glob(os.path.join(mask_dir, "*.jpg"))
    
    if not mask_files:
        print(f"Error: No mask files found in {mask_dir}")
        return False
    
    # Output directory
    output_dir = os.path.join(OUTPUT_ROOT, item_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each mask file
    file_data = []
    label_data = []
    
    # Create a map from file number to file path
    file_number_map = {}
    for mask_path in mask_files:
        file_number = extract_file_number(mask_path)
        if file_number is not None:
            file_number_map[file_number] = mask_path
    
    # Sort file numbers to ensure processing in sequential order
    sorted_file_numbers = sorted(file_number_map.keys())
    
    for file_number in tqdm(sorted_file_numbers, desc=f"Processing {item_name}"):
        mask_path = file_number_map[file_number]
        
        # If foreground_mask is None, use foreground extraction method
        if foreground_mask is None:
            # Try to read the corresponding source image
            source_img_path = os.path.join(source_dir, f"{file_number:03d}.png")
            
            if os.path.exists(source_img_path):
                # Read the source image and extract the foreground
                source_img = cv2.imread(source_img_path)
                if source_img is not None:
                    # Use extract_foreground_connected to extract the foreground mask
                    _, curr_foreground_mask = extract_foreground_connected(source_img)
                    
                    # Ensure the mask has the correct dimensions (256x256)
                    if curr_foreground_mask.shape[:2] != (256, 256):
                        curr_foreground_mask = cv2.resize(curr_foreground_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                else:
                    print(f"  Warning: Cannot read source image: {source_img_path}, using default full foreground mask")
                    curr_foreground_mask = np.ones((256, 256), dtype=np.uint8) * 255
            else:
                print(f"  Warning: Source image not found: {source_img_path}, using default full foreground mask")
                curr_foreground_mask = np.ones((256, 256), dtype=np.uint8) * 255
        else:
            # Use the foreground mask calculated from the original dimensions
            curr_foreground_mask = foreground_mask
        
        # Create the corresponding target file name to look up the prompt
        target_filename = f"{file_number:03d}.png"
        
        # Read the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Cannot read mask: {mask_path}")
            continue
        
        # Ensure the mask is binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Modification: For the mask in semap, dilation is also needed
        if ANOMALY_SCALE_FACTOR > 1.0:
            kernel_size = int(max(3, ANOMALY_SCALE_FACTOR * 3))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            binary_mask_for_semap = cv2.dilate(binary_mask, kernel, iterations=1)
        else:
            binary_mask_for_semap = binary_mask
        
        # Get the prompt vector for this file
        prompt_vector = item_prompts.get(target_filename, None)
        if prompt_vector is None:
            # If the prompt vector is not found, use the default value
            prompt_vector = [0.5, 0.5, 0.5, 0.5]
            print(f"  Warning: Prompt vector for file '{target_filename}' not found, using default value")
        
        # Modification: Scale the first 4 dimensions of the prompt value
        scaled_prompt_vector = []
        for i, value in enumerate(prompt_vector):
            if i < 4:  # Multiply the first 4 dimensions by the scaling factor
                scaled_prompt_vector.append(value * ANOMALY_SCALE_FACTOR)
            else:
                scaled_prompt_vector.append(value)  # Keep subsequent dimensions unchanged (if any)
        
        # Calculate the centroid of the mask
        cx, cy = calculate_centroid(binary_mask_for_semap)
        
        # Create the complete 6D vector (using the scaled first 4 dimensions and original position information)
        full_prompt_vector = scaled_prompt_vector + [cx, cy]
        
        # Generate files - use the original file number and the foreground mask of the current file
        source_path = generate_source_files(file_number, curr_foreground_mask, output_dir)
        target_filename, target_path = generate_target_files(file_number, mask_path, output_dir)
        semap_path = generate_semap_files(file_number, binary_mask_for_semap, full_prompt_vector, output_dir)
        
        # Record data for prompt.json
        file_data.append({
            "source": source_path,
            "target": target_path,
            "semap": semap_path
        })
        
        # Record data for label.json (Note: here we save the scaled vector values)
        label_data.append({
            "image": target_filename,
            "prompt": scaled_prompt_vector[:4]  # Only use the scaled 4D vector part
        })
    
    # Generate prompt.json
    generate_prompt_json(file_data, output_dir)
    
    # Generate label.json in the target directory
    target_dir = os.path.join(output_dir, "target")
    label_json_path = os.path.join(target_dir, "label.json")
    
    with open(label_json_path, 'w') as f:
        for entry in label_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Generated label.json with {len(label_data)} entries")
    return True

def process_all_items():
    """Process all PCB items"""
    # Create the output root directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Get all PCB items
    items = [d for d in os.listdir(PCB_DATA_PATH) if os.path.isdir(os.path.join(PCB_DATA_PATH, d))]
    
    if not items:
        print(f"Error: No PCB items found in {PCB_DATA_PATH}")
        return
    
    print(f"Found {len(items)} PCB items to process: {items}")
    print(f"Using anomaly region enlargement factor: {ANOMALY_SCALE_FACTOR}x")
    
    # Process each item
    for item in items:
        process_pcb_item(item)
    
    print("All PCB items have been processed successfully!")

if __name__ == "__main__":
    process_all_items()