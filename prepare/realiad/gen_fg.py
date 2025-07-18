import cv2
import numpy as np
import os
import random
import glob
from pathlib import Path

def extract_foreground_connected(image, min_contour_area=100, closing_kernel_size=10):

    if image is None or image.size == 0:
        # For invalid input, return empty arrays
        empty_shape = (0, 0, 3) if image is None or len(image.shape) == 3 else (0, 0)
        empty_mask_shape = (0,0)
        if image is not None and image.shape:
            empty_shape = image.shape
            empty_mask_shape = image.shape[:2]

        return np.zeros(empty_shape, dtype=np.uint8), np.zeros(empty_mask_shape, dtype=np.uint8)

    h_img, w_img = image.shape[:2]

    # 1. Sample background region (top-left corner)
    bg_sample_dim = 5
    # Automatically handle boundaries when slicing, if image is smaller than 5x5, take actual size
    bg_patch = image[0:min(bg_sample_dim, h_img), 0:min(bg_sample_dim, w_img)]
    
    if bg_patch.size == 0: # If sampling region is empty (e.g., image height or width is 0)
        return image, np.zeros((h_img, w_img), dtype=np.uint8)

    # 2. Analyze HSV characteristics of background sample
    hsv_bg_patch = cv2.cvtColor(bg_patch, cv2.COLOR_BGR2HSV)
    avg_s_bg = np.mean(hsv_bg_patch[:, :, 1]) # Average saturation
    avg_v_bg = np.mean(hsv_bg_patch[:, :, 2]) # Average brightness (Value)

    # 3. Convert entire image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2] # V channel

    # 4. Define threshold constants for determining background type
    V_DARK_BG_THRESH = 75      # V value below this is considered dark background (0-255)
    V_LIGHT_BG_THRESH = 180    # V value above this is considered light background (0-255)
    S_LOW_THRESH = 60          # S value below this is considered low saturation (close to grayscale, 0-255)
    
    OFFSET_FROM_BG = 30        # Offset of V channel threshold from background V value
    MIN_SEPARATION = 15        # Ensure minimum separation between threshold and background V value

    binary_mask = np.zeros_like(v_channel) # Initialize binary mask

    # 5. Choose threshold strategy based on background characteristics
    is_achromatic_dark_bg = (avg_v_bg < V_DARK_BG_THRESH and avg_s_bg < S_LOW_THRESH)
    is_achromatic_light_bg = (avg_v_bg > V_LIGHT_BG_THRESH and avg_s_bg < S_LOW_THRESH)

    if is_achromatic_dark_bg:
        # Background close to pure black (low V, low S) -> foreground should be brighter
        threshold_value = avg_v_bg + OFFSET_FROM_BG
        threshold_value = np.clip(threshold_value, avg_v_bg + MIN_SEPARATION, 255.0 - MIN_SEPARATION)
        _, binary_mask = cv2.threshold(v_channel, int(threshold_value), 255, cv2.THRESH_BINARY)
    elif is_achromatic_light_bg:
        # Background close to pure white (high V, low S) -> foreground should be darker
        threshold_value = avg_v_bg - OFFSET_FROM_BG
        threshold_value = np.clip(threshold_value, MIN_SEPARATION, avg_v_bg - MIN_SEPARATION)
        _, binary_mask = cv2.threshold(v_channel, int(threshold_value), 255, cv2.THRESH_BINARY_INV)
    else:
        # Fallback logic: background might be colored, or neutral gray, or doesn't fit "very close to pure black/white" assumption
        # Use original logic based on grayscale average, but adjust threshold
        avg_gray_bg = np.mean(cv2.cvtColor(bg_patch, cv2.COLOR_BGR2GRAY))
        if avg_gray_bg < 128:  # Overall dark background
            _, binary_mask = cv2.threshold(v_channel, 70, 255, cv2.THRESH_BINARY)
        else:  # Overall bright background
            _, binary_mask = cv2.threshold(v_channel, 185, 255, cv2.THRESH_BINARY_INV)

    # 6. Morphological operations: First closing operation to smooth initial binary mask and fill small holes
    kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 7. Find contours and select largest contour
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_mask = np.zeros_like(processed_mask) # Create final blank mask
    
    if contours:
        # Filter out contours with too small area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # Second closing operation: apply closing to mask containing only largest contour to fill internal holes
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # 8. Apply mask to get foreground
    foreground = cv2.bitwise_and(image, image, mask=final_mask)
    
    return foreground, final_mask


def process_dataset(realiad_root):
    """Process dataset, extract foreground for each object and save"""
    # Get all object categories
    items_path = os.path.join(realiad_root, "realiad_reorg")
    items = [f.name for f in os.scandir(items_path) if f.is_dir()]
    
    total_images_per_item = 2000
    
    for item in items:
        print(f"Processing object: {item}")
        
        # Create target directory - note the modified path structure
        source_dir = os.path.join(realiad_root, "realiad_fg", item, "source")
        mask_dir = os.path.join(realiad_root, "realiad_fg", item, "mask")
        
        os.makedirs(source_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        good_dir = os.path.join(items_path, item, "train", "good")
        
        # Ensure directory exists
        if not os.path.exists(good_dir):
            print(f"Directory does not exist: {good_dir}")
            continue
            
        # Get all images in directory
        image_files = []
        image_files.extend(glob.glob(os.path.join(good_dir, "*.png")))
        image_files.extend(glob.glob(os.path.join(good_dir, "*.jpg")))
        image_files.extend(glob.glob(os.path.join(good_dir, "*.jpeg")))
        
        if not image_files:
            print(f"No images found in {good_dir}")
            continue
        
        # Prepare to collect enough images
        collected_images = []
        
        # If less than 200 images, copy existing images until reaching required number
        if len(image_files) < total_images_per_item:
            print(f"Object {item} has only {len(image_files)} images, will copy to reach {total_images_per_item}")
            # Calculate how many times each image needs to be copied
            copies_needed = int(np.ceil(total_images_per_item / len(image_files)))
            collected_images = image_files * copies_needed
        else:
            collected_images = image_files.copy()
        
        # Randomly select 200 images
        random.shuffle(collected_images)
        selected_images = collected_images[:total_images_per_item]
        
        # Process selected images
        for i, img_path in enumerate(selected_images):
            # Create new filename
            new_filename = f"{i:03d}.png"
            source_path = os.path.join(source_dir, new_filename)
            mask_path = os.path.join(mask_dir, new_filename)
            
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Cannot read image: {img_path}")
                continue
                
            # Copy original image to source directory
            cv2.imwrite(source_path, image)
            
            # Extract foreground and mask (using advanced version)
            _, mask = extract_foreground_connected(image)
            
            # Save mask to mask directory
            cv2.imwrite(mask_path, mask)
            
            if i % 20 == 0:  # Show progress every 20 images
                print(f"Object {item}: processed {i+1}/{total_images_per_item}")
        
        print(f"Completed object {item}: processed {total_images_per_item} images")

def main():
    # TODO: Please replace with actual REALIADROOT path
    realiad_root = "/home/cody/Projects/AnomDetect/anomaly_generation/langcode/data/realiad"
    
    if not realiad_root:
        print("Please set REALIADROOT environment variable")
        return
    
    if not os.path.exists(realiad_root):
        print(f"Path does not exist: {realiad_root}")
        return
        
    process_dataset(realiad_root)

if __name__ == "__main__":
    main()