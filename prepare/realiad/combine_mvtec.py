import os
import shutil
import itertools
from PIL import Image

def create_source_directory(source_item_path, item_output_dir, target_count, resize_resolution):
    """
    Creates the 'source' directory for an item, populated with augmented 'good' images.

    This function collects all 'good' images from the original train/good and test/good
    directories, resizes them, and cyclically copies them until a target count is
    reached. The output files are numbered sequentially (e.g., 0000.png).

    Args:
        source_item_path (str): The path to the original item directory (e.g., './carpet').
        item_output_dir (str): The path where the new item data will be stored (e.g., './mvtec_reorg/carpet').
        target_count (int): The desired number of images in the 'source' directory.
        resize_resolution (tuple): A tuple (width, height) for resizing the images.
    """
    print(f"  -> Creating 'source' directory for {os.path.basename(source_item_path)}...")
    source_output_dir = os.path.join(item_output_dir, 'source')
    os.makedirs(source_output_dir, exist_ok=True)

    # 1. Collect paths of all 'good' images from the original dataset
    good_image_paths = []
    train_good_dir = os.path.join(source_item_path, 'train', 'good')
    test_good_dir = os.path.join(source_item_path, 'test', 'good')

    for source_dir in [train_good_dir, test_good_dir]:
        if os.path.isdir(source_dir):
            for filename in sorted(os.listdir(source_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    good_image_paths.append(os.path.join(source_dir, filename))
        else:
            print(f"     Warning: Directory not found, skipping: {source_dir}")

    if not good_image_paths:
        print(f"     Warning: No 'good' images found for item. 'source' directory will be empty.")
        return

    print(f"     Found {len(good_image_paths)} original 'good' images. Populating to {target_count} images.")

    # 2. Loop to copy, resize, and rename images until the target count is reached
    image_source_cycler = itertools.cycle(good_image_paths)

    for i in range(target_count):
        original_path = next(image_source_cycler)
        new_filename = f"{i:04d}.png"
        dest_path = os.path.join(source_output_dir, new_filename)

        try:
            with Image.open(original_path) as img:
                img_rgb = img.convert("RGB")
                # Image.Resampling.LANCZOS is a high-quality downsampling filter
                img_resized = img_rgb.resize(resize_resolution, Image.Resampling.LANCZOS)
                img_resized.save(dest_path, "PNG")
        except Exception as e:
            print(f"     Error: Failed to process image {original_path}: {e}")
    
    print(f"     Successfully created 'source' directory with {target_count} images.")


def create_combined_directory(source_item_path, item_output_dir):
    """
    Creates the 'combined' directory for an item, merging ground truth and test data.

    This function reorganizes the defect masks (ground_truth), normal test images (test/good),
    and anomalous test images (test/ng) into a standardized structure, prefixing defect
    files with their type name.

    Args:
        source_item_path (str): The path to the original item directory (e.g., './carpet').
        item_output_dir (str): The path where the new item data will be stored (e.g., './mvtec_reorg/carpet').
    """
    item_name = os.path.basename(source_item_path)
    print(f"  -> Creating 'combined' directory for {item_name}...")
    
    # 1. Define and create the output directory structure
    combined_dir = os.path.join(item_output_dir, "combined")
    combined_ground_truth_dir = os.path.join(combined_dir, "ground_truth")
    combined_test_dir = os.path.join(combined_dir, "test")
    combined_test_ng_dir = os.path.join(combined_test_dir, "ng")
    combined_test_good_dir = os.path.join(combined_test_dir, "good")

    os.makedirs(combined_ground_truth_dir, exist_ok=True)
    os.makedirs(combined_test_ng_dir, exist_ok=True)
    os.makedirs(combined_test_good_dir, exist_ok=True)

    # 2. Identify defect types from the 'ground_truth' directory
    ground_truth_base_dir = os.path.join(source_item_path, "ground_truth")
    defect_types = []
    if os.path.isdir(ground_truth_base_dir):
        defect_types = [d for d in os.listdir(ground_truth_base_dir) if os.path.isdir(os.path.join(ground_truth_base_dir, d))]
    
    if not defect_types:
        print(f"     Warning: No defect types found in {ground_truth_base_dir}. 'ng' and 'ground_truth' folders will be empty.")

    # 3. Process ground_truth mask images
    for type_name in defect_types:
        type_ground_truth_dir = os.path.join(ground_truth_base_dir, type_name)
        if not os.path.isdir(type_ground_truth_dir):
            continue
        for file in os.listdir(type_ground_truth_dir):
            if file.lower().endswith('.png'):
                original_mask_path = os.path.join(type_ground_truth_dir, file)
                new_mask_name = f"{type_name}_{file}"
                dest_mask_path = os.path.join(combined_ground_truth_dir, new_mask_name)
                try:
                    shutil.copy2(original_mask_path, dest_mask_path)
                except Exception as e:
                    print(f"     Error copying mask {original_mask_path}: {e}")

    # 4. Process 'test/ng' (anomalous) images
    test_base_dir = os.path.join(source_item_path, "test")
    for type_name in defect_types:
        type_test_dir = os.path.join(test_base_dir, type_name)
        if not os.path.isdir(type_test_dir):
            continue
        for file in os.listdir(type_test_dir):
            if file.lower().endswith(('.jpg', '.png')):
                original_ng_path = os.path.join(type_test_dir, file)
                base_name, _ = os.path.splitext(file)
                new_ng_name = f"{type_name}_{base_name}.png"
                dest_ng_path = os.path.join(combined_test_ng_dir, new_ng_name)
                try:
                    img = Image.open(original_ng_path).convert("RGB")
                    img.save(dest_ng_path, "PNG")
                except Exception as e:
                    print(f"     Error processing NG image {original_ng_path}: {e}")

    # 5. Process 'test/good' (normal) images
    test_good_source_dir = os.path.join(test_base_dir, "good")
    if os.path.isdir(test_good_source_dir):
        good_files = sorted([f for f in os.listdir(test_good_source_dir) if f.lower().endswith(('.jpg', '.png'))])
        for i, good_file in enumerate(good_files):
            original_good_path = os.path.join(test_good_source_dir, good_file)
            new_good_name = f"{i:03d}.png"
            dest_good_path = os.path.join(combined_test_good_dir, new_good_name)
            try:
                img = Image.open(original_good_path).convert("RGB")
                img.save(dest_good_path, "PNG")
            except Exception as e:
                print(f"     Error processing good image {original_good_path}: {e}")
    else:
        print(f"     Warning: test/good directory not found: {test_good_source_dir}.")

    print(f"     Successfully created 'combined' directory.")


def process_mvtec_dataset(target_count=2000, resize_resolution=(256, 256)):
    """
    Reorganizes the MVTec dataset by creating a new data structure for each object,
    which includes both an augmented 'source' directory of normal images and a
    'combined' directory of structured test and ground truth data.

    This script should be run from the root directory of the MVTec dataset.
    It will create a new directory 'mvtec_reorg' at the same level as the dataset folder.

    Final Structure for each item:
    mvtec_reorg/
    └── [item_name]/
        ├── source/
        │   ├── 0000.png
        │   └── ... (resized normal images, augmented to target_count)
        └── combined/
            ├── ground_truth/
            │   └── (defect masks, prefixed with type)
            └── test/
                ├── good/
                │   └── (renumbered normal images)
                └── ng/
                    └── (anomalous images, prefixed with type)
    """
    try:
        source_root = os.getcwd()
        output_root = os.path.join(os.path.dirname(source_root), 'mvtec_reorg')

        print(f"Source MVTec Directory: {source_root}")
        print(f"Target Output Directory: {output_root}\n")
        os.makedirs(output_root, exist_ok=True)

        # Identify all object categories by checking for the existence of a 'train' folder
        items = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d, 'train'))]
        
        if not items:
            print("Error: No valid MVTec object folders found.")
            print("Please run this script from the root directory of the MVTec dataset.")
            return

        print(f"Found {len(items)} items to process: {items}\n")

        # Process each item
        for item in items:
            print(f"--- Processing item: {item} ---")
            
            source_item_path = os.path.join(source_root, item)
            item_output_dir = os.path.join(output_root, item)
            os.makedirs(item_output_dir, exist_ok=True)

            # Task 1: Create the 'source' directory with augmented good images
            create_source_directory(source_item_path, item_output_dir, target_count, resize_resolution)

            # Task 2: Create the 'combined' directory with merged data
            create_combined_directory(source_item_path, item_output_dir)
            
            print(f"--- Finished processing: {item} ---\n")

        print("All items processed. Dataset reorganization is complete!")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    # Set the total number of 'good' images you want in the 'source' directory.
    TARGET_IMAGE_COUNT = 2000 
    
    # Set the desired resolution for the images in the 'source' directory.
    RESIZE_RESOLUTION = (256, 256)
    
    # Run the main processing function
    # Using a small TARGET_IMAGE_COUNT for quick testing. 
    # For actual use, you might set it to 2000 or higher.
    process_mvtec_dataset(
        target_count=TARGET_IMAGE_COUNT, 
        resize_resolution=RESIZE_RESOLUTION
    )