import os
import shutil
from PIL import Image
import pandas as pd
import numpy as np 

# Define the attribute mapping based on item and type

# TODO: Define the attribute mapping based on item and type!
attribute_mapping = {
    ('toothbrush', 'ch'): [1, 1, 0, 0],
    ('toothbrush', 'qs'): [1, 1, 0, 0],
    ('toothbrush', 'zw'): [1, 0, 1, 0],
    ('vcpill', 'ak'): [1, 1, 0, 0],
    ('vcpill', 'hs'): [1, 1, 0, 0],
    ('vcpill', 'qs'): [1, 1, 0, 0],
    ('vcpill', 'zw'): [1, 0, 1, 0],
    ('zipper', 'bx'): [1, 1, 0, 0],
    ('zipper', 'ps'): [1, 1, 0, 0],
    ('zipper', 'qs'): [1, 1, 0, 0],
    ('zipper', 'zw'): [0, 1, 1, 0],
}

def get_attribute_vector(item_name, type_name):
    """
    Looks up the attribute vector for a given item and type.
    Returns [0.0, 0.0, 0.0, 0.0] if not found.
    Returns as float list to match ratio calculation.
    """
    # Use lower case for robust lookup
    # Return as list of floats
    return [float(val) for val in attribute_mapping.get((item_name.lower(), type_name.lower()), [0, 0, 0, 0])]

def calculate_white_ratio(mask_path):
    """
    Calculates the ratio of white pixels (value 255) in a grayscale image.
    """
    img = Image.open(mask_path)
    # Ensure the mask is in grayscale for pixel counting
    # 'L' mode is 8-bit pixels, grayscale. 0 is black, 255 is white.
    if img.mode != 'L':
         img = img.convert('L')

    pixels = img.getdata()
    # Count pixels with value 255 (typically white in a mask)
    white_pixels = sum(1 for pixel in pixels if pixel == 255)
    total_pixels = len(pixels)

    if total_pixels == 0:
        print(f"Warning: Mask file {mask_path} has zero pixels.")
        return 0.0 # Avoid division by zero

    return white_pixels / total_pixels

def process_data(base_dir="."):
    """
    处理当前目录下的图片数据并重新组织，同时生成带标注的Excel文件。
    只复制有对应掩码的异常图片，并保证图片-mask-Excel条目的对应。

    Args:
        base_dir (str): 基础目录，默认为当前目录。
    """

    mod_dir = os.path.join(base_dir, "realiad_reorg")
    os.makedirs(mod_dir, exist_ok=True)
    print(f"Ensured output directory exists: {mod_dir}")

    excel_data_by_item = {} # Dictionary to store excel data for each item

    # Process items in sorted order for consistent processing log and Excel sheets
    for item in sorted(os.listdir(base_dir)):
        if item == "realiad_reorg":
            continue
        item_path = os.path.join(base_dir, item)
        # Exclude mod folder itself and process only directories
        if os.path.isdir(item_path) and item != "mod":
            print(f"\nProcessing item: {item}...")

            item_mod_dir = os.path.join(mod_dir, item)
            os.makedirs(item_mod_dir, exist_ok=True)

            test_ng_dir = os.path.join(item_mod_dir, "test", "ng")
            ground_truth_ng_dir = os.path.join(item_mod_dir, "ground_truth", "ng")
            test_good_dir = os.path.join(item_mod_dir, "test", "good")
            train_good_dir = os.path.join(item_mod_dir, "train", "good")

            os.makedirs(test_ng_dir, exist_ok=True)
            os.makedirs(ground_truth_ng_dir, exist_ok=True)
            os.makedirs(test_good_dir, exist_ok=True)
            os.makedirs(train_good_dir, exist_ok=True)

            ng_dir = os.path.join(item_path, "NG")
            ok_dir = os.path.join(item_path, "OK")

            # List to store Excel data for the current item
            current_item_excel_data = []
            # Store the list even if empty, so the sheet is created
            excel_data_by_item[item] = current_item_excel_data

            # 1. Process NG samples (image+mask pairs) and generate Excel data
            ng_image_counter = 0
            if os.path.isdir(ng_dir):
                # Process types and ids in sorted order
                for type_name in sorted(os.listdir(ng_dir)):
                    type_path = os.path.join(ng_dir, type_name)
                    if os.path.isdir(type_path):
                        for id_name in sorted(os.listdir(type_path)):
                            id_path = os.path.join(type_path, id_name)
                            if os.path.isdir(id_path):
                                # Get list of files in id folder
                                files_in_id = os.listdir(id_path)
                                # Find all jpg files and sort them
                                jpg_files = sorted([f for f in files_in_id if f.lower().endswith('.jpg')])

                                # Process each jpg file to find its matching png
                                for jpg_file in jpg_files:
                                    file_name_without_ext, _ = os.path.splitext(jpg_file)
                                    png_file = file_name_without_ext + ".png"
                                    png_path = os.path.join(id_path, png_file)

                                    # Check if corresponding png mask exists
                                    if os.path.exists(png_path):
                                        # --- Found a valid NG sample pair (jpg + png mask) ---

                                        original_jpg_path = os.path.join(id_path, jpg_file)
                                        original_png_path = png_path # original path of the mask

                                        # Get the current counter value before using and incrementing
                                        current_file_number = ng_image_counter
                                        new_filename_base = f"{current_file_number:03d}"
                                        new_image_name = f"{new_filename_base}.png" # new filename for both image and mask

                                        dest_img_path = os.path.join(test_ng_dir, new_image_name)
                                        dest_mask_path = os.path.join(ground_truth_ng_dir, new_image_name) # Mask uses same base name as image


                                        # 1a. Copy, convert and save NG real image (JPG to PNG)
                                        img = Image.open(original_jpg_path).convert("RGB")
                                        img.save(dest_img_path, "PNG")
                                        # print(f"  Converted and saved NG image: {original_jpg_path} -> {dest_img_path}") # Reduced log

                                        # 1b. Copy, convert and save NG mask image (PNG to PNG, potentially L/RGBA)
                                        mask_img = Image.open(original_png_path)
                                        # Convert to L (grayscale) or keep RGBA if it has alpha channel
                                        # Added check to ensure it's not already RGB or P mode trying to convert to L
                                        if mask_img.mode not in ("L", "RGBA", "P"):
                                             mask_img = mask_img.convert("L")
                                        elif mask_img.mode == "P": # Handle paletted images, convert to L
                                             mask_img = mask_img.convert("L")

                                        mask_img.save(dest_mask_path, "PNG")
                                        # print(f"  Converted and saved NG mask: {original_png_path} -> {dest_mask_path}") # Reduced log

                                        # 1c. Calculate Label and prepare Excel data for this sample
                                        # Use the newly saved mask for ratio calculation
                                        ratio = calculate_white_ratio(dest_mask_path)
                                        # Use original item and type names for attribute lookup
                                        attribute_vector = get_attribute_vector(item, type_name)

                                        # Calculate label vector = ratio * attribute_vector
                                        # Ensure multiplication results in floats
                                        label = [ratio * val for val in attribute_vector]

                                        # Add data row to the list for this item's sheet
                                        current_item_excel_data.append({
                                            "File": new_filename_base,
                                            "Value_1": label[0],
                                            "Value_2": label[1],
                                            "Value_3": label[2],
                                            "Value_4": label[3]
                                        })
                                        # print(f"  Calculated label for {new_filename_base}.png (item: {item}, type: {type_name}, ratio: {ratio:.2f})") # Reduced log

                                        # 1d. Increment the counter for the next valid NG sample
                                        ng_image_counter += 1

                                    else:
                                        # Skipping jpg without corresponding png mask
                                        print(f"  Skipping NG image without corresponding mask: {os.path.join(id_path, jpg_file)}")


            # 2. Process OK images (Logic unchanged, reduced logs)
            ok_images = [] # Collect original paths first
            if os.path.isdir(ok_dir):
                # Process ids in sorted order
                for id_name in sorted(os.listdir(ok_dir)):
                    id_path = os.path.join(ok_dir, id_name)
                    if os.path.isdir(id_path):
                        # Collect all jpg files and sort them
                        ok_jpg_files = sorted([f for f in os.listdir(id_path) if f.lower().endswith('.jpg')])
                        for ok_jpg_file in ok_jpg_files:
                            original_path = os.path.join(id_path, ok_jpg_file)
                            ok_images.append(original_path)

            # Split OK images into test (20%) and train (80%) - Logic unchanged
            num_ok_images = len(ok_images)
            num_test_ok = int(num_ok_images * 0.2)
            test_ok_images = ok_images[:num_test_ok]
            train_ok_images = ok_images[num_test_ok:]

            # 2a. Save OK test set (Logic unchanged, reduced logs)
            ok_test_counter = 0
            for original_path in test_ok_images:
                new_image_name = f"{ok_test_counter:03d}.png"
                dest_path = os.path.join(test_good_dir, new_image_name)
                img = Image.open(original_path).convert("RGB")
                img.save(dest_path, "PNG")
                # print(f"  Converted and saved OK test image: {original_path} -> {dest_path}") # Reduced log
                ok_test_counter += 1

            # 2b. Save OK train set (Logic unchanged, reduced logs)
            ok_train_counter = 0
            for original_path in train_ok_images:
                new_image_name = f"{ok_train_counter:03d}.png"
                dest_path = os.path.join(train_good_dir, new_image_name)
                img = Image.open(original_path).convert("RGB")
                img.save(dest_path, "PNG")
                # print(f"  Converted and saved OK train image: {original_path} -> {dest_path}") # Reduced log
                ok_train_counter += 1

            print(f"Finished processing item: {item}.")
            print(f"  Processed {ng_image_counter} NG samples with masks.")
            print(f"  Processed {ok_test_counter} OK test samples.")
            print(f"  Processed {ok_train_counter} OK train samples.")


    # --- After processing all items, write the Excel file ---
    excel_output_path = os.path.join(mod_dir, "result.xlsx")
    print(f"\nWriting Excel file: {excel_output_path}...")

    # Use ExcelWriter to write multiple sheets
    try:
        with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
            for item, data in excel_data_by_item.items():
                # Create DataFrame for the current item
                df = pd.DataFrame(data)
                # Define column order explicitly
                required_columns = ["File", "Value_1", "Value_2", "Value_3", "Value_4"]
                # Reindex to ensure columns are in order, adding missing ones if any (shouldn't happen with current logic)
                df = df.reindex(columns=required_columns)

                # Write DataFrame to a sheet named after the item
                # index=False means don't write the pandas DataFrame index as a column
                df.to_excel(writer, sheet_name=item, index=False)
                print(f"  Sheet '{item}' created with {len(data)} NG samples.")
        print("Excel file written successfully.")
    except Exception as e:
        print(f"Error writing Excel file: {e}")
        # Re-raise the exception to stop execution as requested
        raise


    print("\nData processing complete.")


if __name__ == "__main__":
    process_data()
