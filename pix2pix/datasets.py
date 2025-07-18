import glob
import random
import os
import json
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    # Modified __init__ to accept data_root directly
    def __init__(self, data_root, transforms_=None, mode="train", with_type=True):
        self.transform = transforms.Compose(transforms_)
        self.with_type = with_type  # New parameter to control directory structure
        self.mode = mode # Note: mode isn't strictly used in loading anymore but kept for consistency
        self.root = data_root # Root is now the base directory like /path/to/data/mysemap_vec
        self.files = []
        self.dataset_structure = defaultdict(list) # To store discovered items and types
        self._load_data_list()

    def _load_data_list(self):
        print(f"Scanning for datasets in: {self.root} with with_type={self.with_type}")
        if not os.path.isdir(self.root):
             print(f"ERROR: Data root directory not found: {self.root}")
             raise RuntimeError(f"Data root directory not found: {self.root}")

        if self.with_type:
            print("Using hierarchical structure: root/item/type/[source,target,semap]")
            self._scan_with_type_structure(self.root)
        else:
            print("Using flat structure: root/item/[source,target,semap]")
            self._scan_without_type_structure(self.root)

        if not self.files:
             print(f"ERROR: No valid data found scanning subdirectories in {self.root}")
             # List the contents of root to help diagnose
             try:
                 print(f"Contents of root directory ({self.root}):")
                 for name in os.listdir(self.root):
                     path = os.path.join(self.root, name)
                     if os.path.isdir(path):
                         print(f"  Directory: {name}")
                         # List the first-level subdirectories too
                         try:
                             subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                             print(f"    Subdirectories: {subdirs}")
                         except Exception as e:
                             print(f"    Error listing subdirectories: {e}")
                     else:
                         print(f"  File: {name}")
             except Exception as e:
                 print(f"Error listing root directory: {e}")
                 
             raise RuntimeError(f"No valid data found scanning subdirectories in {self.root}")
        print(f"Total samples loaded: {len(self.files)}")

    def _scan_with_type_structure(self, root):
        """Scan directory with structure: root/item/type/[source,target,semap]"""
        # Find all items (first level folders)
        items = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        print(f"Found {len(items)} potential item directories: {items}")
        
        for item in items:
            item_path = os.path.join(root, item)
            # Find all types for this item (second level folders)
            types = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
            print(f"Item '{item}' has {len(types)} potential type directories: {types}")
            
            for type_name in types:
                type_path = os.path.join(item_path, type_name)
                
                # Check for source/target/semap directories
                source_path = os.path.join(type_path, "source")
                target_path = os.path.join(type_path, "target")
                semap_path = os.path.join(type_path, "semap")
                
                src_exists = os.path.exists(source_path)
                tgt_exists = os.path.exists(target_path)
                semap_exists = os.path.exists(semap_path)
                
                print(f"  Checking type '{type_name}' - source:{src_exists}, target:{tgt_exists}, semap:{semap_exists}")
                
                if src_exists and tgt_exists and semap_exists:
                    # Record this item-type combination in the structure
                    self.dataset_structure[item].append(type_name)
                    
                    # Find all files in source directory
                    source_files = sorted(glob.glob(os.path.join(source_path, "*.*")))
                    print(f"    Found {len(source_files)} potential source files")
                    
                    count_valid = 0
                    for source_file in source_files:
                        filename = os.path.basename(source_file)
                        name, ext = os.path.splitext(filename)
                        
                        # Construct corresponding paths
                        target_file = os.path.join(target_path, f"{name}{ext}")
                        semap_file = os.path.join(semap_path, f"{name}.npy")
                        
                        # Check if corresponding files exist
                        if os.path.exists(target_file) and os.path.exists(semap_file):
                            self.files.append({
                                "source": source_file,
                                "target": target_file, 
                                "semap": semap_file,
                                "item": item,
                                "type": type_name
                            })
                            count_valid += 1
                    
                    print(f"    Added {count_valid} valid samples for type '{type_name}'")
    
    def _scan_without_type_structure(self, root):
        """Scan directory with structure: root/item/[source,target,semap]"""
        # Find all items (first level folders)
        items = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        print(f"Found {len(items)} potential item directories: {items}")
        
        for item in items:
            item_path = os.path.join(root, item)
            
            # Check for source/target/semap directories directly under item
            source_path = os.path.join(item_path, "source")
            target_path = os.path.join(item_path, "target")
            semap_path = os.path.join(item_path, "semap")
            
            src_exists = os.path.exists(source_path)
            tgt_exists = os.path.exists(target_path)
            semap_exists = os.path.exists(semap_path)
            
            print(f"Checking item '{item}' - source:{src_exists}, target:{tgt_exists}, semap:{semap_exists}")
            
            if src_exists and tgt_exists and semap_exists:
                # Record this item in the structure (with empty type list)
                self.dataset_structure[item] = []
                
                # Find all files in source directory
                source_files = sorted(glob.glob(os.path.join(source_path, "*.*")))
                print(f"  Found {len(source_files)} potential source files")
                
                count_valid = 0
                for source_file in source_files:
                    filename = os.path.basename(source_file)
                    name, ext = os.path.splitext(filename)
                    
                    # Skip non-image files or special files
                    if ext.lower() not in ['.png', '.jpg', '.jpeg', '.bmp'] or filename == 'label.json':
                        continue
                    
                    # Construct corresponding paths
                    target_file = os.path.join(target_path, f"{name}{ext}")
                    semap_file = os.path.join(semap_path, f"{name}.npy")
                    
                    # Check if corresponding files exist
                    if os.path.exists(target_file) and os.path.exists(semap_file):
                        self.files.append({
                            "source": source_file,
                            "target": target_file, 
                            "semap": semap_file,
                            "item": item,
                            "type": None  # No type for this structure
                        })
                        count_valid += 1
                    else:
                        if not os.path.exists(target_file):
                            print(f"    Missing target file: {target_file}")
                        if not os.path.exists(semap_file):
                            print(f"    Missing semap file: {semap_file}")
                
                print(f"  Added {count_valid} valid samples for item '{item}'")

    # Method to get the discovered structure
    def get_dataset_structure(self):
        return self.dataset_structure

    def __getitem__(self, index):
        """Get the item at the specified index"""
        file_info = self.files[index]
        
        # Load the source image (grayscale mask)
        source_img = Image.open(file_info["source"]).convert('L')
        source_img = self.transform(source_img)
        
        # Load the target image (grayscale mask)
        target_img = Image.open(file_info["target"]).convert('L')
        target_img = self.transform(target_img)
        
        # Load the prompt vector from the semantic map file
        semap = np.load(file_info["semap"])
        # Extract prompt vector from semantic map
        # Assuming the semantic map has dimensions H x W x C
        # We need to extract a single vector from this map
        
        # Method 1: Use the non-zero mean of the semantic map
        non_zero = semap > 0
        if np.any(non_zero):
            # If there are any non-zero values, take their mean
            prompt_vector = np.zeros(semap.shape[-1], dtype=np.float32)
            for c in range(semap.shape[-1]):
                if np.any(non_zero[:, :, c]):
                    prompt_vector[c] = np.mean(semap[:, :, c][non_zero[:, :, c]])
        else:
            # If all values are zero, use zeros for the prompt vector
            prompt_vector = np.zeros(semap.shape[-1], dtype=np.float32)
        
        return {"source": source_img, "target": target_img, "prompt": prompt_vector}

    def __len__(self):
        return len(self.files)