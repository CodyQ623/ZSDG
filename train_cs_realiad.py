from share import *

import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
import cv2
import numpy as np
import json
import glob
import torch

from cldm.model import create_model, load_state_dict
from cldm.logger import ImageLogger

# Define root directory
# TODO: Please replace with your actual paths
ROOT_DIR = "/home/cody/Projects/AnomDetect/anomaly_generation/langcode"
DATASET_PATH = "/home/cody/Projects/AnomDetect/anomaly_generation/langcode/data/realiad/realiad_fake"


class RealiadDataset(Dataset):
    """Dataset loader for Realiad items (without defect-type subdivision)"""
    def __init__(self, item_list):
        """
        Initialize dataset with Realiad items
        
        Args:
            item_list: List of Realiad item names or "all" to use all available items
        """
        self.data = []
        self.items = []
        
        # Handle 'all' option by finding all available items
        if item_list == ["all"]:
            print("Loading ALL available Realiad items...")
            self.items = [d for d in os.listdir(DATASET_PATH) 
                         if os.path.isdir(os.path.join(DATASET_PATH, d))]
        else:
            # Use the provided item list
            self.items = item_list
        
        # Load data from all items
        for item_name in self.items:
            dataset_root = os.path.join(DATASET_PATH, item_name)
            
            # Check if directory and prompt file exist
            prompt_file = os.path.join(dataset_root, "prompt.json")
            if not os.path.exists(prompt_file):
                print(f"Warning: Prompt file not found: {prompt_file}")
                continue
            
            # Load data items
            try:
                with open(prompt_file, 'rt') as f:
                    for line in f:
                        item_data = json.loads(line)
                        # Add item name for tracking
                        item_data['item_name'] = item_name
                        item_data['dataset_root'] = dataset_root
                        self.data.append(item_data)
            except Exception as e:
                print(f"Error loading data from {dataset_root}: {e}")
        
        print(f"Loaded total of {len(self.data)} samples from {len(self.items)} Realiad items")
        for item_name in self.items:
            count = sum(1 for item in self.data if item['item_name'] == item_name)
            print(f"  - {item_name}: {count} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        dataset_root = item['dataset_root']
        
        source_filename = item['source']
        target_filename = item['target']
        semap_filename = item['semap']

        # Load SeMaP
        try:
            semap = np.load(os.path.join(dataset_root, semap_filename))
            if semap.shape[:2] != (256, 256):
                semap = cv2.resize(semap, (256, 256), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"Error loading SeMaP {os.path.join(dataset_root, semap_filename)}: {e}")
            return self.__getitem__((idx + 1) % len(self.data))

        # Load source and target images
        source = cv2.imread(os.path.join(dataset_root, source_filename))
        target = cv2.imread(os.path.join(dataset_root, target_filename))
        
        # Error handling
        if source is None or target is None:
            print(f"Error loading: {os.path.join(dataset_root, source_filename)} or {target_filename}")
            # Return another sample from the dataset
            return self.__getitem__((idx + 1) % len(self.data))
        
        # Resize
        source = cv2.resize(source, (256, 256))
        target = cv2.resize(target, (256, 256))

        # Convert color
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize
        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=semap, hint=source)

def main():
    parser = argparse.ArgumentParser(description="Train AdaBLDM model for Realiad anomaly generation")
    parser.add_argument("--include", type=str, required=True, 
                      help="Realiad item to include for training (e.g., realiad1, realiad2)")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default='./models/control_sd15_ini.ckpt', 
                     help="Initial checkpoint path")
    parser.add_argument("--postfix", type=str, default="cs")
    
    args = parser.parse_args()
    
    # Get all available Realiad items
    all_realiad_items = [d for d in os.listdir(DATASET_PATH) 
                   if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    if not all_realiad_items:
        print(f"Error: No Realiad items found in {DATASET_PATH}")
        return
    
    # Validate the included item
    include_item = args.include
    if include_item not in all_realiad_items:
        print(f"Error: '{include_item}' is not a valid Realiad item. Valid items are: {', '.join(all_realiad_items)}")
        return
    
    # Generate list of items to train on (only the included one)
    train_items = [include_item]
    
    print(f"Training on Realiad item: '{include_item}'")
    
    # Set up checkpoint directory based on included item
    checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints_realiad", f"{include_item}_{args.postfix}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    
    # Load model
    model = create_model('./models/cldm_v15.yaml').cpu()
    state_dict = load_state_dict(args.checkpoint, location='cpu')
    
    # Handle input layer weights
    if 'control_model.input_hint_block.0.weight' in state_dict:
        old_weight = state_dict['control_model.input_hint_block.0.weight']
        new_weight = torch.zeros((16, 4, 3, 3), device=old_weight.device, dtype=old_weight.dtype)
        new_weight[:, :3, :, :] = old_weight
        state_dict['control_model.input_hint_block.0.weight'] = new_weight
        print("Modified input conv weights from 3 to 4 channels")
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.learning_rate = args.lr
    model.sd_locked = True  # Lock SD weights
    model.only_mid_control = False
    
    # Create dataset and loader
    dataset = RealiadDataset(train_items)
    
    # Check if there's data
    if len(dataset) == 0:
        print("Error: No training data found after excluding the specified item.")
        return
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size*2,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
    )
    
    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=5,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        precision=16,
        callbacks=[checkpoint_callback],
        max_epochs=args.epochs,
        benchmark=True,
    )
    
    # Start training
    trainer.fit(model, dataloader)
    
    print(f"Training complete. Model saved to {checkpoint_dir}")

if __name__ == "__main__":
    main()
