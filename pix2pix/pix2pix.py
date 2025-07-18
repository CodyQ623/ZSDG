import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import json
from collections import defaultdict # Added

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm  # Import tqdm for progress bars

# Ensure models and datasets are imported from the current directory
from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda.amp as amp

# Custom function to parse boolean arguments correctly
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# Modified data argument: only need the root now
parser.add_argument("--data_root", type=str, default="/home/cody/Projects/AnomDetect/anomaly_generation/AdaBLDM/data/mysemap_vec", help="Root directory containing item/type subdirectories")
# Added with_type argument with proper boolean parsing
parser.add_argument("--with_type", type=str2bool, default=False, help="Whether items have type subdirectories (True) or directly contain source/target/semap (False)")
# Removed --item_name and --type_name arguments
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches (recommend > 1 for BatchNorm)")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels (grayscale)")
parser.add_argument("--vector_dim", type=int, default=6, help="dimension of the control vector")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between model checkpoints")
parser.add_argument("--output_name", type=str, default="all_items_combined", help="name of the output directory for images and models")
opt = parser.parse_args()
print(opt)

# Updated output paths for combined training
output_dir_base = opt.output_name
output_image_dir = f"images/{output_dir_base}"
output_model_dir = f"saved_models/{output_dir_base}"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_model_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight
lambda_pixel = 200

# Calculate output of image discriminator (PatchGAN)
patch_h = opt.img_height // (2 ** 4)
patch_w = opt.img_width // (2 ** 4)
patch = (1, patch_h, patch_w)

# Initialize generator and discriminator
generator = GeneratorUNet(in_channels=opt.channels, out_channels=opt.channels, vector_dim=opt.vector_dim)
discriminator = Discriminator(in_channels=opt.channels, vector_dim=opt.vector_dim)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    try:
        generator.load_state_dict(torch.load(f"{output_model_dir}_old/generator_{opt.epoch}.pth"))
        discriminator.load_state_dict(torch.load(f"{output_model_dir}_old/discriminator_{opt.epoch}.pth"))
        print(f"Loaded models from epoch {opt.epoch}")
    except FileNotFoundError:
        print(f"Checkpoint not found for epoch {opt.epoch}, initializing from scratch.")
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
]

# Instantiate the dataset with the root directory and with_type parameter
# The dataset class now handles scanning subdirectories based on the with_type flag
full_dataset = ImageDataset(opt.data_root, transforms_=transforms_, with_type=opt.with_type)

# --- Print Dataset Structure ---
print("\n--- Dataset Structure Discovered ---")
dataset_structure = full_dataset.get_dataset_structure()
if not dataset_structure:
    print("No items or types found. Please check the --data_root path and directory structure.")
    exit()
else:
    if opt.with_type:
        for item, types in dataset_structure.items():
            print(f"Item: {item}")
            for type_name in types:
                print(f"  - Type: {type_name}")
    else:
        # When with_type is False, we just list items
        for item in dataset_structure.keys():
            print(f"Item: {item}")
print("----------------------------------\n")
# --- End Print Dataset Structure ---


dataloader = DataLoader(
    full_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    pin_memory=True, # Pin memory for faster data transfer to GPU
)

# Create a validation dataloader using the same dataset (or a subset if needed)
# For simplicity, using the same dataset for validation sampling here.
# Consider creating a separate validation split/dataset for rigorous evaluation.
val_dataloader = DataLoader(
    full_dataset, # Using the same dataset instance
    batch_size=10,
    shuffle=True, # Shuffle to get different samples each time
    num_workers=1,
    pin_memory=True, # Pin memory for faster data transfer to GPU
)


# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    try:
        # Use next(iter()) to get a batch without consuming the whole iterator state
        val_batch = next(iter(val_dataloader))
    except StopIteration:
        print("Validation dataloader exhausted for sampling. Consider increasing dataset size or reducing sampling frequency.")
        # Optionally reset the iterator if you want continuous sampling across epochs:
        # global val_dataloader_iter
        # val_dataloader_iter = iter(val_dataloader)
        # val_batch = next(val_dataloader_iter)
        return # Skip sampling this time

    source_mask = Variable(val_batch["source"].type(Tensor))
    real_target_mask = Variable(val_batch["target"].type(Tensor))
    prompt_vec = Variable(val_batch["prompt"].type(Tensor))

    generator.eval()
    with torch.no_grad():
        fake_target_mask = generator(source_mask, prompt_vec)
    generator.train()

    source_mask = (source_mask * 0.5) + 0.5
    fake_target_mask = (fake_target_mask * 0.5) + 0.5
    real_target_mask = (real_target_mask * 0.5) + 0.5

    fake_target_stats = fake_target_mask.detach()
    print(f"Generator stats: min={fake_target_stats.min().item():.4f}, max={fake_target_stats.max().item():.4f}, mean={fake_target_stats.mean().item():.4f}")

    # Adjust nrow based on the actual batch size used for validation
    n_val_samples = min(val_dataloader.batch_size, source_mask.size(0))
    img_sample = torch.cat((source_mask.data, fake_target_mask.data, real_target_mask.data), -2)
    save_image(img_sample, f"{output_image_dir}/{batches_done}.png", nrow=n_val_samples, normalize=False)


# ----------
#  Training
# ----------

print("Starting Training...")
prev_time = time.time()

# Use tqdm for epoch-level progress tracking
epoch_progress = tqdm(range(opt.epoch, opt.n_epochs), desc="Training Progress", position=0)
for epoch in epoch_progress:
    # Create a batch progress bar that will be reset each epoch
    batch_progress = tqdm(enumerate(dataloader), total=len(dataloader), 
                        desc=f"Epoch {epoch}/{opt.n_epochs}", position=1, leave=False)
    
    # Track running losses for display in the progress bar
    running_loss_D = 0.0
    running_loss_G = 0.0
    running_loss_pixel = 0.0
    running_loss_GAN = 0.0
    
    for i, batch in batch_progress:
        # Model inputs
        source_mask = Variable(batch["source"].type(Tensor))
        real_target_mask = Variable(batch["target"].type(Tensor))
        prompt_vec = Variable(batch["prompt"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((source_mask.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((source_mask.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()
        fake_target_mask = generator(source_mask, prompt_vec)
        pred_fake = discriminator(source_mask, fake_target_mask, prompt_vec)
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_pixel = criterion_pixelwise(fake_target_mask, real_target_mask)
        loss_G = loss_GAN + lambda_pixel * loss_pixel
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        pred_real = discriminator(source_mask, real_target_mask, prompt_vec)
        loss_real = criterion_GAN(pred_real, valid)
        pred_fake = discriminator(source_mask, fake_target_mask.detach(), prompt_vec)
        loss_fake = criterion_GAN(pred_fake, fake)
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Update Progress Bar
        # --------------
        batches_done = epoch * len(dataloader) + i
        
        # Update running losses
        running_loss_D += loss_D.item()
        running_loss_G += loss_G.item()
        running_loss_pixel += loss_pixel.item()
        running_loss_GAN += loss_GAN.item()
        
        # Update batch progress bar with current losses
        batch_progress.set_postfix({
            'D_loss': f"{running_loss_D/(i+1):.4f}",
            'G_loss': f"{running_loss_G/(i+1):.4f}", 
            'pixel': f"{running_loss_pixel/(i+1):.4f}", 
            'adv': f"{running_loss_GAN/(i+1):.4f}"
        })

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # Update epoch progress bar with average losses for the entire epoch
    avg_loss_D = running_loss_D / len(dataloader)
    avg_loss_G = running_loss_G / len(dataloader)
    avg_loss_pixel = running_loss_pixel / len(dataloader)
    avg_loss_GAN = running_loss_GAN / len(dataloader)
    
    epoch_progress.set_postfix({
        'D_loss': f"{avg_loss_D:.4f}",
        'G_loss': f"{avg_loss_G:.4f}", 
        'pixel': f"{avg_loss_pixel:.4f}", 
        'adv': f"{avg_loss_GAN:.4f}"
    })

    # Save model checkpoints periodically
    if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
        torch.save(generator.state_dict(), f"{output_model_dir}/generator_{epoch + 1}.pth")
        torch.save(discriminator.state_dict(), f"{output_model_dir}/discriminator_{epoch + 1}.pth")
        epoch_progress.write(f"Saved model checkpoint at epoch {epoch + 1}")

# Save final models
torch.save(generator.state_dict(), f"{output_model_dir}/generator_final.pth")
torch.save(discriminator.state_dict(), f"{output_model_dir}/discriminator_final.pth")
print(f"\nSaved final models at epoch {opt.n_epochs}")