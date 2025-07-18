import cv2
import numpy as np
import torch
import torch.nn.functional as F
import einops
import matplotlib.pyplot as plt
import copy
import tqdm
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import os
import gc
import argparse
import json
from GAN.patch_train import MultiScaleDiscriminator, SimpleDiscriminator

# TODO: Check the path
ADABLDM_ROOT = "/home/cody/Projects/AnomDetect/anomaly_generation/langcode"

# Load model function
def get_fresh_model(checkpoint_path):
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(
        load_state_dict(checkpoint_path, location='cpu'),
        strict=False
    )
    model = model.cuda()
    return model, DDIMSampler(model)

def reset_model_state():
    torch.cuda.empty_cache()
    gc.collect()
    
    if hasattr(model, 'first_stage_model') and hasattr(model.first_stage_model, 'decoder'):
        original_weights = copy.deepcopy(model.first_stage_model.decoder.state_dict())
        model.first_stage_model.decoder.apply(lambda m: m.reset_parameters() 
                                              if hasattr(m, 'reset_parameters') else None)
        model.first_stage_model.decoder.load_state_dict(original_weights)
    
    global ddim_sampler
    ddim_sampler = DDIMSampler(model)
    
    print("Model state reset completed")

def tensor2img(tensor: torch.Tensor):
    """Convert tensor to numpy image"""
    a = (
        (einops.rearrange(tensor, 'b c h w -> b h w c') * 127.5 + 127.5)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )
    return a

def extract_mask_from_semap(semap: np.ndarray):
    """Extract binary mask from SeMaP by summing channels"""
    channel_sum = np.sum(semap, axis=-1)
    binary_mask = (channel_sum > 0).astype(np.float32)
    return binary_mask

def calculate_discriminator_score(discriminator, image_tensor, is_multiscale=True):
    disc_output = discriminator(image_tensor)
    
    if is_multiscale and isinstance(disc_output, dict):
        receptive_fields = {
            'scale1': 16, 'scale2': 34, 'scale3': 70, 
            'scale4': 142, 'scale5': 286
        }
        
        total_weight = sum(receptive_fields.values())
        weighted_score = 0.0
        
        scale_scores = {}
        for scale_name, score in disc_output.items():
            scale_score = score.mean()
            scale_scores[scale_name] = scale_score.item()
            weight = receptive_fields.get(scale_name, 0)
            weighted_score += scale_score * weight
        
        final_score = weighted_score / total_weight
        return final_score, scale_scores
    else:
        final_score = torch.clamp(disc_output.mean(), 1e-6, 1.0 - 1e-6)
        return final_score, None

def generate_from_semap(
    semap_path,
    normal_image_path=None,
    output_path=None,
    ddim_steps=50,
    guidance_scale=9.0,
    batch_size=1,
    h=256,
    w=256,
    optimize_result=True,
    optimization_steps=2,
    preservation_ratio=100,
    reload_model=False,
    checkpoint_path=None,
    save_interval=200,
    cutoff_index=45,
    diffusion_preview=True,
    diffusion_preview_interval=None,
    discriminator_checkpoint=None,
    discriminator_weight=1.0,
):
    """Generate anomaly image from SeMaP with visualization of diffusion process"""
    
    if diffusion_preview_interval is None:
        diffusion_preview_interval = max(1, ddim_steps // 10)
    
    global model, ddim_sampler
    if reload_model and checkpoint_path is not None:
        reset_model_state()
    
    # 1. Load SeMaP
    semap = np.load(semap_path)
    
    if semap.shape[:2] != (h, w):
        semap = cv2.resize(semap, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 2. Extract mask from SeMaP
    anomaly_mask = extract_mask_from_semap(semap)
    
    pixel_mask = torch.from_numpy(anomaly_mask).float().to("cuda")
    pixel_mask = pixel_mask.unsqueeze(0).unsqueeze(0)
    
    latent_mask = cv2.resize(anomaly_mask, (32, 32), interpolation=cv2.INTER_NEAREST)
    latent_mask = torch.from_numpy(latent_mask).float().to("cuda")
    latent_mask = latent_mask.unsqueeze(0).unsqueeze(0)
    
    # 3. Load normal image
    normal_image_tensor = None
    if normal_image_path is not None:
        normal_image = cv2.imread(normal_image_path)
        if normal_image is None:
            print(f"Warning: Failed to load normal image from {normal_image_path}")
            optimize_result = False
        else: 
            normal_image = cv2.resize(normal_image, (w, h))
            normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
            normal_image = torch.from_numpy(normal_image).float().cuda() / 255.0
            normal_image = einops.rearrange(normal_image, 'h w c -> 1 c h w')
            normal_image_tensor = (normal_image - 0.5) / 0.5
    else:
        print("Warning: Normal image is required for starting diffusion from real image")
        return None, None, [], None, []
    
    # 4. Preprocess SeMaP
    if isinstance(semap, np.ndarray):
        if len(semap.shape) == 3 and semap.shape[-1] == 4:
            semap = np.transpose(semap, (2, 0, 1))
            semap = np.expand_dims(semap, 0)
    
    semap_tensor = torch.from_numpy(semap).float().cuda()
    
    # 5. Create empty context vector
    empty_context = torch.zeros((batch_size, 77, 768), device="cuda")
    
    # 6. Set conditions
    cond = {
        "c_concat": [semap_tensor],
        "c_crossattn": [empty_context],
        "empty_crossattn": [empty_context]
    }
    
    un_cond = {
        "c_concat": [semap_tensor],
        "c_crossattn": [empty_context],
        "empty_crossattn": [empty_context]
    }
    
    # 7. Set sampling shape
    shape = (4, h // 8, w // 8)
    
    # 8. Encode normal image to latent space
    if normal_image_tensor is not None:
        with torch.no_grad():
            encoder_posterior = model.encode_first_stage(normal_image_tensor)
            z = model.get_first_stage_encoding(encoder_posterior).detach()
    else:
        z = torch.randn((batch_size, *shape), device="cuda")
        
    # 9. Use blended sampling
    samples, blended_intermeid = ddim_sampler.sample_blended(
        ddim_steps, 
        batch_size, 
        shape, 
        cond, 
        mask=latent_mask,
        verbose=False, 
        eta=0.0,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=un_cond,
        org_mask=pixel_mask,
        init_image=normal_image_tensor,
        cutoff_index=cutoff_index,
    )
    
    # Process intermediate diffusion results
    diffusion_steps = []
    if diffusion_preview and 'x_inter' in blended_intermeid:
        x_inter = blended_intermeid['x_inter']
        step_indices = list(range(0, ddim_steps, diffusion_preview_interval))
        if (ddim_steps-1) not in step_indices:
            step_indices.append(ddim_steps-1)
        
        for idx in step_indices:
            if idx < len(x_inter):
                with torch.no_grad():
                    inter_image = model.decode_first_stage(x_inter[idx])
                    inter_image = tensor2img(inter_image)
                    diffusion_steps.append((idx, inter_image[0]))
    
    curr_latent = samples.clone().detach()
    
    # 10. Decode to get generated image
    with torch.no_grad():
        x_samples = model.decode_first_stage(samples)
    
    initial_samples = tensor2img(x_samples)
    
    optimization_progress = []
    
    # 11. Apply optimization if requested
    if optimize_result and normal_image_path is not None and normal_image_tensor is not None:
        # Load discriminator if checkpoint provided
        discriminator = None
        is_multiscale = False
        if discriminator_checkpoint is not None and os.path.exists(discriminator_checkpoint):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(discriminator_checkpoint, map_location=device)
                
                is_multiscale = checkpoint.get('multiscale', False)
                
                if is_multiscale:
                    discriminator = MultiScaleDiscriminator().to(device)
                else:
                    discriminator = SimpleDiscriminator().to(device)
                
                if "model_state_dict" in checkpoint:
                    discriminator.load_state_dict(checkpoint["model_state_dict"])
                else:
                    discriminator.load_state_dict(checkpoint)
                
                discriminator.eval()
                for param in discriminator.parameters():
                    param.requires_grad = False
                    
                print(f"Discriminator loaded from {discriminator_checkpoint}")
                print(f"Discriminator type: {'Multi-scale' if is_multiscale else 'Simple'}")
                
            except Exception as e:
                print(f"Warning: Failed to load discriminator: {e}")
                discriminator = None
        
        # Save original decoder state
        original_decoder_state = copy.deepcopy(model.first_stage_model.decoder.state_dict())
        
        torch.cuda.empty_cache()
        gc.collect()
        
        model.first_stage_model.decoder.requires_grad_(True)
        
        optimizer = torch.optim.AdamW(
            model.first_stage_model.decoder.parameters(), 
            lr=0.00001, 
            weight_decay=0.01 
        )
        
        fg_image = x_samples.clone().detach()
        bg_image = normal_image_tensor.clone().detach()
        
        initial_mean = fg_image.mean(dim=[2, 3]).detach()
        initial_std = fg_image.std(dim=[2, 3]).detach()
        
        def combined_loss(fg_image, bg_image, curr_latent, mask, preservation_ratio=100, 
                        model=None, discriminator=None, discriminator_weight=1.0, is_multiscale=False,
                        initial_mean=None, initial_std=None):
            """
            Returns: (total_loss, recon_loss, disc_loss, disc_score, color_loss, perceptual_loss, frequency_loss)
            """
            curr_latent_scaled = 1.0 / model.scale_factor * curr_latent
            curr_reconstruction = model.first_stage_model.decode(curr_latent_scaled)
            
            # Reconstruction loss
            recon_loss = (
                F.mse_loss(fg_image * mask, curr_reconstruction * mask) + 
                F.mse_loss(bg_image * (1 - mask), curr_reconstruction * (1 - mask)) * preservation_ratio
            )
            
            color_loss = torch.tensor(0.0, device=recon_loss.device)
            if initial_mean is not None and initial_std is not None:
                current_mean = curr_reconstruction.mean(dim=[2, 3])
                current_std = curr_reconstruction.std(dim=[2, 3])
                color_loss = F.mse_loss(current_mean, initial_mean) + F.mse_loss(current_std, initial_std)
            
            diff_x = torch.abs(curr_reconstruction[:, :, :-1, :] - curr_reconstruction[:, :, 1:, :])
            diff_y = torch.abs(curr_reconstruction[:, :, :, :-1] - curr_reconstruction[:, :, :, 1:])
            perceptual_loss = diff_x.mean() + diff_y.mean()
            
            frequency_loss = torch.tensor(0.0, device=recon_loss.device)
            eps = 1e-8
            alpha = 1.0
            
            curr_fft = torch.fft.fft2(curr_reconstruction, norm='ortho')
            fg_fft = torch.fft.fft2(fg_image, norm='ortho')
            
            diff = curr_fft - fg_fft
            mag_diff = torch.abs(diff) + eps
            
            weight = mag_diff ** alpha
            
            frequency_loss = (weight * mag_diff).mean()
            
            total_loss = recon_loss + 0.1 * color_loss + 0.0 * perceptual_loss + 10000 * frequency_loss
            disc_loss = torch.tensor(0.0, device=recon_loss.device)
            disc_score = torch.tensor(0.0, device=recon_loss.device)
            
            # Add discriminator loss if discriminator is available
            if discriminator is not None and discriminator_weight > 0:
                try:
                    # Convert to [0,1] range for discriminator
                    disc_input = torch.clamp((curr_reconstruction + 1.0) / 2.0, 0.0, 1.0)
                    
                    disc_score, _ = calculate_discriminator_score(discriminator, disc_input, is_multiscale)
                    
                    target_score = torch.ones_like(disc_score) * 0.8  
                    disc_loss = F.binary_cross_entropy(disc_score, target_score)
                    
                    disc_loss = torch.clamp(disc_loss, 0.0, 10.0)  
                    
                    adaptive_weight = discriminator_weight * min(1.0, recon_loss.item() / (disc_loss.item() + eps))
                    total_loss = total_loss + adaptive_weight * disc_loss
                    
                except Exception as e:
                    print(f"Warning: Error computing discriminator loss: {e}")
                    disc_loss = torch.tensor(0.0, device=recon_loss.device)
                    disc_score = torch.tensor(0.0, device=recon_loss.device)
            
            return total_loss, recon_loss, disc_loss, disc_score, color_loss, perceptual_loss, frequency_loss
        
        # Optimization loop
        print("Running online optimization with discriminator guidance...")
        print(f"Discriminator weight: {discriminator_weight}")
        print(f"Optimization steps: {optimization_steps}")
        
        try:
            for i in tqdm.tqdm(range(optimization_steps)):
                optimizer.zero_grad()
                
                if i % 50 == 0 and i > 0:
                    torch.cuda.empty_cache()
                
                total_loss, recon_loss, disc_loss, disc_score, color_loss, perceptual_loss, frequency_loss = combined_loss(
                    fg_image=fg_image,
                    bg_image=bg_image,
                    curr_latent=curr_latent,
                    mask=pixel_mask,
                    preservation_ratio=preservation_ratio,
                    model=model,
                    discriminator=discriminator,
                    discriminator_weight=discriminator_weight,
                    is_multiscale=is_multiscale,
                    initial_mean=initial_mean,
                    initial_std=initial_std
                )
                
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.first_stage_model.decoder.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                if i % save_interval == 0 or i == optimization_steps - 1:
                    with torch.no_grad():
                        curr_latent_scaled = 1.0 / model.scale_factor * curr_latent
                        optimized_image = model.first_stage_model.decode(curr_latent_scaled)
                        optimized_samples = tensor2img(optimized_image)
                        optimization_progress.append((i, optimized_samples[0]))
                        
                        print(f"\nStep {i}:")
                        print(f"  Total Loss: {total_loss.item():.6f}")
                        print(f"  Recon Loss: {recon_loss.item():.6f}")
                        print(f"  Color Loss: {color_loss.item():.6f}")
                        print(f"  Perceptual Loss: {perceptual_loss.item():.6f}")
                        print(f"  Frequency Loss: {frequency_loss.item():.6f}")
                        if discriminator is not None:
                            print(f"  Disc Loss: {disc_loss.item():.6f}")
                            print(f"  Disc Score: {disc_score.item():.6f}")
                        print("-" * 50)
                
                elif i % (save_interval // 2) == 0:
                    print(f"Step {i}: Total={total_loss.item():.4f}, "
                          f"Recon={recon_loss.item():.4f}, "
                          f"Freq={frequency_loss.item():.4f}, "
                          f"Disc={disc_loss.item():.4f}, "
                          f"Score={disc_score.item():.4f}")
                                
        except RuntimeError as e:
            print(f"Error during optimization: {str(e)}")
            print("Continuing with unoptimized result...")
        
        # Get final optimized result
        with torch.no_grad():
            curr_latent_scaled = 1.0 / model.scale_factor * curr_latent
            optimized_image = model.first_stage_model.decode(curr_latent_scaled)
            final_optimized_samples = tensor2img(optimized_image)
        
        # Restore original decoder state
        model.first_stage_model.decoder.load_state_dict(original_decoder_state)
        model.first_stage_model.decoder.requires_grad_(False)
        
        # Clean up
        if discriminator is not None:
            del discriminator
        del optimizer, fg_image, bg_image, curr_latent_scaled, original_decoder_state
        torch.cuda.empty_cache()
        gc.collect()
    else:
        final_optimized_samples = initial_samples
        optimization_progress = [(i*save_interval, initial_samples[0]) for i in range(optimization_steps // save_interval + 1)]
    
    return initial_samples[0], final_optimized_samples[0], optimization_progress, anomaly_mask, diffusion_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate anomalies for REALIAD dataset using AdaBLDM')
    parser.add_argument('--item', type=str, nargs='+', required=True, help='One or more REALIAD Item types (e.g., REALIAD1 REALIADA)')
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--guidance_scale", type=float, default=9.0, help="Guidance scale")
    parser.add_argument("--cutoff_index", type=int, default=45, help="Cutoff index for blended sampling")
    parser.add_argument("--output_root", type=str, default="./output/realiad", help="Root directory for saving results")

    parser.add_argument("--discriminator_checkpoint", type=str, default=None,
                       help="Path to discriminator checkpoint for guidance (currently, path is derived from item)")
    parser.add_argument("--discriminator_weight", type=float, default=0.1,  
                       help="Weight for discriminator loss in optimization")
    parser.add_argument("--enable_optimization", type=bool, default=True,
                       help="Enable online optimization with discriminator guidance")
    parser.add_argument("--optimization_steps", type=int, default=50,
                       help="Number of optimization steps")

    args = parser.parse_args()

    ddim_steps = args.ddim_steps
    cutoff_index = args.cutoff_index
    guidance_scale = args.guidance_scale
    REALIAD_DATA_ROOT = "./pix2pix/output_realiad"

    for current_item_name in args.item:
        print(f"\nProcessing item: {current_item_name}\n{'='*30}")

        # TODO: Check if the checkpoint exists
        checkpoint_path = f"./checkpoints_realiad/{current_item_name}_cs/epoch=19.ckpt"
        print(f"Loading checkpoint for {current_item_name}: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found at {checkpoint_path} for item {current_item_name}. Skipping this item.")
            continue
        
        model, ddim_sampler = get_fresh_model(checkpoint_path)

        # TODO: Check if the discriminator checkpoint exists
        item_specific_discriminator_path = f"./GAN/checkpoints_discriminator/realiad/{current_item_name}/multiscale_discriminator_best.pth"
        if args.enable_optimization and not os.path.exists(item_specific_discriminator_path):
            print(f"Warning: Discriminator checkpoint for {current_item_name} not found at {item_specific_discriminator_path}. Optimization will proceed without discriminator guidance for this item if it relies on this specific path.")

        item_data_dir = os.path.join(REALIAD_DATA_ROOT, current_item_name)
        prompt_json_path = os.path.join(item_data_dir, "prompt.json")

        if not os.path.isdir(item_data_dir):
            print(f"Error: Item data directory not found: {item_data_dir} for item {current_item_name}. Skipping this item.")
            continue
        if not os.path.exists(prompt_json_path):
            print(f"Error: prompt.json not found at {prompt_json_path} for item {current_item_name}. Skipping this item.")
            continue

        output_dir = os.path.join(args.output_root, current_item_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output for {current_item_name} will be saved to: {output_dir}")

        test_samples_data = []
        try:
            with open(prompt_json_path, 'r') as f:
                for line in f:
                    test_samples_data.append(json.loads(line))
        except Exception as e:
            print(f"Error reading prompt.json for {current_item_name}: {e}. Skipping this item.")
            continue

        if not test_samples_data:
            print(f"Error: No samples found in {prompt_json_path} for item '{current_item_name}'. Skipping this item.")
            continue

        print(f"Found {len(test_samples_data)} samples to process for item '{current_item_name}'.")

        for i, sample_data in enumerate(tqdm.tqdm(test_samples_data, desc=f"Processing {current_item_name}")):
            semap_relative_path = sample_data.get("semap")
            source_relative_path = sample_data.get("source")

            if not semap_relative_path or not source_relative_path:
                print(f"Warning: Missing 'semap' or 'source' path in prompt.json entry: {sample_data}. Skipping.")
                continue

            semap_path = os.path.join(item_data_dir, semap_relative_path)
            normal_image_path = os.path.join(item_data_dir, source_relative_path)
            
            sample_id = os.path.splitext(os.path.basename(semap_relative_path))[0]

            if not os.path.exists(semap_path):
                print(f"Warning: SeMaP file not found for sample {sample_id} at {semap_path}. Skipping.")
                continue
            if not os.path.exists(normal_image_path):
                print(f"Warning: Normal image not found for sample {sample_id} at {normal_image_path}. Skipping.")
                continue
            
            final_output_image_path = os.path.join(output_dir, f"{sample_id}.png")

            initial_image, final_optimized_image, optimization_progress, anomaly_mask, diffusion_steps = generate_from_semap(
                semap_path=semap_path,
                normal_image_path=normal_image_path,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
                optimize_result=args.enable_optimization,
                optimization_steps=args.optimization_steps if args.enable_optimization else 0,
                cutoff_index=cutoff_index,
                diffusion_preview=False,
                batch_size=1,
                discriminator_checkpoint=item_specific_discriminator_path if args.enable_optimization else None,
                discriminator_weight=args.discriminator_weight,
            )

            if initial_image is not None:
                try:
                    image_to_save = final_optimized_image if args.enable_optimization and final_optimized_image is not None else initial_image
                    cv2.imwrite(final_output_image_path, cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"Error saving image {final_output_image_path}: {e}")
            else:
                print(f"Failed to generate image for sample {sample_id} (item: {current_item_name}).")

            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        print(f"\nBatch processing complete for item '{current_item_name}'. Results saved to {output_dir}")
        
        if 'model' in globals() and model is not None:
            del model
        if 'ddim_sampler' in globals() and ddim_sampler is not None:
            del ddim_sampler
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll specified items processed.")