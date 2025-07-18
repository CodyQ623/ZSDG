import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from patch_train import MultiScaleDiscriminator, SimpleDiscriminator

def test_image(checkpoint_path, image_path, image_size=256, save_visualization=False, output_dir=None, show_heatmaps=False):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    is_multiscale = checkpoint.get('multiscale', False)
    
    if is_multiscale:
        model = MultiScaleDiscriminator().to(device)
        print("Using MultiScale Discriminator")
    else:
        model = SimpleDiscriminator().to(device)
        print("Using Simple Discriminator")
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    print(f"Image shape: {image_tensor.shape}")
    print(f"Image range: [{image_tensor.min().item():.4f}, {image_tensor.max().item():.4f}]")
    
    with torch.no_grad():
        if is_multiscale:
            outputs = model(image_tensor)
            
            scale_stats = {}
            for scale_name, score_map in outputs.items():
                scale_stats[scale_name] = {
                    'min': score_map.min().item(),
                    'max': score_map.max().item(),
                    'mean': score_map.mean().item(),
                    'std': score_map.std().item(),
                    'shape': score_map.shape[2:],
                    'map': score_map[0, 0].cpu().numpy()  
                }
            
            total_weight = 0
            weighted_score = 0

            receptive_fields = {
                'scale1': 16,  
                'scale2': 34,
                'scale3': 70, 
                'scale4': 142,
                'scale5': 286   
            }

            for scale_name, stats in scale_stats.items():

                weight = receptive_fields[scale_name]
                
                total_weight += weight
                weighted_score += stats['mean'] * weight
                
                scale_stats[scale_name]['weight'] = weight
                scale_stats[scale_name]['rf_size'] = receptive_fields[scale_name]

            overall_score = weighted_score / total_weight if total_weight > 0 else 0.5
            prediction = "Real" if overall_score >= 0.5 else "Fake"
        else:
            output = model(image_tensor)
            overall_score = output.mean().item()
            prediction = "Real" if overall_score >= 0.5 else "Fake"
            scale_stats = None
    
    if is_multiscale:
        print("\nDetailed scores by scale:")
        
        sorted_scales = sorted(scale_stats.items(), 
                            key=lambda x: x[1]['rf_size'], 
                            reverse=True)
        
        for scale_name, stats in sorted_scales:

            contribution = (stats['mean'] * stats['weight'] / total_weight) * 100
            weight_percent = (stats['weight'] / total_weight) * 100
            
            print(f"  {scale_name} ({stats['shape'][0]}x{stats['shape'][1]}):")
            print(f"    Mean: {stats['mean']:.6f}")
            print(f"    Receptive Field: {stats['rf_size']}x{stats['rf_size']} pixels")
            print(f"    Weight: {weight_percent:.1f}% (contribution: {contribution:.1f}%)")
            print(f"    Min/Max: {stats['min']:.4f}/{stats['max']:.4f}")
            print(f"    Std: {stats['std']:.6f}")
    

    print(f"\nImage: {image_path}")
    print(f"Overall Prediction: {prediction} (Score: {overall_score:.6f})")
    
    if save_visualization:
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)
        
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        if is_multiscale:

            sorted_scales = sorted(scale_stats.items(), 
                                key=lambda x: x[1]['rf_size'], 
                                reverse=True)
            

            plt.figure(figsize=(18, 12))
            
            plt.subplot(2, 3, 1)
            plt.imshow(original_image)
            plt.title(f"Input Image\nOverall: {overall_score:.4f} ({prediction})", 
                    fontsize=12, color='green' if prediction == 'Real' else 'red')
            plt.axis('off')
            
            for i, (scale_name, stats) in enumerate(sorted_scales):
                plt.subplot(2, 3, i+2)
                

                heat_map = stats['map']
                
                im = plt.imshow(heat_map, cmap='jet', vmin=0, vmax=1)
                plt.colorbar(im, fraction=0.046, pad=0.04)
                

                weight_percent = (stats['weight'] / total_weight) * 100
                contribution = (stats['mean'] * stats['weight'] / total_weight) * 100
                
                title = f"{scale_name}: {stats['mean']:.4f}\n"
                title += f"RF: {stats['rf_size']}×{stats['rf_size']} px\n"
                title += f"Weight: {weight_percent:.1f}%\n"
                title += f"Contribution: {contribution:.1f}%"
                
                plt.title(title, fontsize=10)
                
                plt.xlabel(f"Resolution: {stats['shape'][1]}×{stats['shape'][0]}")
            
            plt.suptitle(f"Multi-Scale Discriminator Results\n"
                    f"Overall: {overall_score:.4f} ({prediction})", 
                    fontsize=16, y=0.98,
                    color='green' if prediction == 'Real' else 'red')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])  
            plt.savefig(os.path.join(output_dir, f"{image_name}_multiscale_scores.png"), dpi=150)
            plt.close()
            
        else:
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title(f"Input Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            bars = plt.bar(['Fake', 'Real'], [1-overall_score, overall_score])
            bars[0].set_color('red')
            bars[1].set_color('green')
            plt.ylim(0, 1)
            plt.title(f"Prediction: {prediction} ({overall_score:.6f})",
                    color='green' if prediction == 'Real' else 'red')
            
            plt.text(bars[0].get_x() + bars[0].get_width()/2, 
                    1-overall_score + 0.02, 
                    f"{1-overall_score:.4f}", 
                    ha='center', color='white' if 1-overall_score > 0.3 else 'black')
            
            plt.text(bars[1].get_x() + bars[1].get_width()/2, 
                    overall_score + 0.02, 
                    f"{overall_score:.4f}", 
                    ha='center', color='white' if overall_score > 0.3 else 'black')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{image_name}_prediction.png"), dpi=150)
            plt.close()
    
    if show_heatmaps and is_multiscale:

        sorted_scales = sorted(scale_stats.items(), 
                            key=lambda x: x[1]['rf_size'], 
                            reverse=True)
        
        n_scales = len(sorted_scales)
        n_cols = min(3, n_scales) 
        n_rows = (n_scales + n_cols - 1) // n_cols  
        
        plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        for i, (scale_name, stats) in enumerate(sorted_scales):
            plt.subplot(n_rows, n_cols, i+1)
            
            heat_map = stats['map']
            
            im = plt.imshow(heat_map, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            
            weight_percent = (stats['weight'] / total_weight) * 100
            contribution = (stats['mean'] * stats['weight'] / total_weight) * 100
            
            title = f"{scale_name}: {stats['mean']:.4f}\n"
            title += f"RF: {stats['rf_size']}×{stats['rf_size']} px\n"
            title += f"Resolution: {stats['shape'][1]}×{stats['shape'][0]}"
            
            plt.title(title, fontsize=10)
            plt.axis('on')
        
        plt.suptitle(f"Discriminator Score Heatmaps - Overall: {overall_score:.4f}", 
                    fontsize=14, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        plt.show() 
            
    return overall_score, scale_stats if is_multiscale else None

def main():
    parser = argparse.ArgumentParser(description="Test discriminator on a single image")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_discriminator/realiad/zipper/multiscale_discriminator_best.pth", help="Path to the discriminator checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to test")
    parser.add_argument("--size", type=int, default=256, help="Image size for processing")
    parser.add_argument("--visualize", action="store_true", help="Save visualization of the results")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save visualization results")
    parser.add_argument("--show_heatmaps", action="store_true", help="Show heatmaps for each scale (no saving)")
    
    args = parser.parse_args()
    
    test_image(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        image_size=args.size,
        save_visualization=args.visualize,
        output_dir=args.output_dir,
        show_heatmaps=args.show_heatmaps,
    )

if __name__ == "__main__":
    main()
