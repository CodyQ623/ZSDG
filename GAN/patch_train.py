import os
import glob
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Configuration parameters
class Config:
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ratio = 0.8
    test_ratio = 0.2
    checkpoint_dir = "./checkpoints_discriminator/realiad"
    
    use_multiscale = True
    area_weighted = True
    
    balance_strategy = 'undersample' 

class DefectDiscriminatorDataset(Dataset):
    def __init__(self, real_path, fake_path, item_name, split="train", config=None, transform=None):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
            
        self.split = split
        self.config = config or Config()
        
        real_pattern = os.path.join(real_path, "data", "mvtec_reorg", item_name, "combined", "test", "ng", "*.png")
        real_images = sorted(glob.glob(real_pattern))

        fake_pattern = os.path.join(fake_path, "output", "realiad", item_name, "*.png")
        fake_images = sorted(glob.glob(fake_pattern))
        
        print(f"Found {len(real_images)} real images and {len(fake_images)} fake images for {item_name}")
        
        if self.config.balance_strategy == 'undersample':
            min_count = min(len(real_images), len(fake_images))
            real_images = real_images[:min_count]
            fake_images = random.sample(fake_images, min_count)
            print(f"After balancing: {len(real_images)} real images, {len(fake_images)} fake images")
        elif self.config.balance_strategy == 'oversample':
            max_count = max(len(real_images), len(fake_images))
            if len(real_images) < max_count:
                multiplier = max_count // len(real_images) + 1
                real_images = (real_images * multiplier)[:max_count]
            fake_images = fake_images[:max_count]
            print(f"After balancing: {len(real_images)} real images, {len(fake_images)} fake images")
        
        train_size = int(len(real_images) * self.config.train_ratio)
        
        if split == "train":
            self.real_images = real_images[:train_size]
            self.fake_images = fake_images[:train_size]
        else:  # test
            self.real_images = real_images[train_size:]
            self.fake_images = fake_images[train_size:]
        
        self.all_images = [(path, 1) for path in self.real_images] + [(path, 0) for path in self.fake_images]
        random.shuffle(self.all_images)
        
        print(f"{split} dataset: {len(self.all_images)} images "
              f"({len(self.real_images)} real, {len(self.fake_images)} fake)")
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_path, label = self.all_images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, use_sigmoid=True):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True)
        )
        
        self.head_scale1 = nn.Sequential(
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.head_scale2 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*2, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.head_scale3 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.head_scale4 = nn.Sequential(
            nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.head_scale5 = nn.Sequential(
            nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.sigmoid = nn.Sigmoid() if use_sigmoid else nn.Identity()
    
    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat5 = self.layer5(feat4)
        
        score1 = self.sigmoid(self.head_scale1(feat1))
        score2 = self.sigmoid(self.head_scale2(feat2))
        score3 = self.sigmoid(self.head_scale3(feat3))
        score4 = self.sigmoid(self.head_scale4(feat4))
        score5 = self.sigmoid(self.head_scale5(feat5))
        
        return {
            'scale1': score1,
            'scale2': score2,
            'scale3': score3,
            'scale4': score4,
            'scale5': score5
        }

class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        features = self.main(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        output = self.fc(features)
        return self.sigmoid(output)

class MultiScaleLoss(nn.Module):
    def __init__(self, area_weighted=True):
        super(MultiScaleLoss, self).__init__()
        self.area_weighted = area_weighted
        self.bce = nn.BCELoss(reduction='none')
        
        self.scale_weights = {
            'scale1': 0.1,
            'scale2': 0.1,
            'scale3': 0.2,
            'scale4': 0.2,
            'scale5': 0.4
        }
    
    def forward(self, outputs, target):
        scale_areas = {}
        total_area = 0
        for scale_name, score in outputs.items():
            area = score.shape[2] * score.shape[3]
            scale_areas[scale_name] = area
            total_area += area
        
        area_weights = {}
        for scale_name, area in scale_areas.items():
            area_weights[scale_name] = area / total_area
        
        weighted_loss = torch.zeros_like(target)
        for scale_name, score in outputs.items():
            target_expanded = target.view(-1, 1, 1, 1).expand_as(score)
            patch_loss = self.bce(score, target_expanded)
            sample_loss = patch_loss.mean(dim=[1, 2, 3])
            
            if self.area_weighted:
                weight = self.scale_weights[scale_name] * area_weights[scale_name]
            else:
                weight = self.scale_weights[scale_name]
            
            weighted_loss += sample_loss * weight
        
        return weighted_loss.mean()

def train_discriminator(real_path, fake_path, item_name, config):
    save_dir = f"{config.checkpoint_dir}/{item_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    if config.use_multiscale:
        discriminator = MultiScaleDiscriminator().to(config.device)
        criterion = MultiScaleLoss(area_weighted=config.area_weighted)
    else:
        discriminator = SimpleDiscriminator().to(config.device)
        criterion = nn.BCELoss()
    
    optimizer = optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    
    train_dataset = DefectDiscriminatorDataset(real_path, fake_path, item_name, "train", config)
    test_dataset = DefectDiscriminatorDataset(real_path, fake_path, item_name, "test", config)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    train_losses = []
    train_accuracies = []
    
    best_train_loss = float('inf')
    for epoch in range(config.num_epochs):
        discriminator.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
            
            if config.use_multiscale:
                outputs = discriminator(images)
                loss = criterion(outputs, labels)
                predicted = (outputs['scale3'].mean(dim=[1, 2, 3]) > 0.5).float()
            else:
                outputs = discriminator(images).squeeze()
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({"loss": train_loss/(pbar.n+1), "acc": 100*correct/total})
        
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            model_type = "multiscale" if config.use_multiscale else "simple"
            torch.save({
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'epoch': epoch,
                'item_name': item_name,
                'multiscale': config.use_multiscale,
            }, os.path.join(config.checkpoint_dir, item_name, f"{model_type}_discriminator_best.pth"))
    
    discriminator.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(config.device), labels.to(config.device)
            
            if config.use_multiscale:
                outputs = discriminator(images)
                loss = criterion(outputs, labels)
                probs = outputs['scale3'].mean(dim=[1, 2, 3])
                predicted = (probs > 0.5).float()
                all_probs.extend(probs.cpu().numpy().tolist())
            else:
                outputs = discriminator(images).squeeze()
                loss = criterion(outputs, labels)
                predicted = (outputs > 0.5).float()
                all_probs.extend(outputs.cpu().numpy().tolist())
            
            all_labels.extend(labels.cpu().numpy().tolist())
            test_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    print(f"\nTest Results - Loss: {test_loss/len(test_loader):.4f}, Acc: {test_accuracy:.2f}%")
    
    from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
    
    binary_preds = [1 if p > 0.5 else 0 for p in all_probs]
    
    precision = precision_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)
    
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
    except:
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train defect discriminator model")
    parser.add_argument('--item', type=str, required=True, help='Item name (e.g., bottle, cable, etc.)')
    parser.add_argument('--real_path', type=str, default='../', help='Path to real images dataset')
    parser.add_argument('--fake_path', type=str, default='../', help='Path to fake images dataset')
    parser.add_argument('--single_scale', action='store_true', help='Use single scale discriminator')
    parser.add_argument('--no_area_weighted', action='store_true', help='Disable area-weighted loss')
    parser.add_argument('--balance_strategy', type=str, choices=['undersample', 'oversample'], 
                       default='undersample', help='Strategy for handling data imbalance')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    
    args = parser.parse_args()
    
    config = Config()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.use_multiscale = not args.single_scale
    config.area_weighted = not args.no_area_weighted
    config.balance_strategy = args.balance_strategy
    
    print(f"Training configuration:")
    print(f"  Item: {args.item}")
    print(f"  Real images path: {args.real_path}")
    print(f"  Fake images path: {args.fake_path}")
    print(f"  Model type: {'Multi-scale' if config.use_multiscale else 'Single-scale'}")
    print(f"  Balance strategy: {config.balance_strategy}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    
    train_discriminator(args.real_path, args.fake_path, args.item, config)

if __name__ == "__main__":
    main()