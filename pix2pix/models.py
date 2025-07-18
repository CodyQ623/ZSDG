import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            # Use BatchNorm instead of InstanceNorm if batch size > 1, otherwise InstanceNorm is fine
            layers.append(nn.BatchNorm2d(out_size)) # Changed from InstanceNorm
            # layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
             # Use BatchNorm instead of InstanceNorm if batch size > 1
            nn.BatchNorm2d(out_size), # Changed from InstanceNorm
            # nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    # Modified __init__ to accept vector_dim
    def __init__(self, in_channels=1, out_channels=1, vector_dim=6):
        super(GeneratorUNet, self).__init__()

        self.vector_dim = vector_dim
        # Input to down1 is concatenation of image and expanded vector
        input_dim_down1 = in_channels + vector_dim

        self.down1 = UNetDown(input_dim_down1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # Note: UNetUp input sizes need adjustment due to skip connections
        # up1 input is from down8 output (512)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        # up2 input is from up1 output (512) + skip d7 (512) = 1024
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        # up3 input is from up2 output (512) + skip d6 (512) = 1024
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        # up4 input is from up3 output (512) + skip d5 (512) = 1024
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        # up5 input is from up4 output (512) + skip d4 (512) = 1024
        self.up5 = UNetUp(1024, 256)
        # up6 input is from up5 output (256) + skip d3 (256) = 512
        self.up6 = UNetUp(512, 128)
        # up7 input is from up6 output (128) + skip d2 (128) = 256
        self.up7 = UNetUp(256, 64)

        # Final layer input is from up7 output (64) + skip d1 (64) = 128
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            # Use Tanh for outputs in range [-1, 1] to match normalization
            # nn.Sigmoid(), # Changed from Sigmoid
            nn.Tanh(), # Changed back to Tanh
        )

    # Modified forward to accept prompt_vec
    def forward(self, x, prompt_vec):
        # x: source mask (batch_size, in_channels, H, W)
        # prompt_vec: (batch_size, vector_dim)

        # Reshape and expand vector to match image spatial dimensions
        batch_size, _, H, W = x.size()
        vec_expanded = prompt_vec.view(batch_size, self.vector_dim, 1, 1).expand(batch_size, self.vector_dim, H, W)

        # Concatenate image and expanded vector
        net_input = torch.cat((x, vec_expanded), 1) # Shape: (batch_size, in_channels + vector_dim, H, W)

        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(net_input) # Use concatenated input
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
     # Modified __init__ to accept vector_dim
    def __init__(self, in_channels=1, vector_dim=6):
        super(Discriminator, self).__init__()
        self.vector_dim = vector_dim

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                 # Use BatchNorm instead of InstanceNorm if batch size > 1
                layers.append(nn.BatchNorm2d(out_filters)) # Changed from InstanceNorm
                # layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Input channels: source mask (in_channels) + target mask (in_channels) + vector (vector_dim)
        total_in_channels = in_channels * 2 + vector_dim

        self.model = nn.Sequential(
            *discriminator_block(total_in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
            # No Sigmoid here, output raw scores for GAN loss (e.g., MSELoss or BCEWithLogitsLoss)
        )

    # Modified forward to accept prompt_vec
    def forward(self, img_A, img_B, prompt_vec):
        # img_A: source mask (batch_size, in_channels, H, W)
        # img_B: real/fake target mask (batch_size, in_channels, H, W)
        # prompt_vec: (batch_size, vector_dim)

        # Reshape and expand vector
        batch_size, _, H, W = img_A.size()
        vec_expanded = prompt_vec.view(batch_size, self.vector_dim, 1, 1).expand(batch_size, self.vector_dim, H, W)

        # Concatenate source mask, target mask, and expanded vector by channels
        img_input = torch.cat((img_A, img_B, vec_expanded), 1)
        return self.model(img_input)
