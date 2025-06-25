import torch.nn as nn

class DeeperUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Encoder
        self.conv1 = self.conv_block(in_channels, 64)
        self.down1 = self.down_block(64, 128)
        self.down2 = self.down_block(128, 256)
        self.down3 = self.down_block(256, 512)
        self.down4 = self.down_block(512, 1024)

        # Bottleneck
        self.middle = self.conv_block(1024, 1024)

        # Decoder
        self.up1 = self.up_block(1024, 512)
        self.up2 = self.up_block(512, 256)
        self.up3 = self.up_block(256, 128)
        self.up4 = self.up_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, 
                     dilation=2, groups=min(in_channels, 4), bias=False),
            nn.GroupNorm(out_channels // 2, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                     dilation=1, groups=min(out_channels, 4), bias=False),
            nn.GroupNorm(out_channels // 2, out_channels, affine=False),
            nn.ReLU(),
        )

    def down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                     dilation=1, groups=1, bias=False),
            nn.Dropout2d(0.2),
            self.conv_block(out_channels, out_channels),
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                     dilation=1, groups=min(in_channels, 4), bias=False),
            self.conv_block(out_channels, out_channels),
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        c5 = self.down4(c4)
        
        m = self.middle(c5)
        
        u1 = self.up1(m)
        u2 = self.up2(u1 + c4)
        u3 = self.up3(u2 + c3)
        u4 = self.up4(u3 + c2)
        
        return self.final(u4 + c1)