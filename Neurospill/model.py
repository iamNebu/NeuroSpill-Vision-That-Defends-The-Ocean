import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Standard double convolution block with BatchNorm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.conv(x)

class AttentionGate(nn.Module):
    """Spatial attention gate to weight skip connection features."""
    def __init__(self, Fg, Fl, Fi):
        super().__init__()
        self.Wg = nn.Conv2d(Fg, Fi, 1)
        self.Wx = nn.Conv2d(Fl, Fi, 1)
        self.psi = nn.Conv2d(Fi, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, g, x):
        # g: gating signal from lower layer
        # x: skip connection from encoder
        psi = self.relu(self.Wg(g) + self.Wx(x))
        psi = self.sigmoid(self.psi(psi))
        return x * psi

class AttentionUNetPP(nn.Module):
    """Attention U-Net++ Architecture for Oil Spill Segmentation."""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        f = [64, 128, 256, 512]
        
        # Encoder
        self.pool = nn.MaxPool2d(2)
        self.c0 = ConvBlock(in_channels, f[0])
        self.c1 = ConvBlock(f[0], f[1])
        self.c2 = ConvBlock(f[1], f[2])
        self.c3 = ConvBlock(f[2], f[3])
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        self.att2 = AttentionGate(f[3], f[2], f[1])
        self.d2 = ConvBlock(f[3] + f[2], f[2])
        
        self.att1 = AttentionGate(f[2], f[1], f[0])
        self.d1 = ConvBlock(f[2] + f[1], f[1])
        
        self.att0 = AttentionGate(f[1], f[0], f[0] // 2)
        self.d0 = ConvBlock(f[1] + f[0], f[0])
        
        self.final = nn.Conv2d(f[0], out_channels, 1)
        
    def forward(self, x):
        # Encoder Pass
        x0 = self.c0(x)
        x1 = self.c1(self.pool(x0))
        x2 = self.c2(self.pool(x1))
        x3 = self.c3(self.pool(x2))
        
        # Decoder Pass with Attention Skip Connections
        u2 = self.up(x3)
        x2d = self.d2(torch.cat([u2, self.att2(u2, x2)], dim=1))
        
        u1 = self.up(x2d)
        x1d = self.d1(torch.cat([u1, self.att1(u1, x1)], dim=1))
        
        u0 = self.up(x1d)
        x0d = self.d0(torch.cat([u0, self.att0(u0, x0)], dim=1))
        
        return self.final(x0d)