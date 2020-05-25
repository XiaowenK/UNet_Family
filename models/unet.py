
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)



def maxpool2x2(x):
    mp = nn.MaxPool2d(kernel_size=2, stride=2)
    x = mp(x)
    return x



class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.encoderblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.encoderblock(x)
        return x



class CenterBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CenterBlock, self).__init__()
        mid_channels = int(in_channels*2)

        self.centerblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels*2), mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.centerblock(x)
        return x



class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        mid_channels = int(in_channels/2)

        self.decoderblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.decoderblock(x)
        return x



class FinalBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FinalBlock, self).__init__()
        mid_channels = int(in_channels/2)

        self.finalblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            )

    def forward(self, x):
        x = self.finalblock(x)
        return x



class UNet(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        # Encoder part.
        self.encoder1 = EncoderBlock(in_channels=3, out_channels=64)
        self.encoder2 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder3 = EncoderBlock(in_channels=128, out_channels=256)
        self.encoder4 = EncoderBlock(in_channels=256, out_channels=512)
        # Center part.
        self.center = CenterBlock(in_channels=512, out_channels=512)
        # Decoder part.
        self.decoder4 = DecoderBlock(in_channels=1024, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=512, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=256, out_channels=64)
        # Final part.decoder2
        self.final = FinalBlock(in_channels=128, out_channels=num_class)

    def forward(self, x):
        # Encoding, compressive pathway.
        out_encoder1 = self.encoder1(x)
        out_encoder2 = self.encoder2(maxpool2x2(out_encoder1))
        out_encoder3 = self.encoder3(maxpool2x2(out_encoder2))
        out_encoder4 = self.encoder4(maxpool2x2(out_encoder3))
        # Decoding, expansive pathway.
        out_center = self.center(maxpool2x2(out_encoder4))
        out_decoder4 = self.decoder4(torch.cat((out_center, out_encoder4), 1))
        out_decoder3 = self.decoder3(torch.cat((out_decoder4, out_encoder3), 1))
        out_decoder2 = self.decoder2(torch.cat((out_decoder3, out_encoder2), 1))
        out_final = self.final(torch.cat((out_decoder2, out_encoder1), 1))
        return out_final


