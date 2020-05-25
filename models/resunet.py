
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        mid_channels = int(in_channels / 2)

        self.decoderblock = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.decoderblock(x)
        return x



class FinalBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(FinalBlock, self).__init__()
        self.finalblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x):
        x = self.finalblock(x)
        return x



# class ResUNet50(nn.Module):

#     def __init__(self, num_class, pretrained=False):
#         super().__init__()
#         self.resnet = torchvision.models.resnet50(pretrained=pretrained)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # Encoder part.
#         self.encoder1 = nn.Sequential(
#             self.resnet.conv1,
#             self.resnet.bn1,
#             nn.ReLU(inplace=True),
#             )
#         self.encoder2 = self.resnet.layer1
#         self.encoder3 = self.resnet.layer2
#         self.encoder4 = self.resnet.layer3
#         self.encoder5 = self.resnet.layer4
#         # Decoder part.
#         self.decoder5 = DecoderBlock(in_channels=2048, out_channels=1024)
#         self.decoder4 = DecoderBlock(in_channels=1024, out_channels=512)
#         self.decoder3 = DecoderBlock(in_channels=512, out_channels=256)
#         self.decoder2 = DecoderBlock(in_channels=256, out_channels=64)
#         # Final part.
#         self.final = FinalBlock(in_channels=64, mid_channels=32, out_channels=num_class)

#     def forward(self, x):
#         # Encoding, compressive pathway.
#         out_encoder1 = self.encoder1(x)
#         out_encoder2 = self.encoder2(self.maxpool(out_encoder1))
#         out_encoder3 = self.encoder3(out_encoder2)
#         out_encoder4 = self.encoder4(out_encoder3)
#         out_encoder5 = self.encoder5(out_encoder4)
#         # Decoding, expansive pathway.
#         out_decoder5 = self.decoder5(out_encoder5)
#         out_decoder4 = self.decoder4(out_decoder5+out_encoder4)
#         out_decoder3 = self.decoder3(out_decoder4+out_encoder3)
#         out_decoder2 = self.decoder2(out_decoder3+out_encoder2)
#         out_final = self.final(out_decoder2+out_encoder1)
#         return out_final



class ResUNet50(nn.Module):

    def __init__(self, num_class, pretrained=False):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Encoder part.
        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            nn.ReLU(inplace=True),
            )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        # Decoder part.
        self.decoder5 = DecoderBlock(in_channels=2048, out_channels=1024)
        self.decoder4 = DecoderBlock(in_channels=1024, out_channels=512)
        self.decoder3 = DecoderBlock(in_channels=512, out_channels=256)
        self.decoder2 = DecoderBlock(in_channels=256, out_channels=64)
        # Final part.
        self.final = FinalBlock(in_channels=256, mid_channels=128, out_channels=num_class)

    def forward(self, x):
        # Encoding, compressive pathway.
        out_encoder1 = self.encoder1(x)
        out_encoder2 = self.encoder2(out_encoder1)
        out_encoder3 = self.encoder3(out_encoder2)
        out_encoder4 = self.encoder4(out_encoder3)
        out_encoder5 = self.encoder5(out_encoder4)
        # Decoding, expansive pathway.
        out_decoder5 = self.decoder5(out_encoder5)
        out_decoder4 = self.decoder4(out_decoder5+out_encoder4)
        out_decoder3 = self.decoder3(out_decoder4+out_encoder3)
        out_final = self.final(out_decoder3+out_encoder2)
        return out_final



class ResUNet101(nn.Module):

    def __init__(self, num_class, pretrained=False):
        super().__init__()
        self.resnet = torchvision.models.resnet101(pretrained=pretrained)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Encoder part.
        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            nn.ReLU(inplace=True),
            )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        # Decoder part.
        self.decoder5 = DecoderBlock(in_channels=2048, out_channels=1024)
        self.decoder4 = DecoderBlock(in_channels=1024, out_channels=512)
        self.decoder3 = DecoderBlock(in_channels=512, out_channels=256)
        self.decoder2 = DecoderBlock(in_channels=256, out_channels=64)
        # Final part.
        self.final = FinalBlock(in_channels=256, mid_channels=128, out_channels=num_class)

    def forward(self, x):
        # Encoding, compressive pathway.
        out_encoder1 = self.encoder1(x)
        out_encoder2 = self.encoder2(out_encoder1)
        out_encoder3 = self.encoder3(out_encoder2)
        out_encoder4 = self.encoder4(out_encoder3)
        out_encoder5 = self.encoder5(out_encoder4)
        # Decoding, expansive pathway.
        out_decoder5 = self.decoder5(out_encoder5)
        out_decoder4 = self.decoder4(out_decoder5+out_encoder4)
        out_decoder3 = self.decoder3(out_decoder4+out_encoder3)
        out_final = self.final(out_decoder3+out_encoder2)
        return out_final


