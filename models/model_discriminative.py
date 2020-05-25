
__author__ = "Xiaowen"

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



# class Discriminator(nn.Module):

#     def __init__(self, nc=4, ndf=16):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             # input size: 512*512, 544*448, 1024*640
#             # ---- Conv Layer 1 ----
#             nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),    # 256*256, 272*224, 512*320
#             nn.LeakyReLU(0.2, inplace=True),
#             # ---- Conv Layer 2 ----
#             nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1, bias=False),    # 128*128, 136*112, 256*160
#             nn.BatchNorm2d(ndf*2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # ---- Conv Layer 3 ----
#             nn.Conv2d(ndf*2, ndf*2, 4, stride=2, padding=1, bias=False),    # 64*64, 68*56, 128*80
#             nn.BatchNorm2d(ndf*2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # ---- Conv Layer 4 ----
#             nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1, bias=False),    # 32*32, 34*28, 64*40
#             nn.BatchNorm2d(ndf*4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # ---- Conv Layer 5 ----
#             nn.Conv2d(ndf*4, ndf*4, 4, stride=2, padding=1, bias=False),    # 16*16, 17*14, 32*20
#             nn.BatchNorm2d(ndf*4),
#             nn.LeakyReLU(0.2, inplace=True), #
#             # # ---- Conv Layer 6 ----
#             # nn.Conv2d(ndf*4, ndf*4, 4, stride=2, padding=1, bias=False),    # 8*8, 8*7, 16*10
#             # nn.BatchNorm2d(ndf*4),
#             # nn.LeakyReLU(0.2, inplace=True),
#             # # ---- Conv Layer 7 ----
#             # nn.Conv2d(ndf*4, ndf*8, 4, stride=2, padding=1, bias=False),    # 4*4, 4*3, 8*5
#             # nn.BatchNorm2d(ndf*8),    # ndf*8  or  1
#             # nn.LeakyReLU(0.2, inplace=True),
#             # ---- Conv Layer Final ----
#             nn.Conv2d(ndf*4, 1, 3, stride=1, padding=0, bias=False),    # 1*1, kernel =4 for 512*512 input, kernel=3 for 448 input
            
#             # nn.AdaptiveMaxPool2d((1, 1)),    # for not square input
#         )

#     def forward(self, input):
#         return self.main(input)



class Discriminator(nn.Module):

    def __init__(self, nc=4, ndf=16):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input size: 512*512, 544*448, 1024*640
            # ---- Conv Layer 1 ----
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),    # 256*256, 272*224, 512*320
            nn.LeakyReLU(0.2, inplace=True),
            # ---- Conv Layer 2 ----
            nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1, bias=False),    # 128*128, 136*112, 256*160
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # ---- Conv Layer 3 ----
            nn.Conv2d(ndf*2, ndf*2, 4, stride=2, padding=1, bias=False),    # 64*64, 68*56, 128*80
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # ---- Conv Layer 4 ----
            nn.Conv2d(ndf*2, ndf*2, 4, stride=2, padding=1, bias=False),    # 32*32, 34*28, 64*40
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # ---- Conv Layer 5 ----
            nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1, bias=False),    # 16*16, 17*14, 32*20
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # ---- Conv Layer 6 ----
            nn.Conv2d(ndf*4, ndf*4, 4, stride=2, padding=1, bias=False),    # 8*8, 8*7, 16*10
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # ---- Conv Layer 7 ----
            nn.Conv2d(ndf*4, ndf*8, 4, stride=2, padding=1, bias=False),    # 4*4, 4*3, 8*5
            nn.BatchNorm2d(ndf*8),    # ndf*8  or  1
            nn.LeakyReLU(0.2, inplace=True),
            # ---- Conv Layer Final ----
            nn.Conv2d(ndf*8, 1, 4, stride=1, padding=0, bias=False),    # 1*1, kernel =4 for 512*512 input, kernel=3 for 448 input
            
            # nn.AdaptiveMaxPool2d((1, 1)),    # for not square input
        )

    def forward(self, input):
        return self.main(input)


