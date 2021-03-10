import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.BatchNorm2d(out_channels), nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return(x)


class upconv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upconv_block, self).__init__()
        self.upsampleconv = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor = 2),
                                          nn.Conv2d(in_channels, in_channels, 3, 1, 1)
                                          )
        self.conv = conv_block(in_channels + in_channels // 2, out_channels)

    def forward(self, x1, x2):
        x2 = self.upsampleconv(x2)
        y_offsets = x1.size()[2] - x2.size()[2]
        x_offsets = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, [x_offsets // 2, x_offsets - x_offsets // 2,
                        y_offsets // 2, y_offsets - y_offsets // 2])
        x = torch.cat([x1, x2], 1)
        x = self.conv(x)

        return(x)


class unet_pl(torch.nn.Module):
    def __init__(self, n_classes = 2, in_channels = 3):
        super(unet_pl, self).__init__()
        self.in_channels = in_channels
        self.pool = nn.MaxPool2d(2)
        filters = [32, 64, 128, 256, 512]

        self.conv1 = conv_block(self.in_channels, filters[0])
        self.conv2 = conv_block(filters[0], filters[1])
        self.conv3 = conv_block(filters[1], filters[2])
        self.conv4 = conv_block(filters[2], filters[3])
        self.center = conv_block(filters[3], filters[4])

        self.upconv5 = upconv_block(filters[4], filters[3])
        self.upconv4 = upconv_block(filters[3], filters[2])
        self.upconv3 = upconv_block(filters[2], filters[1])
        self.upconv2 = upconv_block(filters[1], filters[0])

        self.final = nn.Conv2d(filters[0], n_classes, 3, 1, 1)


    def forward(self, x):

        #encoder
        conv1 = self.conv1(x)
        pool1 = self.pool(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool(conv4)

        center = self.center(pool4)

        #decoder

        up5 = self.upconv5(conv4, center)
        up4 = self.upconv4(conv3, up5)
        up3 = self.upconv3(conv2, up4)
        up2 = self.upconv2(conv1, up3)

        output = self.final(up2)

        return(output)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet_pl()
    model.to(device)
    summary(model, (3, 360, 640))




