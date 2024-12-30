import torchvision.models as models
import torch
import torch.nn as nn

class x_glcm(nn.Module):
    def __init__(self):
        super(y_glcm, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x - x.mean(dim=(2, 3), keepdim=True)
        cov = torch.einsum('bikj,bilk->bijl', x, x)
        cov = cov / (x.size(2) * x.size(3) - 1)
        x = self.conv(cov)
        return x

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32, num_classes=63):
        super(EMA, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        alexnet.eval()
        self.feature_extractor = alexnet.features
        self.x_glcm = x_glcm()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.channel_compress = nn.Conv2d(512 // self.groups, 256 // self.groups, kernel_size=1)
        self.fc = nn.Sequential(nn.Conv2d(channels // self.groups, channels // self.groups, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(channels // self.groups, channels // self.groups, 1, bias=False))

    def forward(self, x, y):
        x_glcm = self.x_glcm(x)
        x = self.feature_extractor(x)
        y = self.feature_extractor(y)
        x_glcm = self.feature_extractor(x_glcm)

        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        group_x_glcm = x_glcm.reshape(b * self.groups, -1, h, w)
        group_x_glcm = torch.cat([self.conv3x3(group_x), group_x_glcm], dim=1)
        group_x_glcm = self.fc(self.channel_compress(group_x_glcm))
        x = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid() * group_x_glcm.sigmoid())

        group_y = y.reshape(b * self.groups, -1, h, w)
        avg_out = self.fc(self.avg_pool(group_y))
        max_out = self.fc(self.max_pool(group_y))
        y1 = self.gn(group_y * (avg_out + max_out).sigmoid())

        x11 = self.softmax(self.agp(x).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = y1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(y1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        x = (group_x * weights.sigmoid() + group_y * weights.sigmoid()).reshape(b, c, h, w)
        return x
