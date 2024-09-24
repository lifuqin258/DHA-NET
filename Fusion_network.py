import torchvision.models as models
import torch
import torch.nn as nn

class y_glcm(nn.Module):
    def __init__(self):
        super(y_glcm, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x - x.mean(dim=(2, 3), keepdim=True)
        cov = torch.einsum('bikj,bilk->bijl', x, x)
        cov = cov / (x.size(2) * x.size(3) - 1)
        x = self.conv(cov)
        return x

class DHA_NET(nn.Module):
    def __init__(self, channels, c2=None, factor=32, num_classes=117):
        super(DHA_NET, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        alexnet.eval()
        self.feature_extractor = alexnet.features
        self.y_glcm = y_glcm()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(p=0.6)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_0 = nn.Sequential(
            nn.Linear(channels // self.groups, channels // self.groups, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // self.groups, channels // self.groups, bias=False),
            nn.Sigmoid()
        )
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)

        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.channel_compress = nn.Conv2d(512 // self.groups, 256 // self.groups, kernel_size=1)
        self.fc = nn.Sequential(nn.Conv2d(channels // self.groups, channels // self.groups, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(channels // self.groups, channels // self.groups, 1, bias=False))
        self.fc_1 = nn.Linear(channels * 6 * 6, 117)

    def forward(self, x, y):
        y_glcm = self.y_glcm(y)
        x = self.feature_extractor(x)
        y = self.feature_extractor(y)
        y_glcm = self.feature_extractor(y_glcm)
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        b_g, c_g, _, _ = group_x.size()
        x = self.avg_pool(group_x).view(b_g, c_g)
        x = self.fc_0(x).view(b_g, c_g, 1, 1)


        group_y = y.reshape(b * self.groups, -1, h, w)
        group_y_glcm = y_glcm.reshape(b * self.groups, -1, h, w)
        group_y_glcm = torch.cat([self.conv3x3(group_y), group_y_glcm], dim=1)
        group_y_glcm = self.fc(self.channel_compress(group_y_glcm))
        avg_out = self.fc(self.avg_pool(group_y))
        max_out = self.fc(self.max_pool(group_y))
        y1 = self.gn(group_y * (avg_out + max_out + group_y_glcm).sigmoid())

        x11 = self.softmax(self.agp(x).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = y1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(y1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        x = (group_x * weights.sigmoid() + group_y * weights.sigmoid()).reshape(b, c, h, w)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc_1(x)
        return x

