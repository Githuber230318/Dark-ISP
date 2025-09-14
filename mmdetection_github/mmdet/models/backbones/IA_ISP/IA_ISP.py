import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageProcessor(nn.Module):
    def __init__(self, cfg, dim=16, input_dim=2048, hidden_dim=64, num_params=14):
        super().__init__()
        self.cfg = cfg
        self.num_params = num_params

        self.conv1 = Block(3, dim, kernel_size=3, stride=(2, 2))
        self.conv2 = Block(dim, 2*dim, kernel_size=3, stride=(2, 2))
        self.conv3 = Block(2*dim, 2*dim, kernel_size=3, stride=(2, 2))
        self.conv4 = Block(2*dim, 2*dim, kernel_size=3, stride=(2, 2))
        self.conv5 = Block(2*dim, 2*dim, kernel_size=3, stride=(2, 2))

        self.fc_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_params)
        self._init_weights()
        self.relu = nn.LeakyReLU(negative_slope=0.1)

        # 获取过滤器配置
        filters = self.cfg.filters
        self.filters = [filter(self.cfg) for filter in filters]

    def _init_weights(self):
        # 对fc1和fc2应用Xavier初始化
        nn.init.xavier_normal_(self.fc1.weight)
        if self.fc2 is not None:
            nn.init.xavier_normal_(self.fc2.weight)

    def extract_parameters_2(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.reshape(-1, self.fc_dim)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, input_data):
        filter_imgs_series = []
        
        # 调整输入图像大小
        x = F.interpolate(input_data, size=(256, 256), mode='bilinear', align_corners=False)
        filter_features = self.extract_parameters_2(x)  # 假设有一个相应的函数

        filter_parameters = []

        for j, filter in enumerate(self.filters): 
            # 应用过滤器
            x, filter_parameter = filter(input_data, filter_features)
            filter_parameters.append(filter_parameter)
            filter_imgs_series.append(x)

        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=(2,2)):
        super(Block, self).__init__()
        pad_h, pad_w = (kernel_size - 2) // 2 + 1, (kernel_size - 2) // 2 + 1
        self.conv = nn.Conv2d(in_channels=in_channels,  # 输入通道数
                            out_channels=out_channels,  # 输出通道数
                            kernel_size=kernel_size,  # 卷积核大小
                            stride=stride,  # 步幅
                            padding=(pad_h, pad_w),  # 'VALID' 对应于无填充
                            bias=False)
        nn.init.normal_(self.conv.weight, std=0.01)  # 正态分布初始化
        self.bias = nn.Parameter(torch.zeros(out_channels)) 
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.relu(x)
        return x