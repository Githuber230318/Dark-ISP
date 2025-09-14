from torch.nn.parallel import DataParallel, DistributedDataParallel
from functools import partial
import torch
import math
from copy import deepcopy
from torch import nn
from torch.nn import functional as F


class UNetArch(nn.Module):
    def __init__(self, inchannels=3, outchannels=3, channels=32, load_path=None, fozen=True, param_key='params_deploy') -> None:
        super().__init__()

        self.scale_factor = 0.1
        self.load_path = load_path
        self.encoder = Encoder(inchannels, outchannels, channels)
        self.decoder = Decoder(inchannels, outchannels, channels)
        # self.uncertainty = Discriminator(input_dim=4, num_filters=32)
        self.conv_s = nn.Conv2d(channels, outchannels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        if load_path is not None:
            self.loading_repnr_checkpoint(param_key)
        if fozen:
            self.fozen_parameters()
        self.conv_s.weight.data.copy_(self.decoder.conv10_1.weight.data)
        self.conv_s.bias.data.copy_(self.decoder.conv10_1.bias.data)

    def fozen_parameters(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def loading_repnr_checkpoint(self, param_key='params_deploy'):
        if self.load_path is not None:
            self.load_network(self.encoder, self.load_path, param_key=param_key)
        if self.load_path is not None:
            self.load_network(self.decoder, self.load_path, param_key=param_key)

    def load_network(self, net, load_path, strict=True, param_key='params_deploy'):
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
        load_net = load_net[param_key]
       
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net_state_dict = net.state_dict()
        filtered_load_net = {k: v for k, v in load_net.items() if k in net_state_dict and net_state_dict[k].shape == v.shape}
        net.load_state_dict(filtered_load_net, strict=strict)

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def _check_and_padding(self, x):
        ### This function is totally writen by ChatGPT
        # Calculate the required size based on the input size and required factor
        _, _, h, w = x.size()
        stride = (2 ** (5 - 1))

        # Calculate the number of pixels needed to reach the required size
        dh = -h % stride
        dw = -w % stride

        # Calculate the amount of padding needed for each side
        top_pad = dh // 2
        bottom_pad = dh - top_pad
        left_pad = dw // 2
        right_pad = dw - left_pad
        self.crop_indices = (left_pad, w+left_pad, top_pad, h+top_pad)

        # Pad the tensor with reflect mode
        padded_tensor = F.pad(
            x, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect"
        )

        return padded_tensor

    def _check_and_crop(self, x):
        left, right, top, bottom = self.crop_indices
        x = x[:, :, top:bottom, left:right]
        return x

    def forward(self, x):
        x = self._check_and_padding(x)
        conv5, conv4, conv3, conv2, conv1 = self.encoder(x)
        conv10 = self.decoder(conv5, conv4, conv3, conv2, conv1)
        out = self._check_and_crop(conv10).clip(0, 1)
        return out
    
    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt
    

class DK(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, c, k):
        super(DK, self).__init__()

        self.conv1 = nn.Conv2d(c, k, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(k, c, kernel_size=3, stride=1, padding=1)  
        self.mid = nn.Conv2d(k, k, kernel_size=3, stride=1, padding=1)
        
        self.weights = nn.Sequential(nn.Linear(k,k), nn.Sigmoid())
        self.fFP = nn.Conv2d(k, k, kernel_size=3, stride=1, padding=1) 
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.tanh = nn.Tanh()

        self.kmlp = nn.Sequential(
            nn.Linear(1,k),
            nn.ReLU(),)         
      
        self.BLE = nn.Sequential(
            nn.Linear(1,k),
            nn.ReLU(),) 

    def forward(self, x, iso):
        # The first 1x1 conv
        x = self.conv1(x)  
        b, c, h, w = x.shape

        # Fixed Patern Noise
        k = self.kmlp(iso)             # b * k 
        k = k.view(b, c, 1, 1)         # b * k * 1 * 1

        NFP = self.fFP(x) * k         # b * k * h * w
        # Black Level Error
        gol = self.gap(x)            # global component
                                     # b * k * 1 * 1
        BLE = self.BLE(iso)          # b * k   
        BLE = BLE.view(b, c, 1, 1)
        weight = self.weights((gol + BLE).squeeze()).view(b, c, 1, 1)
        mid = self.mid(x)            # b * k * h * w
        NBLE = weight * mid 
        
        noise = NBLE + NFP
        return noise
    
    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt

    
class Encoder(nn.Module):
    def __init__(self, inchannels=3, outchannels=3, channels=32) -> None:
        super(Encoder, self).__init__()
        self.conv1_1 = nn.Conv2d(inchannels, channels, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(channels * 4, channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(channels * 8, channels * 8, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(channels * 8, channels * 16, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.lrelu(self.conv5_1(pool4))
        return conv5, conv4, conv3, conv2, conv1

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt
    

class Decoder(nn.Module):
    def __init__(self, inchannels=3, outchannels=3, channels=32) -> None:
        super(Decoder, self).__init__()
        self.conv5_2 = nn.Conv2d(channels * 16, channels * 16, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(channels * 16, channels * 8, 2, stride=2)
        self.conv6_1 = nn.Conv2d(channels * 16, channels * 8, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(channels * 8, channels * 8, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(channels * 8, channels * 4, 2, stride=2)
        self.conv7_1 = nn.Conv2d(channels * 8, channels * 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(channels * 4, channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.conv9_1 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(channels, outchannels, kernel_size=1, stride=1)

    def forward(self, conv5, conv4, conv3, conv2, conv1):
        conv5 = self.lrelu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.lrelu(self.conv8_1(up8))
        conv8 = self.lrelu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.lrelu(self.conv9_1(up9))
        conv9 = self.lrelu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)

        return conv10

    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt