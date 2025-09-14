import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os

from .utils import gaussian_kernel, Tayler_res

class Matrix_Predictor(nn.Module):
    def __init__(self, dim=32, num_heads=1, qkv_bias=True, qk_scale=None, down_drop=0.2, attn_drop=0., proj_drop=0., head_drop=0., data='NOD'):
        super().__init__()

        self.lc_block1 = nn.Sequential(
            # nn.Conv2d(4, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(4, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Dropout(down_drop))
        self.lc_block2 = nn.Sequential(
            # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Dropout(down_drop))
        self.lc_block3 = nn.Sequential(
            # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Dropout(down_drop))
        self.lc_block4 = nn.Sequential(
            # nn.Conv2d(4, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Dropout(down_drop))
        self.lc_block5 = nn.Sequential(
            # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Dropout(down_drop))

        self.gl_block1 = nn.Sequential(
            # nn.Conv2d(4, dim, kernel_size=3, stride=1, padding=1, bias=False),
            GaussianConv2d(4, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Dropout(down_drop))
        self.gl_block2 = nn.Sequential(
            # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            GaussianConv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Dropout(down_drop))
        self.gl_block3 = nn.Sequential(
            # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            GaussianConv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Dropout(down_drop))
        self.gl_block4 = nn.Sequential(
            # nn.Conv2d(4, dim, kernel_size=3, stride=1, padding=1, bias=False),
            GaussianConv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Dropout(down_drop))
        self.gl_block5 = nn.Sequential(
            # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            GaussianConv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Dropout(down_drop))
       
        self.feature_atten1 = FA(channels=dim)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        # Query Adaptive Learning (QAL)
        # self.q = nn.Parameter(torch.rand((1, 9 + 1, dim)), requires_grad=True)
        self.q = nn.Linear(1, dim)
     
        self.pooling = nn.AvgPool2d(kernel_size=16, stride=16)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.data = data
        self.down = nn.Linear(dim, 1)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mat):

        l_x1 = self.lc_block1(x)
        l_x2 = self.lc_block2(l_x1)
        l_x3 = self.lc_block3(l_x2)
        l_x4 = self.lc_block4(l_x3)
        d_x1 = self.lc_block5(l_x4+l_x3+l_x2+l_x1)

        g_x1 = self.gl_block1(x)
        g_x2 = self.gl_block2(g_x1)
        g_x3 = self.gl_block2(g_x2)
        g_x4 = self.gl_block2(g_x3)
        d_x2 = self.gl_block5(g_x4+g_x3+g_x2+g_x1)
        
        local = self.feature_atten1(d_x2, mat)

        d_x = self.pooling(d_x1).flatten(2).transpose(1, 2) # (4 988 64)

        B, N, C = d_x.shape
        k = self.k(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (4 1 988 64)
        v = self.v(d_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (4 1 988 64)

        q_input = mat.flatten(1).unsqueeze(2) # (4 12 1)
        q = self.sigmoid(self.q(q_input.unsqueeze(2))) # (4 12 1 64)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (4 1 12 64)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale # (4 1 12 988)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        glo = (attn @ v).transpose(1, 2).reshape(B, 12, C) # (4 12 64)
        glo = self.proj(glo)
        glo = self.proj_drop(glo) # 4 12 64

        glo = self.down(glo).squeeze(2)
       
        glo = glo.unsqueeze(2).unsqueeze(2)
        out = glo + local
        # out = local
        # B, C, H, W = x.shape
        # out = glo.repeat(1, 1, H, W)

        return out, glo, local


class GaussianConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, sigma: float = 1, stride: int = 1, padding: int = 0, bias=False):
        super(GaussianConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.stride = stride
        self.padding = padding

        kernel = gaussian_kernel(kernel_size, sigma)

        self.weight = nn.Parameter(data=kernel.unsqueeze(0).unsqueeze(0).repeat(out_channels, in_channels, 1, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=bias)  # 不训练偏置

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class FA(nn.Module):

    """
      FA happens in five steps

      2 : Converts the feature maps into query and key representaions.
      3 : Concats the downgraded features and divide it into blocks.
      4 : Performs Multi-Head Attention.
      5 : Aggregates and outputs the enhanced representation

    """
    def __init__(self, channels, elements=12):
        super().__init__()
        self.mult_scale_heads = 8
        self.elements = elements
        self.q = nn.Conv2d(in_channels=channels, out_channels=elements , kernel_size=1, stride=1)
        self.proj = nn.Sequential(nn.Linear(1, 32),
                                  nn.Linear(32, 8))
        self.sigmoid = nn.Sigmoid()
        self.k = nn.Linear(1, channels)
        self.v = nn.Linear(1, channels)
        self.ada_drop = nn.Dropout(0.1)
        self.lc_drop = nn.Dropout(0.1)
        # self.qg = nn.Linear(1, channels)
        # self.proj_g = nn.Linear(channels, 1)
        # self.channel = nn.Linear(channels, channels)

    def forward(self, x, q_input):
        """ 
        Args : 
            x:  a list of, feature map in (B, C, H, W) format.
            q_input: in (B, 12, 1) format.
        Returns :
            q is x
            k, v is q_input
        """
        B, C, H, W = x.size()
        q_input = q_input.flatten(1).unsqueeze(2) # (4 12 1)
        # Key Query Generation
        q = self.q(x)  # (B, 12, H, W)
        k = self.k(q_input) # (batch_size, 12, C)
        v = self.v(q_input)

        # (batch_size, 12, H, W, C)
        q_ = q[:, :, :, :, None]
        k_ = k[:, :, None, None, :]
        v_ = v[:, :, None, None, :]

        # (batch_size, 1, H, W, C)
        ada_weights = torch.sum(
            q_ * k_, dim=1, keepdim=True) / (float(self.elements)**0.5)
        ada_weights = ada_weights.softmax(dim=4)  
        ada_weights = self.ada_drop(ada_weights)
        # (batch_size, 12, H, W)
        local =  (v_ * ada_weights).sum(dim=4)
        local = local.unsqueeze(-1)
        local = self.proj(local)
        local = self.lc_drop(local)
        local = local.sum(dim=-1)

        return local



class LUT3D(nn.Module):
    r"""The 3DLUT generator module.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks (or the number of basis LUTs).
    """

    def __init__(self, n_colors, n_vertices, n_ranks, n_feats=108) -> None:
        super().__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.layer = nn.Linear(n_feats, n_feats//2)
        # h0
        self.weights_generator = nn.Linear(n_feats//2, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_colors), bias=False)
        
        self.relu = torch.nn.ReLU()
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks

        dim = n_vertices
        self.weight_r = torch.ones(3,dim,dim,dim-1, dtype=torch.float).cuda()
        self.weight_r[:,:,:,(0,dim-2)] *= 2.0
        self.weight_g = torch.ones(3,dim,dim-1,dim, dtype=torch.float).cuda()
        self.weight_g[:,:,(0,dim-2),:] *= 2.0
        self.weight_b = torch.ones(3,dim-1,dim,dim, dtype=torch.float).cuda()
        self.weight_b[:,(0,dim-2),:,:] *= 2.0

        self.init_weights()

    def init_weights(self):
        r"""Init weights for models.

        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).

        """
        nn.init.ones_(self.weights_generator.bias)
        identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
                dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        # identity_lut = torch.stack([
        #     *[torch.zeros(
        #         self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks)]
        #     ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, dx):
        # dx = self.adaptive_avg_pool(dx).flatten(2)
        # x = self.layer(dx).squeeze(2)
        x = self.adaptive_avg_pool(dx).flatten(1)
        x = self.layer(x)
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
    
        return luts


class Color_Level_Process(nn.Module):
    def __init__(self, number_f=32):
        super().__init__()
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)  
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f,24,3,1,1,bias=True) 
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.drop = nn.Dropout(0.1)
        

    def forward(self, imgs):
        x1 = self.relu(self.e_conv1(imgs))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x_r = self.e_conv5(x1+x2+x3+x4)
        # x_r =  self.drop(x_r)
        # x_r = self.tanh(x_r)
        r1, r2, r3 = torch.split(x_r, 8, dim=1)
        r1 = r1.softmax(dim=1)
        r2 = r2.softmax(dim=1)
        r3 = r3.softmax(dim=1)

        # bais = self.bais(x_r)
        # outs = embed_nonlinear_channel(imgs, [r1, r2, r3])
        # imgs = (imgs + outs).clip(0, 1)

        # outs = embed_nonlinear_channel_nores(imgs, [r1, r2, r3])
        # imgs = outs.clip(0, 1)
        # imgs = (imgs + outs).clip(0, 1)
        

        outs = Tayler_res(imgs, [r1, r2, r3])
        imgs = (imgs + outs).clip(0, 1)
        # imgs = (imgs + outs + bais).clip(0, 1)

        return imgs

    

class Zero_DCE(nn.Module):
    def __init__(self, number_f=32):
        super().__init__()
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 
        self.relu = nn.ReLU(inplace=True)
        self.bais = nn.Conv2d(24,3,3,1,1,bias=True,groups=3)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
        x = x + r6*(torch.pow(x,2)-x)	
        x = x + r7*(torch.pow(x,2)-x)
        enhance_image = x + r8*(torch.pow(x,2)-x)
        return enhance_image

class NILUT(nn.Module):
    """
    Simple residual coordinate-based neural network for fitting 3D LUTs
    Official code: https://github.com/mv-lab/nilut
    """
    def __init__(self, in_features=3, hidden_features=64, hidden_layers=3, out_features=3, res=True):
        super().__init__()
        
        self.res = res
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())
        
        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.Tanh())
        
        self.net.append(nn.Linear(hidden_features, out_features))
        if not self.res:
            self.net.append(torch.nn.Sigmoid())
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, intensity):
        x = intensity.permute(0,2,3,1)
        output = self.net(x)
        if self.res:
            output = output + x
            output = torch.clamp(output, 0.,1.)
        return output.permute(0,3,1,2)