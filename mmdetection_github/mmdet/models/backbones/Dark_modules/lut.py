
import torch
import torch.nn as nn
import torch.nn.functional as F


def lut_transform(imgs, luts):
    # img (b, 3, h, w), lut (b, c, m, m, m)     
    # normalize pixel values
    imgs = (imgs - .5) * 2. # 由 [0, 1] 归一化成 [-1, 1]
    # reshape img to grid of shape (b, 1, h, w, 3)
    grids = imgs.permute(0, 2, 3, 1).unsqueeze(1)

    # after gridsampling, output is of shape (b, c, 1, h, w)
    outs = F.grid_sample(luts, grids,
        mode='bilinear', padding_mode='border', align_corners=True)
    # remove the extra dimension
    outs = outs.squeeze(2)
    return outs


class LUT1DGenerator(nn.Module):
    r"""The 1DLUT generator module.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points.
        n_feats (int): Dimension of the input image representation vector.
        color_share (bool, optional): Whether to share a single 1D LUT across
            three color channels. Default: False.
    """

    def __init__(self, n_colors, n_vertices, n_feats, 
                 hidden_layers=2, color_share=False, 
                 hidden_drop=0.1) -> None:
        super().__init__()
        
        self.lut1d_generator = []
        self.lut1d_generator.append(nn.Linear(n_feats, n_feats))
        self.lut1d_generator.append(nn.ReLU())
        repeat_factor = n_colors if not color_share else 1

        for _ in range(hidden_layers):
            self.lut1d_generator.append(nn.Linear(n_feats, n_feats))
            self.lut1d_generator.append(nn.Tanh())
            self.lut1d_generator.append(nn.Dropout(hidden_drop))
        self.lut1d_generator.append(nn.Linear(
            n_feats, n_vertices * repeat_factor))
        self.lut1d_generator = nn.Sequential(*self.lut1d_generator)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.color_share = color_share

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        lut1d = self.lut1d_generator(x).view(
            x.shape[0], -1, self.n_vertices)
        if self.color_share:
            lut1d = lut1d.repeat_interleave(self.n_colors, dim=1)
        lut1d = lut1d.sigmoid()
        return lut1d


class LUT3DGenerator(nn.Module):
    r"""The 3DLUT generator module.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks (or the number of basis LUTs).
    """

    def __init__(self, n_colors, n_vertices, n_feats, n_ranks, return_loss=False) -> None:
        super().__init__()
        self.return_loss = return_loss
        # h0
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        # h1
        self.basis_luts_bank = nn.Linear(
            n_ranks, n_colors * (n_vertices ** n_colors), bias=False)
        # self.basis_luts_banks = nn.ModuleList([nn.Linear(n_ranks, n_colors * (n_vertices ** n_colors), bias=False)
        #                           for _ in range(4)])
        
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
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())
        # for i in range(len(self.basis_luts_banks)):
        #     self.basis_luts_banks[i].weight.data.copy_(identity_lut.t())

    def forward(self, x, value):
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        # luts = self.basis_luts_banks[value](weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        if self.return_loss:
            lut = self.basis_luts_bank.weight.data.t()[0]
            # lut = self.basis_luts_banks[0].weight.data.t()[0]
            lut = lut.view(self.n_colors, self.n_vertices, self.n_vertices, self.n_vertices)
            dif_r = lut[:,:,:,:-1] - lut[:,:,:,1:]
            dif_g = lut[:,:,:-1,:] - lut[:,:,1:,:]
            dif_b = lut[:,:-1,:,:] - lut[:,1:,:,:]

            mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))
            return luts, mn
        
        else:
            return luts
