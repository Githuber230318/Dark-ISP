import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os

from .utils import color_transform
from .noise_predictor import Matrix_Predictor, Color_Level_Process, Matrix_Predictor_v4


class Dark_ISP(nn.Module):
    def __init__(self, matloss_epoch, loss_weight=0.05):
        super().__init__()
        self.linear = Matrix_Predictor()
        self.nonlinear = Color_Level_Process()
        self.matloss_epoch = matloss_epoch
        self.loss_weight = loss_weight
        
    def forward(self, x_i, mat, cur_epoch):
        dmat = self.linear(x_i, mat)
        x_g, mat = color_transform(x_i, mat, dmat=dmat)
        x_g = x_g.clip(0, 1)
        x_l = self.nonlinear(x_g) 

        if self.training:
            if cur_epoch >= self.matloss_epoch:
                # loss_mat = torch.zeros_like(x_l)
                loss_mat = self.mat_loss2(x_i, x_l, mat, cur_epoch)
            else:
                loss_mat = torch.zeros_like(x_l)  # For Inference
                # loss_mat = self.mat_loss(x_i, x_l, mat)
        else:
            loss_mat = torch.zeros_like(x_l)  # For Inference
        
        return x_l, loss_mat

    
    def mat_loss2(self, x_i, x_l, P, epoch):
        """P: matrix to be optimized, shape (4, 3, 4)"""
        b, c, h, w = x_i.shape
        x_i = x_i.view(b, c, -1) # Shape:[4 4 hw]
        # 重塑张量
        x_l = x_l.view(b, 3, -1)  # Shape: [4, 3, hw]
        # 进行批量矩阵乘法
        result1 = x_l.unsqueeze(2) * x_i.unsqueeze(1)  # Shape: [4, 3, 4, hw]
        result2 = x_i.unsqueeze(2) * x_i.unsqueeze(1)  # Shape: [4, 4, 4, hw]
        result2 = result2.permute(0, 3, 1, 2)
        eps = 1e-6
        I = torch.eye(4, device=result2.device).unsqueeze(0).unsqueeze(0)
        # [1,1,4,4]
        out_reg = result2 + eps * I
        result2 = torch.linalg.inv(out_reg) # Shape: [4, hw, 4, 4]
        result1 = result1.permute(0, 3, 1, 2) # Shape: [4, hw, 3, 4]
        A = torch.matmul(result1, result2) # A shape: [4, hw, 3, 4]
        A = A.view(b, h, w, 3, 4).permute(0, 3, 4, 1, 2)
        P = P.permute(0, 3, 4, 1, 2)  # P shape: [4, 3, 4, h, w]


        # Calculate dot product
        dot_product = (A * P).sum(dim=2)  # Shape: [4, 3, h, w]

        # Calculate Norm
        norm1 = torch.norm(P, dim=2) + 1e-5  # Shape: [4, 3, h, w]
        norm2 = torch.norm(A, dim=2) + 1e-5 # Shape: [4, 3, h, w]

        # Calculate coscine similarity
        similarity = dot_product / (norm1 * norm2)
        loss_mat = (1 - similarity).mean() * min(self.loss_weight, (epoch+1) / 10 * self.loss_weight) 


        return loss_mat 