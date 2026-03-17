import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

import numpy as np
# @ KANBlock
class KANLayer(nn.Module):
    def __init__(
            self,in_dim, out_dim, grid_size=5,spline_order=3, grid_range=[-1, 1], 
            base_activation=torch.nn.SiLU, scale_noise=0.1,scale_base=1.0,
            scale_spline=1.0,enable_standalone_scale_spline=True,grid_eps=0.02
            ):
        super(KANLayer,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        h = (grid_range[1] - grid_range[0]) / grid_size   # 计算网格步长
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0] 
            )
            .expand(in_dim, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_dim, in_dim, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.scale_noise = scale_noise 
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)# 使用 Kaiming 均匀初始化基础权重
        with torch.no_grad():
            noise = (# 生成缩放噪声
                (
                    torch.rand(self.grid_size + 1, self.in_dim, self.out_dim)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_( # 计算分段多项式权重
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则使用 Kaiming 均匀初始化分段多项式缩放参数
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    
    def b_splines(self, x):
        #定义 (in_features, grid_size + 2 * spline_order + 1)
        grid: torch.Tensor = (self.grid) 
        x = x.unsqueeze(-1)
        # 计算 0 阶 B-样条基函数值
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # 递归计算 k-1 阶B-样条基函数值
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()
    def curve2coeff(self, x, y):
        # 计算 B-样条基函数
        # (in_features, batch_size, grid_size + spline_order)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        # 使用最小二乘法求解线性方程组
        # (in_features, grid_size + spline_order, out_features)
        solution = torch.linalg.lstsq(A, B).solution
        # 调整结果的维度顺序
        # (out_features, in_features, grid_size + spline_order)
        result = solution.permute(2,0,1)
        return result.contiguous()  
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline else 1.0)
    def forward(self, x):
        # 计算基础线性层的输出
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # 计算分段多项式线性层的输出
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_dim, -1),)
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        batch = x.size(0)
        splines = self.b_splines(x)  # (batch, in, coeff)  # 计算 B-样条基函数
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)  # 调整维度顺序为 (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)  # 调整维度顺序为 (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # (batch, in, out)
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0] # 对每个通道单独排序以收集数据分布
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)   # 更新网格和分段多项式权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

class KANBlock(nn.Module):
    def __init__(
        self, layers_hidden,grid_size=5,spline_order=3,grid_range=[-1, 1],
        scale_noise=0.1,scale_base=1.0,scale_spline=1.0,grid_eps=0.02,
        base_activation=torch.nn.SiLU):
        super(KANBlock,self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLayer(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
    def forward(self, x: torch.Tensor, update_grid=False):    
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
    
# @ ViT4Block
class PatchEmbed(nn.Module):
    def __init__(self, block_size, in_c, embed_dim, norm_layer=None):
        super(PatchEmbed,self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=block_size,stride=block_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self,x):
        # [num_node, in_dim]  -> [1, 1, num_node, in_dim]
        x = x[None, None, :, :]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
# Win-attention    
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim*3, bias=False)
    
    def forward(self,x):
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # [batch_size, num_heads, num_patches, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches]
        # @: multiply -> [batch_size, num_heads, num_patches, num_patches]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # @: multiply -> [batch_size, num_heads, num_patches, embed_dim_per_head]
        # transpose: -> [batch_size, num_patch, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches, total_embed_dim]
        x = (attn @ v).transpose(1,2).reshape(B, N, C)

        return x
    
class ViT4Block(nn.Module):
    def __init__(self, black_size,num_patches=8, embed_dim=776):
        super(ViT4Block,self).__init__()
        self.patch_embed = PatchEmbed(black_size, 1, embed_dim)
        self.norm_layer=nn.LayerNorm(embed_dim)
        self.attn_layer = Attention(embed_dim,num_patches)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
    def forward(self, x):

        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x) # [B, 8, 512]
        x = self.norm(x)
        # Win-attention
        x = self.attn_layer(x)
        return x
    

if __name__ == '__main__':
    batch_size = 32
    x = torch.rand(1546,256)
    # net = KANBlock([32,512,256])
    # y = net(x)  # (32,256)
    # print(y.shape)

    net = PatchEmbed((16,32), 1, 512)
    y = net(x)
    print(y.shape)

    # Net1 = ViT4Block()

    # print(Net1(x).shape)