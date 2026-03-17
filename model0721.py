import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=(16,32), in_c=1, embed_dim=512, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x) # [128, 1, 16, 256] -> [128, 512, 1, 8]
        x = x.flatten(2) # [128, 512, 1, 8] -> [128, 512, 8]
        x = x.transpose(1, 2) # [128, 512, 8] -> [128, 8, 512]
        x = self.norm(x) # [128, 8, 512]
        return x
    
class Attention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        
    def forward(self, x):
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
    def __init__(self, num_patches=8, embed_dim=512):
        super(ViT4Block,self).__init__()
        self.patch_embed = PatchEmbed()
        self.norm_layer=nn.LayerNorm(embed_dim)
        self.attn_layer = Attention()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
    def forward(self, x):

        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x) # [B, 8, 512]
        x = self.norm(x)
        x = self.attn_layer(x)
        # [1, 1, 512] -> [B, 1, 512]
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # x = torch.cat((cls_token, x), dim=1)
        # x = self.pos_drop(x + self.pos_embed)
        # x = self.norm(x)
        # return x[:, 0]
        return x
    
if __name__ == '__main__':
    Net1 = ViT4Block()
    x = torch.rand(128,1,16,256)
    print(Net1(x).shape)
    