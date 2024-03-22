'''
@Article{DistgLF,
    author    = {Wang, Yingqian and Wang, Longguang and Wu, Gaochang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
    title     = {Disentangling Light Fields for Super-Resolution and Disparity Estimation},
    journal   = {IEEE TPAMI},
    year      = {2022},
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim 

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = 64
        n_group = 4
        n_block = 4
        self.angRes = args.angRes_in
        self.factor = args.scale_factor
        # self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=self.angRes, padding=self.angRes, bias=False)
        self.sr_transformer=SRTransformer(1,channels,embed_size=64,heads=n_group*1,num_blocks=n_block)
        self.disentg = CascadeDisentgGroup(n_group, n_block, self.angRes, channels)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * self.factor ** 2, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(self.factor),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x, info=None):
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        x = SAI2MacPI(x, self.angRes)
        # buffer = self.init_conv(x)
        buffer=self.sr_transformer(x)
        buffer = self.disentg(buffer)
        buffer_SAI = MacPI2SAI(buffer, self.angRes)
        out = self.upsample(buffer_SAI) + x_upscale
        return out


class CascadeDisentgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeDisentgGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(DisentgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer)
        return self.conv(buffer) + x


class DisentgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(DisentgGroup, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DisentgBlock(angRes, channels))
        self.Block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_block):
            buffer = self.Block[i](buffer)
        return self.conv(buffer) + x


class DisentgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DisentgBlock, self).__init__()
        SpaChannel, AngChannel, EpiChannel = channels, channels//4, channels//2

        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(SpaChannel, SpaChannel, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.AngConv = nn.Sequential(
            nn.Conv2d(channels, AngChannel, kernel_size=angRes, stride=angRes, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(AngChannel, angRes * angRes * AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(angRes),
        )
        self.EPIConv = nn.Sequential(
            nn.Conv2d(channels, EpiChannel, kernel_size=[1, angRes * angRes], stride=[1, angRes], padding=[0, angRes * (angRes - 1)//2], bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(EpiChannel, angRes * EpiChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            PixelShuffle1D(angRes),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(SpaChannel + AngChannel + 2 * EpiChannel, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False),
        )

    def forward(self, x):
        feaSpa = self.SpaConv(x)
        feaAng = self.AngConv(x)
        feaEpiH = self.EPIConv(x)
        feaEpiV = self.EPIConv(x.permute(0, 1, 3, 2).contiguous()).permute(0, 1, 3, 2)
        buffer = torch.cat((feaSpa, feaAng, feaEpiH, feaEpiV), dim=1)
        buffer = self.fuse(buffer)
        return buffer + x


class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tensor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()           # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        # self.criterion_Loss = torch.nn.L1Loss()
        # self.criterion_Loss = torch.nn.SmoothL1Loss()
        self.criterion_L1Loss = nn.L1Loss()  
        # self.criterion_SSIMLoss=SSIMLoss()

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_L1Loss(SR, HR)
        return loss
    


def weights_init(m):
    pass


##### oybk add #######
class MultiHeadAttention(nn.Module):  
    def __init__(self, embed_size, num_heads):  
        super(MultiHeadAttention, self).__init__()  
        self.embed_size = embed_size  
        self.num_heads = num_heads  
        self.head_dim = embed_size // num_heads  
          
        assert (self.head_dim * num_heads == embed_size), "Embedding size needs to be divisible by num_heads"  
          
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)  
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)  
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)  
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)  
          
    def forward(self, values, keys, query, mask=None):  
        N = query.shape[0]  
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  
          
        # Split the embedding into self.num_heads different pieces  
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)  
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)  
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)  
          
        values = self.values(values)  # (N, value_len, num_heads, head_dim)  
        keys = self.keys(keys)  # (N, key_len, num_heads, head_dim)  
        queries = self.queries(queries)  # (N, query_len, num_heads, head_dim)  
          
        # Compute the attention scores  
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / (self.embed_size ** (1 / 2))  
          
        if mask is not None:  
            energy = energy.masked_fill(mask == 0, float('-1e20'))  
          
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)  
          
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.num_heads * self.head_dim)  
        out = self.fc_out(out)  
          
        return out  		
  
class TransformerBlock(nn.Module):  
    def __init__(self, embed_size, heads):  
        super(TransformerBlock, self).__init__()  
        self.attn = MultiHeadAttention(embed_size, heads)  
        # self.attn = nn.MultiheadAttention(embed_size, heads) 
        self.norm1 = nn.LayerNorm(embed_size)  
        self.norm2 = nn.LayerNorm(embed_size)  
        self.fc1 = nn.Linear(embed_size, embed_size)  
        self.fc2 = nn.Linear(embed_size, embed_size)  
        self.dropout = nn.Dropout(0.1)  
          
    def forward(self, src, src_mask=None, src_key_padding_mask=None):  
        out = self.attn(src, src, src)  
        # out, attn_output_weights = self.attn(src, src, src, attn_mask=src_mask,  
        #                                        key_padding_mask=src_key_padding_mask)  
        src = src + self.dropout(out)  
        src = self.norm1(src)  
          
        src = self.fc2(F.relu(self.fc1(src)))  
        src = src + self.dropout(src)  
        src = self.norm2(src)  
          
        return src  
  
class SRTransformer(nn.Module):  
    def __init__(self, in_channels, out_channels, embed_size, heads, num_blocks):  
        super(SRTransformer, self).__init__()  
        self.dilation=5
        self.padding=5
        # self.bias=False
        self.conv_in = nn.Conv2d(in_channels, embed_size, kernel_size=3, dilation=self.dilation,padding=self.padding)  
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_size, heads) for _ in range(num_blocks)])  
        self.conv_out = nn.Conv2d(embed_size, out_channels, kernel_size=3,dilation=self.dilation,padding=self.padding)  
        # self.pixel_shuffle = nn.PixelShuffle(2)  # 假设上采样2倍  
          
    def forward(self, x):  
        # 将输入图像转换为嵌入向量  
        x = self.conv_in(x)  
          
        # 将嵌入向量展平以适应Transformer的输入  
        b, c, h, w = x.shape  
        x = x.view(b, c, -1).permute(2, 0, 1)  # [hw, b, c]  
          
        # 应用Transformer块  
        for block in self.transformer_blocks:  
            x = block(x)  
          
        # 将输出重新整形回图像形状  
        x = x.permute(1, 2, 0).view(b, c, h, w)  
          
        # 上采样并重建图像  
        # x = self.pixel_shuffle(x)  
        x = self.conv_out(x)  
          
        return x  
  
# 定义SSIM损失函数  
class SSIMLoss(nn.Module):  
    def __init__(self, window_size=5, size_average=True, channel=1):  
        super(SSIMLoss, self).__init__()  
  
        self.C1 = 0.01 ** 2  
        self.C2 = 0.03 ** 2  
  
    def forward(self, x, y):  
        mu1 = x.mean([1, 2, 3], keepdim=True)  
        mu2 = y.mean([1, 2, 3], keepdim=True)  
        sigma1_sq = (x - mu1).pow(2).mean([1, 2, 3], keepdim=True)  
        sigma2_sq = (y - mu2).pow(2).mean([1, 2, 3], keepdim=True)  
        sigma12 = (x - mu1) * (y - mu2).mean([1, 2, 3], keepdim=True)  
      
        ssim_map = ((2 * mu1 * mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1.pow(2) + mu2.pow(2) + self.C1) * (sigma1_sq + sigma2_sq + self.C2))  
        return 1-ssim_map.mean()  
    