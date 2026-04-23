import sys

import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from models.clip.clip_base import *
import torchvision


from einops import rearrange, einsum
class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out

class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5,5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7,7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

class deformable_LKA_Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)

        return self._net(f)

class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_Bins=256, score_map_dim=8):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(num_features + score_map_dim, features, kernel_size=1, stride=1, padding=1)

        self.attn1 = deformable_LKA_Attention(features // 1)
        # self.attn2 = deformable_LKA_Attention(features // 2)
        # self.attn3 = deformable_LKA_Attention(features // 4)
        # self.attn4 = deformable_LKA_Attention(features // 8)

        self.casab32 = CASAB(features)  # 2048
        self.casab16 = CASAB(features // 2)  # 1024
        self.casab8 = CASAB(features // 4)  # 512
        self.casab4 = CASAB(features // 8)  # 256

        self.up1 = UpSampleBN(skip_input=features // 1 + features // 2, output_features=features // 2)  # 16
        self.up2 = UpSampleBN(skip_input=features // 2 + features // 4, output_features=features // 4)  # 8
        self.up3 = UpSampleBN(skip_input=features // 4 + features // 8, output_features=features // 8)  # 4

        self.conv3 = nn.Conv2d(features // 8, num_Bins, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        en4, en8, en16, en32, score_map = features[0], features[1], features[2], features[3], features[5]


        # en4 = self.attn4(en4)
        # en8 = self.attn3(en8)
        # en16 = self.attn2(en16)
        en32 = self.attn1(en32)

        de_in = torch.concat([en32, score_map], dim=1)
        de32 = self.conv2(de_in)
        de32 = self.casab32(de32)
        de16 = self.up1(de32, en16)
        de16 = self.casab16(de16)
        de8 = self.up2(de16, en8)
        de8 = self.casab8(de8)
        de4 = self.up3(de8, en4)
        de4 = self.casab4(de4)

        de2 = F.interpolate(de4, size=[de4.size(2) * 2, de4.size(3) * 2], mode='bilinear', align_corners=True)
        out = self.conv3(de2)

        return out, [de2, de4, de8, de16, de32]

class ComputeAdaptiveBins(nn.Module):
    def __init__(self,n_bins=256, min_val=0.001, max_val=10, embedding_dim=1024, dim_out=256):
        super(ComputeAdaptiveBins,self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val

        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 512),
                                       nn.LeakyReLU(),
                                       nn.Linear(512, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

        self.n_group = 8
        self.group_size = self.num_classes // self.n_group



    def forward(self, global_f):
        Bin = self.regressor(global_f)
        Bin = torch.relu(Bin)
        eps=0.1
        Bin = Bin + eps

        bin_widths_normed = Bin / Bin.sum(dim=1, keepdim=True)
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        n, dout = centers.size()
        centers_ = centers.view(n, dout, 1, 1)

        # generate 8 CaBins
        centers_group = centers.reshape(n, self.n_group, self.group_size)

        mean = torch.mean(centers_group, dim=-1, keepdim=True)
        diff = torch.abs(centers_group - mean) + 1e-4
        weighted_diff = torch.softmax((1/diff), dim=-1)
        CaBins = torch.sum(centers_group * weighted_diff, dim=-1) / torch.sum(weighted_diff, dim=-1)

        CaBins = CaBins.reshape(-1, n * self.n_group).squeeze(dim=0)

        CaBins = CaBins.tolist()
        CaBins = [str(round(element,4)) for element in CaBins]

        return bin_edges, centers_, CaBins


class Predict_Depthmap(nn.Module):
    def __init__(self,n_bins=256):
        super(Predict_Depthmap,self).__init__()
        self.n_bins = n_bins
        self.conv_out = nn.Sequential(nn.Conv2d(256, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))

    def forward(self, out, centers_):
        p_map = self.conv_out(out)
        pred = torch.sum(p_map * centers_, dim=1, keepdim=True)

        return pred


class PolaLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(
            in_channels=self.head_dim,
            out_channels=self.head_dim,
            kernel_size=kernel_size,
            groups=self.head_dim,
            padding=kernel_size // 2
        )

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha
        self.scale_param = nn.Parameter(torch.zeros(size=(1, 1, dim)))

        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        self.kernel_function = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape

        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)

        k, v = kv[0], kv[1]
        n = k.shape[1]


        pe_len = self.positional_encoding.shape[1]
        if n != pe_len:

            pe = F.interpolate(self.positional_encoding.permute(0, 2, 1), size=n, mode='linear').permute(0, 2, 1)
            k = k + pe
        else:
            k = k + self.positional_encoding

        scale = nn.Softplus()(self.scale_param)
        q = q / scale
        k = k / scale

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        power = 1 + self.alpha * torch.sigmoid(self.power)
        q_pos = self.kernel_function(q) ** power
        q_neg = self.kernel_function(-q) ** power
        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)

        k_pos = self.kernel_function(k) ** power
        k_neg = self.kernel_function(-k) ** power
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z

        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)

        v_dwc = torch.zeros_like(x)

        if N == H * W:
            v_dwc_input = v.transpose(1, 2).reshape(B, n, C).permute(0, 2, 1)

            if n != N:
                v_dwc_input = F.interpolate(v_dwc_input, size=N, mode='linear', align_corners=False)

            v_dwc_input = v_dwc_input.reshape(B, C, H, W)

            v_dwc_input = v_dwc_input.reshape(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2).reshape(
                B * self.num_heads, H, W, self.head_dim).permute(0, 3, 1, 2)

            v_dwc = self.dwc(v_dwc_input).permute(0, 2, 3, 1).reshape(B, self.num_heads, N, self.head_dim).transpose(1,
                                                                                                                     2).reshape(
                B, N, C)

        x = x + v_dwc

        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)
        return x != H * W
        pass

        v_dwc = torch.zeros_like(x)
        if N == H * W:
            v_dwc_input = v.transpose(1, 2).reshape(B, n, C).permute(0, 2, 1)

            if n != N:
                v_dwc_input = F.interpolate(v_dwc_input, size=N, mode='linear', align_corners=False)

            v_dwc_input = v_dwc_input.reshape(B, C, H, W)

            v_dwc_input = v_dwc_input.reshape(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2).reshape(
                B * self.num_heads, H, W, self.head_dim).permute(0, 3, 1, 2)

            v_dwc = self.dwc(v_dwc_input).permute(0, 2, 3, 1).reshape(B, self.num_heads, N, self.head_dim).transpose(1,
                                                                                                                     2).reshape(
                B, N, C)

        x = x + v_dwc 
        x = x * g  

        pass
        x = x + v_dwc  
        x = x * g  

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape

        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                      3)  
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 3,
                                                                                      1) 
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                      3) 


        attn = (q @ k) * self.scale  # (B, num_heads, N, M)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dropout=0.1,
            num_patches=64,  
            sr_ratio=1,
            kernel_size=5,
            alpha=4,
    ):
        super().__init__()

        self.self_attn = PolaLinearAttention(
            dim=d_model,
            num_patches=num_patches,
            num_heads=nhead,
            proj_drop=dropout,
            sr_ratio=sr_ratio,
            kernel_size=kernel_size,
            alpha=alpha
        )

        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )


    def forward(self, x, mem, H, W):
       
        q = k = v = self.norm1(x)
        attn_out = self.self_attn(q, H, W)
        x = x + attn_out  

        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)

        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=512,
                 dropout=0.1,
                 num_patches=64,
                 sr_ratio=1,
                 kernel_size=5,
                 alpha=4,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(
                transformer_width,
                transformer_heads,
                dropout,
                num_patches=num_patches,
                sr_ratio=sr_ratio,
                kernel_size=kernel_size,
                alpha=alpha
            ) for _ in range(transformer_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual, H, W):
        B, N, C = visual.shape

        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        for layer in self.decoder:
            x = layer(x, visual, H, W)

        return self.out_proj(x)
        
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // ratio, kernel_size=1, bias=False)
        self.act = nn.SiLU()  
        self.fc2 = nn.Conv2d(channels // ratio, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='sigmoid')

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        out = avg_out + max_out
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        scale = self.sigmoid(out)

        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout2d(p=0.5)

        self.conv7 = nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=True)
        self.act = nn.SiLU() 
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.conv7.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='sigmoid')

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_min, _ = torch.min(x, dim=1, keepdim=True)
        x_sum = torch.sum(x, dim=1, keepdim=True)

        feats = torch.cat([x_mean, x_max, x_min, x_sum], dim=1)
        feats = self.dropout(feats)

        feats = self.conv7(feats)
        feats = self.act(feats)
        feats = self.conv1(feats)
        mask = self.sigmoid(feats)

        return x * mask

class CA_SA(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        ca_x = self.ca(x)
        sa_x = self.sa(x)
        return ca_x + sa_x



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, rate=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=rate,
            dilation=rate, groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(inplace=True)

        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

class CASAB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = ConvBlock(channels, channels)
        self.ca_sa = CA_SA(channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.ca_sa(x)
        return x


class TotalModel(nn.Module):
    def __init__(self, clip_model, gpu, max_depth, min_depth, n_bins):
        super(TotalModel, self).__init__()
        self.clip_model=clip_model
        self.gpu=gpu

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.n_bins = n_bins

        self.context_length=8
        self.token_embed_dim=512

        self.contexts = nn.Parameter(torch.randn(1, self.context_length, self.token_embed_dim))

        nn.init.trunc_normal_(self.contexts)

        self.context_decoder = ContextDecoder(visual_dim=512)
        self.gamma = nn.Parameter(torch.ones(512) * 1e-4)

        self.decoder = DecoderBN(num_Bins=256)
        self.computeAdaptiveBin = ComputeAdaptiveBins(embedding_dim=512, max_val=self.max_depth)

        self.predict_D = Predict_Depthmap(n_bins=256)


    def forward(self, img):
        # extract image embedding

        encoded_features = self.clip_model.encode_image(img)
        global_f, visual_embeddings = encoded_features[4]
        B, C, H, W = visual_embeddings.shape
        
        # generate adaptive bins
        bin_edges, centers_, CaBins = self.computeAdaptiveBin(global_f)

        texts = torch.cat([tokenize(c, context_length=30 - self.context_length) for c in CaBins]).to(self.gpu)
        texts = texts.reshape(B, len(CaBins)//B, 30 - self.context_length)
        texts = texts.to(self.clip_model.token_embedding.weight.device)
        learnable = self.contexts.to(self.gpu)
        learnable = learnable.expand(B, -1, -1)
        device = img.device
        texts = texts.to(device)
        text_features = self.clip_model.encode_learnable_text(texts, learnable)

        visual_context = torch.cat([global_f.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],dim=2).permute(0, 2, 1).contiguous()
        text_diff = self.context_decoder(text_features, visual_context, H=H, W=W)
        text_features = text_features + self.gamma * text_diff 

        visual = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_features, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual, text)
        encoded_features.append(score_map)
        out, features = self.decoder(encoded_features)

        pred = self.predict_D(out, centers_)

        return bin_edges, pred
