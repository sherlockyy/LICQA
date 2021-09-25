# Implementation of Vision Transformer
# From https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py
from importlib import import_module
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
import torchvision


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


def get_sinusoid_position_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])   # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])   # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)   #(1,N,d)


class ViT(nn.Module):
    def __init__(self, input_size, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.,
                 backbone='resnet18', use_layer=(3, 4), position_embedding='sine', pretrain='tmp.pt'):
        super().__init__()
        backbone = getattr(torchvision.models, backbone)()
        if pretrain is not None:
            load_from = torch.load(pretrain, {'map_location': lambda storage, loc: storage})
            backbone.load_state_dict(load_from, strict=False)
       
        self.start_layer, self.end_layer = use_layer
        self.chns = [64, 128, 256, 512]
        self.chn_sum = sum(self.chns[self.start_layer-1:self.end_layer])

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.patch_size = int(input_size / (2 ** (self.end_layer + 1)))
        pos_length = self.patch_size ** 2 + 1
        if position_embedding == 'sine':
            self.register_buffer('pos_embedding', get_sinusoid_position_encoding(pos_length, dim))
        else:
            self.register_parameter('pos_embedding', nn.Parameter(torch.randn(1, pos_length, dim)))

        self.patch_to_embedding = nn.Linear(3 * self.chn_sum, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 5)
        )

        self.mlp_class = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 7)
        )

    def forward_once(self, x):
        h0 = self.conv1(x)
        # h0 = self.maxpool(self.relu(self.bn1(h0)))
        h0 = self.relu(self.bn1(h0))

        h1 = self.layer1(h0)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)

        return [h1, h2, h3, h4]

    def forward(self, data, mask=None, batch_avg=False):
        x1, x2 = data
        if len(x1.shape) == 5:
            x1 = x1.squeeze(0)
            x2 = x2.squeeze(0)
            batch_avg = True

        h1s = self.forward_once(x1)
        h2s = self.forward_once(x2)

        hyperfeat_lists = []
        for h1, h2 in zip(h1s[self.start_layer-1:self.end_layer], h2s[self.start_layer-1:self.end_layer]):
            hyper_1 = F.adaptive_max_pool2d(h1, self.patch_size)
            hyper_2 = F.adaptive_max_pool2d(h2, self.patch_size)
            hyperfeat_lists.append(hyper_1)
            hyperfeat_lists.append(hyper_2)
            hyperfeat_lists.append(hyper_1 - hyper_2)
        hyper_feat = torch.cat(hyperfeat_lists, 1)

        x = rearrange(hyper_feat, 'b c h w -> b (h w) c')
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim=1)
        x = self.to_latent(x)
        score_bins = self.mlp_score(x)
        class_bins = self.mlp_class(x)

        xrate_scale = score_bins.new_tensor([float(j) / 5 for j in range(1, 6)])
        x_mean = torch.sum(score_bins * xrate_scale / torch.sum(xrate_scale), dim=-1)

        if batch_avg:
            return class_bins.mean(), x_mean.mean()
        return class_bins, x_mean.squeeze(-1)


def make_model(args):
    use_layer = list(map(lambda x: int(x), args.use_layer.split('-')))
    return ViT(input_size=args.vit_input_size,
               dim=args.linear_dim,
               depth=args.ts_blk,
               heads=args.att_head,
               mlp_dim=args.mlp_dim,
               dropout=args.dropout,
               emb_dropout=args.emb_dropout,
               backbone=args.backbone,
               use_layer=use_layer,
               position_embedding=args.position_embedding,
               pretrain=args.vit_pretrain)
