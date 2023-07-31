import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Union, List, Tuple


def get_activation(name, inplace=True, lrelu_param=0.2):
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        return nn.LeakyReLU(lrelu_param, inplace=inplace)
    else:
        raise NotImplementedError


class CoordMLP(nn.Module):
    def __init__(self,
                 cin,
                 cout,
                 num_layers,
                 nf=256,
                 dropout=0,
                 activation=None,
                 min_max=None,
                 n_harmonic_functions=10,
                 embedder_scalar=1,
                 embed_concat_pts=True,
                 extra_feat_dim=0,
                 symmetrize=False):
        super().__init__()
        self.extra_feat_dim = extra_feat_dim

        if n_harmonic_functions > 0:
            self.embedder = HarmonicEmbedding(n_harmonic_functions, embedder_scalar)
            dim_in = cin * 2 * n_harmonic_functions
            self.embed_concat_pts = embed_concat_pts
            if embed_concat_pts:
                dim_in += cin
        else:
            self.embedder = None
            dim_in = cin
        
        self.in_layer = nn.Linear(dim_in, nf)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = MLP(nf + extra_feat_dim, cout, num_layers, nf, dropout, activation)
        self.symmetrize = symmetrize
        if min_max is not None:
            self.register_buffer('min_max', min_max)  # Cx2
        else:
            self.min_max = None
        self.bsdf = None

    def forward(self, x, feat=None):
        # x: (B, ..., 3), feat: (B, C)
        assert (feat is None and self.extra_feat_dim == 0) or (feat.shape[-1] == self.extra_feat_dim)
        if self.symmetrize:
            xs, ys, zs = x.unbind(-1)
            x = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x
        
        if self.embedder is not None:
            x_in = self.embedder(x)
            if self.embed_concat_pts:
                x_in = torch.cat([x, x_in], -1)
        else:
            x_in = x
        
        x_in = self.in_layer(x_in)
        if feat is not None:
            for i in range(x_in.dim() - feat.dim()):
                feat = feat.unsqueeze(1)
            feat = feat.expand(*x_in.shape[:-1], -1)
            x_in = torch.concat([x_in, feat], dim=-1)
        out = self.mlp(self.relu(x_in))  # (B, ..., C)
        if self.min_max is not None:
            out = out * (self.min_max[:,1] - self.min_max[:,0]) + self.min_max[:,0]
        return out

    def sample(self, x, feat=None):
        return self.forward(x, feat)


class HarmonicEmbedding(nn.Module):
    def __init__(self, n_harmonic_functions=10, scalar=1):
        """
        Positional Embedding implementation (adapted from Pytorch3D).
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**self.n_harmonic_functions * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**self.n_harmonic_functions * x[..., i])
            ]
        Note that `x` is also premultiplied by `scalar` before
        evaluting the harmonic functions.
        """
        super().__init__()
        self.frequencies = scalar * (2.0 ** torch.arange(n_harmonic_functions))

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies.to(x.device)).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class VGGEncoder(nn.Module):
    def __init__(self, cout, pretrained=False):
        super().__init__()
        if pretrained:
            raise NotImplementedError
        vgg = models.vgg16()
        self.vgg_encoder = nn.Sequential(vgg.features, vgg.avgpool)
        self.linear1 = nn.Linear(25088, 4096)
        self.linear2 = nn.Linear(4096, cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        out = self.relu(self.linear1(self.vgg_encoder(x).view(batch_size, -1)))
        return self.linear2(out)


class ResnetEncoder(nn.Module):
    def __init__(self, cout, pretrained=False):
        super().__init__()
        self.resnet = nn.Sequential(list(models.resnet18(weights="DEFAULT" if pretrained else None).modules())[:-1])
        self.final_linear = nn.Linear(512, cout)

    def forward(self, x):
        return self.final_linear(self.resnet(x))


class Encoder(nn.Module):
    def __init__(self, cin, cout, in_size=128, zdim=None, nf=64, activation=None):
        super().__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16, nf),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16*2, nf*2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*4, nf*4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            # nn.GroupNorm(16*8, nf*8),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        add_downsample = int(np.log2(in_size//128))
        if add_downsample > 0:
            for _ in range(add_downsample):
                network += [
                    nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
                    # nn.GroupNorm(16*8, nf*8),
                    # nn.ReLU(inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                ]

        network += [
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if zdim is None:
            network += [
                nn.Conv2d(nf*8, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                ]
        else:
            network += [
                nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(zdim, cout, kernel_size=1, stride=1, padding=0, bias=False),
                ]

        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class EncoderWithDINO(nn.Module):
    def __init__(self, cin_rgb, cin_dino, cout, in_size=128, zdim=None, nf=64, activation=None):
        super().__init__()
        network_rgb_in = [
            nn.Conv2d(cin_rgb, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16, nf),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16*2, nf*2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*4, nf*4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.network_rgb_in = nn.Sequential(*network_rgb_in)
        network_dino_in = [
            nn.Conv2d(cin_dino, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16, nf),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16*2, nf*2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*4, nf*4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.network_dino_in = nn.Sequential(*network_dino_in)

        network_fusion = [
            nn.Conv2d(nf*4*2, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            # nn.GroupNorm(16*8, nf*8),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        add_downsample = int(np.log2(in_size//128))
        if add_downsample > 0:
            for _ in range(add_downsample):
                network_fusion += [
                    nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
                    # nn.GroupNorm(16*8, nf*8),
                    # nn.ReLU(inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                ]

        network_fusion += [
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if zdim is None:
            network_fusion += [
                nn.Conv2d(nf*8, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                ]
        else:
            network_fusion += [
                nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(zdim, cout, kernel_size=1, stride=1, padding=0, bias=False),
                ]

        if activation is not None:
            network_fusion += [get_activation(activation)]
        self.network_fusion = nn.Sequential(*network_fusion)

    def forward(self, rgb_image, dino_image):
        rgb_feat = self.network_rgb_in(rgb_image)
        dino_feat = self.network_dino_in(dino_image)
        out = self.network_fusion(torch.cat([rgb_feat, dino_feat], dim=1))
        return out.reshape(rgb_image.size(0), -1)


class Encoder32(nn.Module):
    def __init__(self, cin, cout, nf=256, activation=None):
        super().__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
        ]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class MLP(nn.Module):
    def __init__(self, cin, cout, num_layers, nf=256, dropout=0, activation=None):
        super().__init__()
        assert num_layers >= 1
        if num_layers == 1:
            network = [nn.Linear(cin, cout, bias=False)]
        else:
            network = [nn.Linear(cin, nf, bias=False)]
            for _ in range(num_layers-2):
                network += [
                    nn.ReLU(inplace=True),
                    nn.Linear(nf, nf, bias=False)]
                if dropout:
                    network += [nn.Dropout(dropout)]
            network += [
                nn.ReLU(inplace=True),
                nn.Linear(nf, cout, bias=False)]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class Embedding(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64, activation=None):
        super().__init__()
        network = [
            nn.Linear(cin, nf, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf, zdim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(zdim, cout, bias=False)]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input.reshape(input.size(0), -1)).reshape(input.size(0), -1)


## from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.norm_layer = norm_layer
        if norm_layer is not None:
            self.bn1 = norm_layer(planes)
            self.bn2 = norm_layer(planes)

        if inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResEncoder(nn.Module):
    def __init__(self, cin, cout, in_size=128, zdim=None, nf=64, activation=None):
        super().__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            # nn.GroupNorm(16, nf),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            # nn.GroupNorm(16*2, nf*2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            BasicBlock(nf*2, nf*2, norm_layer=None),
            BasicBlock(nf*2, nf*2, norm_layer=None),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            # nn.GroupNorm(16*4, nf*4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            BasicBlock(nf*4, nf*4, norm_layer=None),
            BasicBlock(nf*4, nf*4, norm_layer=None),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            BasicBlock(nf*8, nf*8, norm_layer=None),
            BasicBlock(nf*8, nf*8, norm_layer=None),
        ]

        add_downsample = int(np.log2(in_size//64))
        if add_downsample > 0:
            for _ in range(add_downsample):
                network += [
                    nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
                    # nn.ReLU(inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    BasicBlock(nf*8, nf*8, norm_layer=None),
                    BasicBlock(nf*8, nf*8, norm_layer=None),
                ]

        if zdim is None:
            network += [
                nn.Conv2d(nf*8, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                ]
        else:
            network += [
                nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(zdim, cout, kernel_size=1, stride=1, padding=0, bias=False),
                ]

        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class ViTEncoder(nn.Module):
    def __init__(self, cout, which_vit='dino_vits8', pretrained=False, frozen=False, final_layer_type='none'):
        super().__init__()
        self.ViT = torch.hub.load('facebookresearch/dino:main', which_vit, pretrained=pretrained)
        if frozen:
            for p in self.ViT.parameters():
                p.requires_grad = False
        if which_vit == 'dino_vits8':
            self.vit_feat_dim = 384
            self.patch_size = 8
        elif which_vit == 'dino_vitb8':
            self.vit_feat_dim = 768
            self.patch_size = 8
        
        self._feats = []
        self.hook_handlers = []

        if final_layer_type == 'none':
            pass
        elif final_layer_type == 'conv':
            self.final_layer_patch_out = Encoder32(self.vit_feat_dim, cout, nf=256, activation=None)
            self.final_layer_patch_key = Encoder32(self.vit_feat_dim, cout, nf=256, activation=None)
        elif final_layer_type == 'attention':
            raise NotImplementedError
            self.final_layer = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.fc = nn.Linear(self.vit_feat_dim, cout)
        else:
            raise NotImplementedError
        self.final_layer_type = final_layer_type
    
    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook
    
    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.ViT.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")
    
    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []
    
    def forward(self, x, return_patches=False):
        b, c, h, w = x.shape
        self._feats = []
        self._register_hooks([11], 'key')

        x = self.ViT.prepare_tokens(x)
        for blk in self.ViT.blocks:
            x = blk(x)
        out = self.ViT.norm(x)
        self._unregister_hooks()

        ph, pw = h // self.patch_size, w // self.patch_size
        patch_out = out[:, 1:]  # first is class token
        patch_out = patch_out.reshape(b, ph, pw, self.vit_feat_dim).permute(0, 3, 1, 2)

        patch_key = self._feats[0][:,:,1:]  # B, num_heads, num_patches, dim
        patch_key = patch_key.permute(0, 1, 3, 2).reshape(b, self.vit_feat_dim, ph, pw)

        if self.final_layer_type == 'none':
            global_feat_out = out[:, 0].reshape(b, -1)  # first is class token
            global_feat_key = self._feats[0][:, :, 0].reshape(b, -1)  # first is class token
        elif self.final_layer_type == 'conv':
            global_feat_out = self.final_layer_patch_out(patch_out).view(b, -1)
            global_feat_key = self.final_layer_patch_key(patch_key).view(b, -1)
        elif self.final_layer_type == 'attention':
            raise NotImplementedError
        else:
            raise NotImplementedError
        if not return_patches:
            patch_out = patch_key = None
        return global_feat_out, global_feat_key, patch_out, patch_key


class ArticulationNetwork(nn.Module):
    def __init__(self, net_type, feat_dim, pos_dim, num_layers, nf, n_harmonic_functions=0, embedder_scalar=1, activation=None):
        super().__init__()
        if n_harmonic_functions > 0:
            self.posenc = HarmonicEmbedding(n_harmonic_functions=n_harmonic_functions, scalar=embedder_scalar)
            pos_dim = pos_dim * (n_harmonic_functions * 2 + 1)
        else:
            self.posenc = None
            pos_dim = 4
        cout = 3
        
        if net_type == 'mlp':
            self.network = MLP(
                feat_dim + pos_dim,  # + bone xyz pos and index
                cout,  # We represent the rotation of each bone by its Euler angles ψ, θ, and φ
                num_layers,
                nf=nf,
                dropout=0,
                activation=activation
            )
        elif net_type == 'attention':
            self.in_layer = nn.Sequential(
                nn.Linear(feat_dim + pos_dim, nf),
                nn.GELU(),
                nn.LayerNorm(nf),
            )
            self.blocks = nn.ModuleList([
            Block(
                dim=nf, num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)
            for i in range(num_layers)])
            out_layer = [nn.Linear(nf, cout)]
            if activation:
                out_layer += [get_activation(activation)]
            self.out_layer = nn.Sequential(*out_layer)
        else:
            raise NotImplementedError
        self.net_type = net_type
    
    def forward(self, x, pos):
        if self.posenc is not None:
            pos = torch.cat([pos, self.posenc(pos)], dim=-1)
        x = torch.cat([x, pos], dim=-1)
        if self.net_type == 'mlp':
            out = self.network(x)
        elif self.net_type == 'attention':
            x = self.in_layer(x)
            for blk in self.blocks:
                x = blk(x)
            out = self.out_layer(x)
        else:
            raise NotImplementedError
        return out


## Attention block from ViT (https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
