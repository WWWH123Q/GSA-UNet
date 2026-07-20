from __future__ import annotations
from typing import Tuple, Union
import torch
from einops import rearrange
from mamba_ssm import Mamba
from torch import nn
import torch.nn.functional as F


class ChebyshevFunction(nn.Module):
    def __init__(self, degree: int = 4):
        super(ChebyshevFunction, self).__init__()
        self.degree = degree

    def forward(self, x):
        chebyshev_polynomials = [torch.ones_like(x), x]
        for n in range(2, self.degree):
            chebyshev_polynomials.append(
                2 * x * chebyshev_polynomials[-1] - chebyshev_polynomials[-2]
            )
        return torch.stack(chebyshev_polynomials, dim=-1)


class SplineConv2D(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        init_scale: float = 0.1,
        padding_mode: str = "zeros",
        **kw,
    ) -> None:
        self.init_scale = init_scale
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            **kw,
        )

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class ChebyshevKANConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        degree: int = 4,
        use_base_update: bool = True,
        base_activation=F.relu,
        spline_weight_init_scale: float = 0.1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.basis = ChebyshevFunction(degree)
        self.spline_conv = SplineConv2D(
            in_channels * degree,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            spline_weight_init_scale,
            padding_mode,
        )
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                padding_mode,
            )

    def forward(self, x):
        (batch_size, channels, height, width) = x.shape
        basis = self.basis(x.view(batch_size, channels, -1)).view(
            batch_size, channels, height, width, -1
        )
        basis = basis.permute(0, 4, 1, 2, 3).contiguous().view(batch_size, -1, height, width)
        ret = self.spline_conv(basis)
        if self.use_base_update:
            base = self.base_conv(self.base_activation(x))
            ret = ret + base
        return ret


class KANEncoder(nn.Module):
    def __init__(self, input_frames, output_frames):
        super(KANEncoder, self).__init__()
        self.input_frames = input_frames
        self.out_frames = output_frames
        self.conv1_1 = nn.Conv2d(
            self.input_frames, self.input_frames, (3, 3), padding=(1, 1), stride=(1, 1)
        )
        self.conv2_1 = nn.Sequential(
            ChebyshevKANConv2d(
                in_channels=self.input_frames,
                out_channels=self.out_frames,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.out_frames),
            nn.ReLU(),
        )

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.gelu(self.conv2_1(x))
        return x


class KANDecoder(nn.Module):
    def __init__(self, input_frames, output_frames):
        super(KANDecoder, self).__init__()
        self.input_frames = input_frames
        self.out_frames = output_frames
        self.conv1_2 = nn.Conv2d(
            self.input_frames, self.input_frames, (3, 3), padding=(1, 1), stride=(1, 1)
        )
        self.conv2_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ChebyshevKANConv2d(
                in_channels=self.input_frames,
                out_channels=self.out_frames,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(self.out_frames),
            nn.ReLU(),
        )

    def forward(self, x):
        x = F.relu(self.conv1_2(x))
        x = F.gelu(self.conv2_2(x))
        return x


class EncoderStage(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_frames = input_dim
        self.output_frames = output_dim
        self.block = KANEncoder(self.input_frames, self.output_frames)

    def forward(self, x):
        return self.block(x)


class DecoderStage(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_frames = input_dim
        self.output_frames = output_dim
        self.block = KANDecoder(self.input_frames, self.output_frames)

    def forward(self, x):
        return self.block(x)


class GlobalSpatialAttention(nn.Module):
    def __init__(self, global_dim, local_dim, kernel_size, pool_size, head, qk_scale=None):
        super().__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.head = head
        self.norm = nn.LayerNorm(global_dim + local_dim)
        self.global_head = int(self.head * self.global_dim / (self.global_dim + self.local_dim))
        self.fc1 = nn.Linear(global_dim, global_dim * 3)
        self.pool1 = nn.AvgPool2d(pool_size)
        self.qk_scale = qk_scale or global_dim ** (-0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.local_head = int(self.head * self.local_dim / (self.global_dim + self.local_dim))
        self.fc2 = nn.Linear(local_dim, local_dim * 3)
        self.qconv = nn.Conv2d(
            local_dim // self.local_head,
            local_dim // self.local_head,
            kernel_size,
            padding=kernel_size // 2,
            groups=local_dim // self.local_head,
        )
        self.kconv = nn.Conv2d(
            local_dim // self.local_head,
            local_dim // self.local_head,
            kernel_size,
            padding=kernel_size // 2,
            groups=local_dim // self.local_head,
        )
        self.vconv = nn.Conv2d(
            local_dim // self.local_head,
            local_dim // self.local_head,
            kernel_size,
            padding=kernel_size // 2,
            groups=local_dim // self.local_head,
        )
        self.fc3 = nn.Conv2d(local_dim // self.local_head, local_dim // self.local_head, 1)
        self.swish = nn.SiLU()
        self.fc4 = nn.Conv2d(local_dim // self.local_head, local_dim // self.local_head, 1)
        self.tanh = nn.Tanh()
        self.fc5 = nn.Conv2d(global_dim + local_dim, global_dim + local_dim, 1)

    def forward(self, x):
        identity = x
        batch_size, _, height, width = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        (x_local, x_global) = torch.split(x, [self.local_dim, self.global_dim], dim=-1)
        global_qkv = self.fc1(x_global)
        global_qkv = rearrange(global_qkv, "b n (m h c) -> m b h n c", m=3, h=self.global_head)
        (global_q, global_k, global_v) = (global_qkv[0], global_qkv[1], global_qkv[2])
        global_k = rearrange(global_k, "b h (n1 n2) c -> b (h c) n1 n2", n1=height, n2=width)
        global_k = self.pool1(global_k)
        global_k = rearrange(global_k, "b (h c) n1 n2 -> b h (n1 n2) c", h=self.global_head)
        global_v = rearrange(global_v, "b h (n1 n2) c -> b (h c) n1 n2", n1=height, n2=width)
        global_v = self.pool1(global_v)
        global_v = rearrange(global_v, "b (h c) n1 n2 -> b h (n1 n2) c", h=self.global_head)
        attn = global_q @ global_k.transpose(-2, -1) * self.qk_scale
        attn = self.softmax(attn)
        x_global = attn @ global_v
        x_global = rearrange(x_global, "b h (n1 n2) c -> b (h c) n1 n2", n1=height, n2=width)
        local_qkv = self.fc2(x_local)
        local_qkv = rearrange(
            local_qkv,
            "b (n1 n2) (m h c) -> m (b h) c n1 n2",
            m=3,
            n1=height,
            n2=width,
            h=self.local_head,
        )
        (local_q, local_k, local_v) = (local_qkv[0], local_qkv[1], local_qkv[2])
        local_q = self.qconv(local_q)
        local_k = self.kconv(local_k)
        local_v = self.vconv(local_v)
        attn = local_q * local_k
        attn = self.fc4(self.swish(self.fc3(attn)))
        attn = self.tanh(attn / self.local_dim ** (-0.5))
        x_local = attn * local_v
        x_local = rearrange(x_local, "(b h) c n1 n2 -> b (h c) n1 n2", b=batch_size)
        x = torch.cat([x_local, x_global], dim=1)
        x = self.fc5(x)
        out = identity + x
        return out


class MambaBridge(nn.Module):
    def __init__(self, input_dim, output_dim=None, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(d_model=input_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        (B, C) = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class GSAUNet(nn.Module):
    """GSA-UNet for multi-frame precipitation nowcasting."""

    def __init__(self, num_classes=3, input_channels=5, channels=(16, 32, 64, 128), bridge=True):
        super().__init__()
        head_counts = (4, 8, 8, 8)
        if len(channels) != 4 or any(
            channel % heads for channel, heads in zip(channels, head_counts)
        ):
            raise ValueError("channels must contain four values divisible by 4, 8, 8, and 8")
        self.bridge = bridge
        self.encoder1 = nn.Sequential(
            EncoderStage(input_dim=input_channels, output_dim=channels[0])
        )
        self.encoder2 = nn.Sequential(
            GlobalSpatialAttention(
                global_dim=channels[0] // 2,
                local_dim=channels[0] // 2,
                kernel_size=3,
                head=4,
                pool_size=2,
            ),
            EncoderStage(input_dim=channels[0], output_dim=channels[1]),
        )
        self.encoder3 = nn.Sequential(
            GlobalSpatialAttention(
                global_dim=channels[1] // 2,
                local_dim=channels[1] // 2,
                kernel_size=3,
                head=8,
                pool_size=2,
            ),
            EncoderStage(input_dim=channels[1], output_dim=channels[2]),
        )
        self.encoder4 = nn.Sequential(
            GlobalSpatialAttention(
                global_dim=channels[2] // 2,
                local_dim=channels[2] // 2,
                kernel_size=3,
                head=8,
                pool_size=2,
            ),
            EncoderStage(input_dim=channels[2], output_dim=channels[3]),
        )
        if bridge:
            self.bridge1 = MambaBridge(channels[0])
            self.bridge2 = MambaBridge(channels[1])
            self.bridge3 = MambaBridge(channels[2])
        self.decoder1 = nn.Sequential(
            GlobalSpatialAttention(
                global_dim=channels[3] // 2,
                local_dim=channels[3] // 2,
                kernel_size=3,
                head=8,
                pool_size=2,
            ),
            DecoderStage(input_dim=channels[3], output_dim=channels[2]),
        )
        self.decoder2 = nn.Sequential(
            GlobalSpatialAttention(
                global_dim=channels[2] // 2,
                local_dim=channels[2] // 2,
                kernel_size=3,
                head=8,
                pool_size=2,
            ),
            DecoderStage(input_dim=channels[2], output_dim=channels[1]),
        )
        self.decoder3 = nn.Sequential(
            GlobalSpatialAttention(
                global_dim=channels[1] // 2,
                local_dim=channels[1] // 2,
                kernel_size=3,
                head=8,
                pool_size=2,
            ),
            DecoderStage(input_dim=channels[1], output_dim=channels[0]),
        )
        self.decoder4 = nn.Sequential(
            GlobalSpatialAttention(
                global_dim=channels[0] // 2,
                local_dim=channels[0] // 2,
                kernel_size=3,
                head=4,
                pool_size=2,
            ),
            DecoderStage(input_dim=channels[0], output_dim=input_channels),
        )
        self.output_head = nn.Conv2d(input_channels, num_classes, 3, 1, 1)

    def forward(self, x):
        out = self.encoder1(x)
        t1 = out
        out = self.encoder2(out)
        t2 = out
        out = self.encoder3(out)
        t3 = out
        out = self.encoder4(out)
        if self.bridge:
            t1 = self.bridge1(t1)
            t2 = self.bridge2(t2)
            t3 = self.bridge3(t3)
        out4 = self.decoder1(out)
        out4 = torch.add(out4, t3)
        out3 = self.decoder2(out4)
        out3 = torch.add(out3, t2)
        out2 = self.decoder3(out3)
        out2 = torch.add(out2, t1)
        out1 = self.decoder4(out2)
        out1 = torch.add(out1, x[:, -1, ...].unsqueeze(1))
        return self.output_head(out1)
