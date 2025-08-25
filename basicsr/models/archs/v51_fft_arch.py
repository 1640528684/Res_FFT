# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from einops import rearrange
from .layers import *
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint  # 用于梯度检查点

class SimpleGate(nn.Module):
    def __init__(self, dim):
        super(SimpleGate,self).__init__()
        self.norm = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return self.norm(x1) * x2
        # return x1 * x2
class SimpleGate2(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
# 新增FFT-ReLU块实现
# class ResBlock_do_fft_bench(nn.Module):
#     def __init__(self, out_channel, norm='backward'):
#         super(ResBlock_do_fft_bench, self).__init__()
#         self.main = nn.Sequential(
#             BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
#             BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
#         )
#         self.main_fft = nn.Sequential(
#             BasicConv(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
#             BasicConv(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
#         )
#         self.dim = out_channel
#         self.norm = norm
        
#     def forward(self, x):
#         _, _, H, W = x.shape
#         dim = 1
        
#         # FFT处理分支
#         y_fft = torch.fft.rfft2(x, norm=self.norm)
#         y_imag = y_fft.imag
#         y_real = y_fft.real
#         y_f = torch.cat([y_real, y_imag], dim=dim)
#         y_f = self.main_fft(y_f)
#         y_real, y_imag = torch.chunk(y_f, 2, dim=dim)
#         y_fft = torch.complex(y_real, y_imag)
#         y_fft = torch.fft.irfft2(y_fft, s=(H, W), norm=self.norm)
        
#         # 主处理分支
#         y_main = self.main(x)
        
#         # 残差连接
#         return y_main + x + y_fft
class FFTBlock(nn.Module):
    def __init__(self, in_channels, mid_channels=None, norm='backward'):
        super().__init__()
        self.norm = norm
        # 压缩通道数（默认压缩为原通道的1/2，可根据情况调整）
        self.mid_channels = mid_channels if mid_channels is not None else max(1, in_channels // 2)
        self.conv_compress = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False)
        self.conv_restore = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=False)
        # FFT处理后的特征变换
        self.conv_fft = nn.Conv2d(self.mid_channels * 2, self.mid_channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # 动态获取输入尺寸（避免固定尺寸导致的内存浪费）
        B, C, H, W = x.shape
        
        # 通道压缩：减少FFT处理的数据量
        x_compress = self.conv_compress(x)  # [B, mid_C, H, W]
        
        # 用checkpoint包装FFT核心操作（用计算时间换内存）
        fft_processed = checkpoint(self._fft_core, x_compress, H, W,self.norm)
        
        # 恢复通道数并残差连接
        x_fft = self.conv_restore(fft_processed)
        return x + x_fft  # 残差连接
    #@staticmethod
    def _fft_core(self,x_compress, H, W,norm):
        
        # 保存原始数据类型（可能是float16）
        orig_dtype = x_compress.dtype
        # 强制将输入转换为float32，因为torch.fft.rfft2不支持float16
        x_compress = x_compress.to(dtype=torch.float32)
        # FFT变换（仅保留必要中间变量）
        x_fft = torch.fft.rfft2(x_compress, norm=norm)  # [B, mid_C, H, W//2+1] (复数)
        # 将复数分解为实部和虚部（合并为通道维度）
        x_fft_real = x_fft.real  # [B, mid_C, H, W//2+1]
        x_fft_imag = x_fft.imag  # [B, mid_C, H, W//2+1]
        # x_fft = torch.cat([x_fft_real, x_fft_imag], dim=1)  # [B, 2*mid_C, H, W//2+1]
        x_fft_stack = torch.cat([x_fft_real, x_fft_imag], dim=1)  # [B, 2*mid_C, H, W//2+1]
        
        # 通过1x1卷积处理频域特征（保持float32）
        x_fft_processed = self.conv_fft(x_fft_stack)  # [B, 2*mid_C, H, W//2+1]
        mid_C = x_fft_processed.shape[1] // 2
        x_fft_real_processed = x_fft_processed[:, :mid_C, ...]
        x_fft_imag_processed = x_fft_processed[:, mid_C:, ...]
        
        # 合并实部和虚部，重建复数张量（float32）
        x_fft_processed = torch.complex(x_fft_real_processed, x_fft_imag_processed)
        # 执行逆FFT（float32支持）
        x_ifft = torch.fft.irfft2(x_fft_processed, s=(H, W), norm=norm)  # [B, mid_C, H, W]
        # 将结果转换回原始数据类型（如float16），保证与网络其他部分精度一致
        x_ifft = x_ifft.to(dtype=orig_dtype)
        return x_ifft
    
        # # 特征处理（缩小尺寸后计算，减少内存）
        # # x_fft = self.act(self.conv_fft(x_fft))  # [B, mid_C, H, W//2+1]
        # x_fft = F.leaky_relu(x_fft, 0.1, inplace=True)  # 替换self.act，避免依赖self
        # x_fft = nn.Conv2d(x_fft.shape[1], x_fft.shape[1]//2, kernel_size=3, padding=1, bias=False).to(x_fft.device)(x_fft)
        # # 逆FFT变换（使用动态尺寸）
        # x_fft = torch.complex(x_fft, torch.zeros_like(x_fft))  # 重构复数
        # x_ifft = torch.fft.irfft2(x_fft, s=(H, W), norm=norm)  # [B, mid_C, H, W]
        # return x_ifft

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, 
                 bias=False, norm=False, relu=True, groups=1):
        super(BasicConv, self).__init__()
        padding = kernel_size // 2
        layers = []
        layers.append(
            nn.Conv2d(in_channel, out_channel, kernel_size, 
                      padding=padding, stride=stride, groups=groups, bias=bias)
        )
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.main(x)

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        # x_patch_fft = torch.fft.rfft2(x_patch.float())
        # x_patch_fft = x_patch_fft * self.fft
        # x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        # 关键：用checkpoint包装分块FFT核心计算
        x_patch = checkpoint(
            self._fftn_core,  # 待包装的核心函数
            x_patch,          # 核心函数参数1：分块后的特征
            self.fft          # 核心函数参数2：FFT权重参数（避免依赖self）
        )
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    # 新增的静态方法：独立封装FFT核心计算（替换原forward中对应的逻辑）
    @staticmethod
    def _fftn_core(x_patch, fft_weight):
        # 分块FFT变换（复数张量，显存占用高）
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        # 应用FFT权重（参数从外部传入，不依赖self）
        x_patch_fft = x_patch_fft * fft_weight
        # 逆FFT变换（恢复分块空间维度）
        x_patch = torch.fft.irfft2(x_patch_fft, s=(x_patch.shape[-2], x_patch.shape[-1]))
        return x_patch


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1, bias=False, LayerNorm_type='WithBias', att=False,fft_block=False, fft_norm='backward'):
        super(TransformerBlock, self).__init__()
        
        self.naf = NAFBlock(dim,fft_block=fft_block, fft_norm=fft_norm)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        inp = x
        # x = self.naf(x)
        x = checkpoint(
            self.naf.forward,  # NAFBlock的forward方法
            x                  # NAFBlock的输入（仅依赖输入x，无其他参数）
        )
        x = x + self.ffn(self.norm2(x))

        return x


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.,fft_block=False, fft_norm='backward'):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        c2wh = dict([(16, 224), (32, 112), (64, 56), (128, 28), (256, 14), (512, 7), (1024, 4)])
        self.sca = MultiSpectralAttentionLayer(dw_channel // 2, c2wh[dw_channel // 2], c2wh[dw_channel // 2],
                                               freq_sel_method='top16')

        # SimpleGate
        self.sg = SimpleGate(dim=dw_channel//2)
        self.sg2 = SimpleGate2()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        
        # 新增Res FFT-ReLU Block
        # 根据参数条件初始化FFT模块（使用传递的fft_norm）
        self.fft_block_enabled = fft_block  # 保存启用标志
        # self.fft_block = ResBlock_do_fft_bench(c, norm=fft_norm) if fft_block else None
        self.fft_block = FFTBlock(c, norm=fft_norm) if fft_block else None
        

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        
        # ============== 关键修改 ==============
        # 在残差路径上插入FFT-ReLU Block
        # residual = inp + x * self.beta
        # fft_out = self.fft_block(residual)
        # y = residual + fft_out
        # if self.fft_block_enabled and self.fft_block is not None:
        #     fft_out = self.fft_block(residual)
        #     y = residual + fft_out
        # else:
        #     y = residual  # 不启用FFT时直接传递残差
        # =====================================
        # FFT模块处理（保留开关逻辑，新增内存优化）
        residual = inp + x * self.beta
        if self.fft_block_enabled and self.fft_block is not None:
            fft_out = self.fft_block(residual)
            y = residual + fft_out
            # 内存优化：释放临时变量
            del fft_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清理CUDA缓存
        else:
            y = residual  # 不启用FFT时直接传递残差
        
        # 后续处理
        y = self.norm2(y)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        y = self.dropout2(y)

        y + self.gamma * residual

        return y


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],fft_block=False,fft_norm='backward'):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)
        self.fft_block = fft_block
        self.fft_norm = fft_norm

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[TransformerBlock(chan,fft_block=fft_block,fft_norm=fft_norm) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[TransformerBlock(chan,fft_block=fft_block,fft_norm=fft_norm) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[TransformerBlock(chan,fft_block=fft_block,fft_norm=fft_norm) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x1 = x
        # x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class v51fftLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        # base_size = (512, 512)

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 16] #enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                 enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
