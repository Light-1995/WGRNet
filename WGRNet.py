import os
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from LearnableWT import Learnable_DWT, Learnable_IDWT
from LearnableWT import get_filter_tensors
import pywt


class ChannelAmplifier(nn.Module):
    def __init__(self, in_channel, target_channel):
        super(ChannelAmplifier, self).__init__()
        self.amplify_path = nn.Sequential(
            nn.Conv2d(in_channel, target_channel, 5, 1, 2, bias=True),
            nn.PReLU(num_parameters=target_channel, init=0.01),
            nn.Conv2d(target_channel, target_channel, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        return self.amplify_path(x)


class ChannelCompressor(nn.Module):
    def __init__(self, ms_target_channel, L_up_channel):
        super(ChannelCompressor, self).__init__()
        self.compress_path = nn.Sequential(
            nn.Conv2d(ms_target_channel, ms_target_channel, 3, 1, 1, bias=True),
            nn.PReLU(num_parameters=ms_target_channel, init=0.01),
            nn.Conv2d(ms_target_channel, L_up_channel, 3, 1, 1, bias=True),
            nn.Conv2d(L_up_channel, L_up_channel, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        return self.compress_path(x)


class FeatureProcessor(nn.Module):
    def __init__(self, in_channel, FFN_channel, out_channel):
        super(FeatureProcessor, self).__init__()
        self.FFN_channel, self.out_channel = FFN_channel, out_channel
        self.linear_1 = nn.Linear(in_channel, FFN_channel)
        self.conv1 = nn.Conv2d(FFN_channel, FFN_channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(FFN_channel, FFN_channel, 1, 1, 0, bias=True)
        self.linear_2 = nn.Linear(FFN_channel, out_channel)
        self.act = nn.PReLU(num_parameters=FFN_channel, init=0.01)

    def forward(self, x):
        B, C, H, W = x.shape
        rs1 = self.linear_1(x.permute(0, 2, 3, 1).reshape(B, -1, C)).permute(0, 2, 1).reshape(B, self.FFN_channel, H, W)
        rs2 = self.act(self.conv1(rs1))
        rs3 = self.conv2(rs2) + rs1
        rs4 = self.linear_2(rs3.permute(0, 2, 3, 1).reshape(B, -1, self.FFN_channel)).permute(0, 2, 1).reshape(B,
                                                                                                               self.out_channel,
                                                                                                               H, W)
        return rs4


class SpatialEnhancer(nn.Module):
    def __init__(self, in_channel, FFN_channel, out_channel):
        super(SpatialEnhancer, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, FFN_channel, 3, 1, 2, bias=True, dilation=2)
        self.conv2 = nn.Conv2d(FFN_channel, FFN_channel, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(FFN_channel, FFN_channel, 1, 1, 0, bias=True)
        self.conv4 = nn.Conv2d(FFN_channel, out_channel, 3, 1, 1, bias=True)
        self.act = nn.PReLU(num_parameters=FFN_channel, init=0.01)

    def forward(self, x):
        rs1 = self.conv1(x)
        rs2 = self.act(self.conv2(rs1))
        rs3 = self.conv3(rs2) + rs1
        rs4 = self.conv4(rs3)
        return rs4


class IDWTProcessor(nn.Module):
    def __init__(self, channel, rec_lo, rec_hi):
        super(IDWTProcessor, self).__init__()
        self.res_block = ResidualBlock(channel=channel)
        self.IDWT = Learnable_IDWT(rec_lo, rec_hi)

    def forward(self, x):
        rs1 = self.IDWT(x)
        rs2 = self.res_block(rs1)
        return rs2


class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
        self.act = nn.PReLU(num_parameters=channel, init=0.01)

    def forward(self, x):
        rs1 = self.act(self.conv1(x))
        rs2 = self.conv2(rs1) + x
        return rs2


class ChannelGating(nn.Module):
    def __init__(self, channel):
        super(ChannelGating, self).__init__()
        self.linear = nn.Linear(channel, channel, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        rs1 = self.linear(x.permute(0, 2, 3, 1).reshape(B, -1, C))
        rs2 = self.sigmoid(rs1).permute(0, 2, 1).reshape(B, C, H, W)
        return rs2


class SpectralFocus(nn.Module):
    def __init__(self, channel, head_channel, dropout):
        super(SpectralFocus, self).__init__()
        self.head_channel, self.channel = head_channel, channel
        self.q = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.k = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.v = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.scale = head_channel ** 0.5
        self.num_head = channel // self.head_channel
        self.mlp_1 = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.mlp_2 = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=True),
            nn.LeakyReLU(0.01),
            nn.Linear(channel * 2, channel, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, q, k, v):
        B, q_C, H, W = q.shape
        _, v_C, _, _ = v.shape
        q_attn = self.q(q.permute(0, 2, 3, 1).reshape(B, -1, q_C)).reshape(B, -1, self.num_head,
                                                                           self.head_channel).permute(0, 2, 1, 3)
        k_attn = self.k(k.permute(0, 2, 3, 1).reshape(B, -1, q_C)).reshape(B, -1, self.num_head,
                                                                           self.head_channel).permute(0, 2, 3, 1)
        v_attn_1 = self.v(v.permute(0, 2, 3, 1).reshape(B, -1, v_C))
        v_attn = v_attn_1.reshape(B, -1, self.num_head, self.head_channel).permute(0, 2, 1, 3)
        attn = ((q_attn @ k_attn) / self.scale).softmax(dim=-1)
        x = (attn @ v_attn).permute(0, 2, 1, 3).reshape(B, -1, v_C)
        rs1 = v_attn_1.permute(0, 2, 1).reshape(B, q_C, H, W) + self.mlp_1(x).permute(0, 2, 1).reshape(B, v_C, H, W)
        rs2 = rs1 + self.mlp_2(rs1.permute(0, 2, 3, 1).reshape(B, -1, v_C)).permute(0, 2, 1).reshape(B, v_C, H, W)
        return rs2


class TripathFusion(nn.Module):
    def __init__(self, channel):
        super(TripathFusion, self).__init__()
        self.resblock = ResidualBlock(channel=channel)
        self.a = nn.Parameter(torch.tensor(0.33), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(0.33), requires_grad=True)

    def forward(self, x1, x2, x3):
        rs1 = self.a * x1 + self.b * x2 + (1 - self.a - self.b) * x3
        rs2 = self.resblock(rs1)
        return rs2


class EdgeAwareBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, pool_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.grad_x = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels, bias=False
        )
        self.grad_y = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), groups=in_channels, bias=False
        )
        self._init_gradient_kernels()
        self.gradient_conv = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _init_gradient_kernels(self):
        kernel_x = torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 3)
        kernel_y = torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 1)
        with torch.no_grad():
            self.grad_x.weight.copy_(kernel_x.repeat(self.grad_x.out_channels, 1, 1, 1))
            self.grad_y.weight.copy_(kernel_y.repeat(self.grad_y.out_channels, 1, 1, 1))

    def forward(self, x):
        low_freq = self.avg_pool(x)
        high_freq = x - low_freq
        grad_x = self.grad_x(high_freq)
        grad_y = self.grad_y(high_freq)
        grad_feat = torch.cat([grad_x, grad_y], dim=1)
        out = self.gradient_conv(grad_feat)
        return out


class SubSpectralTransformer(nn.Module):
    def __init__(self, pan_ll_channel, L_up_channel, head_channel, dropout, dec_lo, dec_hi, rec_lo, rec_hi):
        super(SubSpectralTransformer, self).__init__()
        self.pan_ll_channel = pan_ll_channel
        self.WD = Learnable_DWT(dec_lo, dec_hi, rec_lo, rec_hi, wavelet='haar', level=1, mode="replicate")
        self.v_ll_attn = SpectralFocus(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.v_lh_attn = SpectralFocus(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.v_hl_attn = SpectralFocus(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.v_hh_attn = SpectralFocus(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.conv_idwt_pan = IDWTProcessor(channel=pan_ll_channel, rec_lo=rec_lo, rec_hi=rec_hi)
        self.conv_idwt_up = IDWTProcessor(channel=L_up_channel, rec_lo=rec_lo, rec_hi=rec_hi)
        self.fusion_block = TripathFusion(channel=L_up_channel)
        self.resblock = ResidualBlock(channel=L_up_channel)
        self.resblock_1 = ResidualBlock(channel=L_up_channel)
        self.mlp = FeatureProcessor(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.conv_x = SpatialEnhancer(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.conv_v = SpatialEnhancer(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.ll_conv = nn.Conv2d(pan_ll_channel, pan_ll_channel, kernel_size=3, padding=1)
        self.lh_conv = nn.Conv2d(pan_ll_channel, pan_ll_channel, kernel_size=3, padding=1)
        self.hl_conv = nn.Conv2d(pan_ll_channel, pan_ll_channel, kernel_size=3, padding=1)
        self.hh_conv = nn.Conv2d(pan_ll_channel, pan_ll_channel, kernel_size=3, padding=1)
        self.fusion_conv = nn.Conv2d(pan_ll_channel * 4, pan_ll_channel, kernel_size=1)
        self.edge_block = EdgeAwareBlock(in_channels=pan_ll_channel, out_channels=pan_ll_channel)

    def forward(self, pan_ll, L_up, back_img):
        wd_ll, wd_lh, wd_hl, wd_hh = torch.split(self.WD(pan_ll), [self.pan_ll_channel] * 4, dim=1)
        pre_v = self.fusion_block(x1=wd_ll, x2=L_up, x3=self.mlp(back_img))
        v = self.resblock(pre_v)
        ll_feat = self.ll_conv(wd_ll)
        lh_feat = self.lh_conv(wd_lh)
        hl_feat = self.hl_conv(wd_hl)
        hh_feat = self.hh_conv(wd_hh)
        concatenated_features = torch.cat([ll_feat, lh_feat, hl_feat, hh_feat], dim=1)
        spatial_key = self.fusion_conv(concatenated_features)
        v_ll = self.v_ll_attn(q=wd_ll, k=spatial_key, v=v)
        v_lh = self.v_lh_attn(q=wd_lh, k=spatial_key, v=v)
        v_hl = self.v_hl_attn(q=wd_hl, k=spatial_key, v=v)
        v_hh = self.v_hh_attn(q=wd_hh, k=spatial_key, v=v)
        v_idwt = self.conv_idwt_up(torch.cat([v_ll, v_lh, v_hl, v_hh], dim=1))
        x_gsrb = self.edge_block(pan_ll)
        x_1 = self.conv_x(x_gsrb) + self.conv_v(v_idwt)
        x = self.resblock_1(x_1)
        return x


class FinalSpectralFuser(nn.Module):
    def __init__(self, pan_channel, L_up_channel, head_channel, dropout, dec_lo, dec_hi, rec_lo, rec_hi):
        super(FinalSpectralFuser, self).__init__()
        self.sub_transformer = SubSpectralTransformer(
            pan_ll_channel=pan_channel,
            L_up_channel=L_up_channel,
            head_channel=head_channel,
            dropout=dropout,
            dec_lo=dec_lo,
            dec_hi=dec_hi,
            rec_lo=rec_lo,
            rec_hi=rec_hi
        )
        self.fusion_block = TripathFusion(channel=L_up_channel)
        self.mlp = FeatureProcessor(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.resblock = ResidualBlock(channel=L_up_channel)

    def forward(self, pan, L_up, back_img, lms):
        x = self.sub_transformer(pan_ll=pan, L_up=L_up, back_img=back_img)
        x = self.fusion_block(x1=pan, x2=lms, x3=self.mlp(x))
        x = self.resblock(x)
        return x


class LowFreqSpectralNet(nn.Module):
    def __init__(self, pan_ll_channel, L_up_channel, head_channel, dropout, dec_lo, dec_hi, rec_lo, rec_hi):
        super(LowFreqSpectralNet, self).__init__()
        self.pan_ll_channel = pan_ll_channel
        self.WD = Learnable_DWT(dec_lo, dec_hi, rec_lo, rec_hi, wavelet='haar', level=1, mode="replicate")
        self.v_ll_attn = SpectralFocus(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.v_lh_attn = SpectralFocus(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.v_hl_attn = SpectralFocus(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.v_hh_attn = SpectralFocus(channel=L_up_channel, head_channel=head_channel, dropout=dropout)
        self.mlp = FeatureProcessor(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.conv_idwt_pan = IDWTProcessor(channel=pan_ll_channel, rec_lo=rec_lo, rec_hi=rec_hi)
        self.conv_idwt_up = IDWTProcessor(channel=L_up_channel, rec_lo=rec_lo, rec_hi=rec_hi)
        self.fusion_block = TripathFusion(channel=L_up_channel)
        self.resblock = ResidualBlock(channel=L_up_channel)
        self.resblock_1 = ResidualBlock(channel=L_up_channel)
        self.conv_x = SpatialEnhancer(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.conv_v = SpatialEnhancer(in_channel=L_up_channel, FFN_channel=L_up_channel // 2, out_channel=L_up_channel)
        self.ll_conv = nn.Conv2d(pan_ll_channel, pan_ll_channel, kernel_size=3, padding=1)
        self.lh_conv = nn.Conv2d(pan_ll_channel, pan_ll_channel, kernel_size=3, padding=1)
        self.hl_conv = nn.Conv2d(pan_ll_channel, pan_ll_channel, kernel_size=3, padding=1)
        self.hh_conv = nn.Conv2d(pan_ll_channel, pan_ll_channel, kernel_size=3, padding=1)
        self.fusion_conv = nn.Conv2d(pan_ll_channel * 4, pan_ll_channel, kernel_size=1)
        self.edge_block = EdgeAwareBlock(in_channels=pan_ll_channel, out_channels=pan_ll_channel)

    def forward(self, pan_ll, back_img, L_up):
        wd_ll, wd_lh, wd_hl, wd_hh = torch.split(self.WD(pan_ll), [self.pan_ll_channel] * 4, dim=1)
        pre_v = self.fusion_block(x1=wd_ll, x2=L_up, x3=self.mlp(back_img))
        v = self.resblock(pre_v)
        ll_feat = self.ll_conv(wd_ll)
        lh_feat = self.lh_conv(wd_lh)
        hl_feat = self.hl_conv(wd_hl)
        hh_feat = self.hh_conv(wd_hh)
        concatenated_features = torch.cat([ll_feat, lh_feat, hl_feat, hh_feat], dim=1)
        spatial_key = self.fusion_conv(concatenated_features)
        v_ll = self.v_ll_attn(q=wd_ll, k=spatial_key, v=v)
        v_lh = self.v_lh_attn(q=wd_lh, k=spatial_key, v=v)
        v_hl = self.v_hl_attn(q=wd_hl, k=spatial_key, v=v)
        v_hh = self.v_hh_attn(q=wd_hh, k=spatial_key, v=v)
        v_idwt = self.conv_idwt_up(torch.cat([v_ll, v_lh, v_hl, v_hh], dim=1))
        x_gsrb = self.edge_block(pan_ll)
        x_1 = self.conv_x(x_gsrb) + self.conv_v(v_idwt)
        x = self.resblock_1(x_1)
        return x


class SpectralFusionCore(nn.Module):
    def __init__(self, L_up_channel, pan_channel, pan_target_channel, ms_target_channel, head_channel, dropout):
        super(SpectralFusionCore, self).__init__()
        self.pan_channel = pan_channel
        self.lms = nn.Sequential(
            nn.Conv2d(L_up_channel, L_up_channel * 16, 3, 1, 1, bias=True),
            nn.PixelShuffle(4),
        )
        wavelet = 'haar'
        self.wavelet = _as_wavelet(wavelet)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(wavelet, flip=True, device='cpu')
        initialize = True
        if initialize:
            epsilon = 0.05
            self.dec_lo_L1 = nn.Parameter(dec_lo.clone() + torch.randn_like(dec_lo) * epsilon, requires_grad=True)
            self.dec_hi_L1 = nn.Parameter(dec_hi.clone() + torch.randn_like(dec_hi) * epsilon, requires_grad=True)
            self.rec_lo_L1 = nn.Parameter(rec_lo.flip(-1).clone() + torch.randn_like(rec_lo) * epsilon,
                                          requires_grad=True)
            self.rec_hi_L1 = nn.Parameter(rec_hi.flip(-1).clone() + torch.randn_like(rec_hi) * epsilon,
                                          requires_grad=True)
            self.dec_lo_L2 = nn.Parameter(dec_lo.clone() + torch.randn_like(dec_lo) * epsilon, requires_grad=True)
            self.dec_hi_L2 = nn.Parameter(dec_hi.clone() + torch.randn_like(dec_hi) * epsilon, requires_grad=True)
            self.rec_lo_L2 = nn.Parameter(rec_lo.flip(-1).clone() + torch.randn_like(rec_lo) * epsilon,
                                          requires_grad=True)
            self.rec_hi_L2 = nn.Parameter(rec_hi.flip(-1).clone() + torch.randn_like(rec_hi) * epsilon,
                                          requires_grad=True)
        else:
            self.dec_lo_L1 = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi_L1 = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
            self.rec_lo_L1 = nn.Parameter(torch.rand_like(rec_lo) * 2 - 1, requires_grad=True)
            self.rec_hi_L1 = nn.Parameter(torch.rand_like(rec_hi) * 2 - 1, requires_grad=True)
            self.dec_lo_L2 = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi_L2 = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
            self.rec_lo_L2 = nn.Parameter(torch.rand_like(rec_lo) * 2 - 1, requires_grad=True)
            self.rec_hi_L2 = nn.Parameter(torch.rand_like(rec_hi) * 2 - 1, requires_grad=True)

        self.pan_amplifier = ChannelAmplifier(in_channel=pan_channel, target_channel=pan_target_channel)
        self.lms_amplifier = ChannelAmplifier(in_channel=L_up_channel, target_channel=ms_target_channel)
        self.ms_amplifier = ChannelAmplifier(in_channel=L_up_channel, target_channel=ms_target_channel)
        self.channel_compressor = ChannelCompressor(ms_target_channel=ms_target_channel, L_up_channel=L_up_channel)

        self.Fusion_Block = FinalSpectralFuser(
            L_up_channel=ms_target_channel,
            pan_channel=pan_target_channel,
            head_channel=head_channel,
            dropout=dropout,
            dec_lo=self.dec_lo_L1,
            dec_hi=self.dec_hi_L1,
            rec_lo=self.rec_lo_L1,
            rec_hi=self.rec_hi_L1
        )

        self.LowFreq_Block = LowFreqSpectralNet(
            L_up_channel=ms_target_channel,
            pan_ll_channel=pan_target_channel,
            head_channel=head_channel,
            dropout=dropout,
            dec_lo=self.dec_lo_L2,
            dec_hi=self.dec_hi_L2,
            rec_lo=self.rec_lo_L2,
            rec_hi=self.rec_hi_L2
        )

        self.lms_down_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lms_down_4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pan_down_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.act_1 = nn.PReLU(num_parameters=L_up_channel, init=0.01)
        self.act_2 = nn.PReLU(num_parameters=L_up_channel, init=0.01)

    def forward(self, pan, ms):
        pan = self.pan_amplifier(pan)
        lms_1 = self.act_1(self.lms(ms))
        lms_2 = self.lms_amplifier(lms_1)
        back_1 = self.LowFreq_Block(pan_ll=self.pan_down_2(pan), L_up=self.lms_down_4(lms_2),
                                    back_img=self.ms_amplifier(ms))
        back_2 = self.Fusion_Block(pan=pan, L_up=self.lms_down_2(lms_2), back_img=back_1, lms=lms_2)
        back = self.channel_compressor(back_2)
        result = self.act_2(back + lms_1)
        return result