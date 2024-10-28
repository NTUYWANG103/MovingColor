import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from model.modules.base_module import BaseNetwork
from model.modules.sparse_transformer import TemporalSparseTransformerBlock, SoftSplit, SoftComp
from model.modules.spectral_norm import spectral_norm as _spectral_norm
from model.modules.flow_loss_utils import flow_warp
from model.modules.deformconv import ModulatedDeformConv2d
from model.ffc_arch import *
import os

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)

def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

    # fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).float()
    fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).to(flow_fw)
    return fb_valid_fw
        
        
class DeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""
    def __init__(self, *args, **kwargs):
        # self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 3)

        super(DeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2*self.out_channels + 2 + 1 + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, cond_feat, flow):
        out = self.conv_offset(cond_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, 
                                             self.stride, self.padding,
                                             self.dilation, mask)


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel, learnable=True):
        super(BidirectionalPropagation, self).__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        self.prop_list = ['backward_1', 'forward_1']
        self.learnable = learnable

        if self.learnable:
            for i, module in enumerate(self.prop_list):
                self.deform_align[module] = DeformableAlignment(
                    channel, channel, 3, padding=1, deform_groups=16)

                self.backbone[module] = nn.Sequential(
                    nn.Conv2d(2*channel+2, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                )

            self.fuse = nn.Sequential(
                    nn.Conv2d(2*channel+2, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                ) 
            
    def binary_mask(self, mask, th=0.1):
        mask[mask>th] = 1
        mask[mask<=th] = 0
        # return mask.float()
        return mask.to(mask)

    def forward(self, x, flows_forward, flows_backward, mask, interpolation='bilinear'):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """

        # For backward warping
        # pred_flows_forward for backward feature propagation
        # pred_flows_backward for forward feature propagation
        b, t, c, h, w = x.shape
        feats, masks = {}, {}
        feats['input'] = [x[:, i, :, :, :] for i in range(0, t)]
        masks['input'] = [mask[:, i, :, :, :] for i in range(0, t)]

        prop_list = ['backward_1', 'forward_1']
        cache_list = ['input'] +  prop_list

        for p_i, module_name in enumerate(prop_list):
            feats[module_name] = []
            masks[module_name] = []

            if 'backward' in module_name:
                frame_idx = range(0, t)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, t)
                flow_idx = range(-1, t - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]
                mask_current = masks[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                    flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check)
                    feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)

                    if self.learnable:
                        cond = torch.cat([feat_current, feat_warped, flow_prop, flow_vaild_mask, mask_current], dim=1)
                        feat_prop = self.deform_align[module_name](feat_prop, cond, flow_prop)
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = flow_warp(mask_prop, flow_prop.permute(0, 2, 3, 1))
                        mask_prop_valid = self.binary_mask(mask_prop_valid)

                        union_vaild_mask = self.binary_mask(mask_current*flow_vaild_mask*(1-mask_prop_valid))
                        feat_prop = union_vaild_mask * feat_warped + (1-union_vaild_mask) * feat_current
                        # update mask
                        mask_prop = self.binary_mask(mask_current*(1-(flow_vaild_mask*(1-mask_prop_valid))))
                
                # refine
                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop, mask_current], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)
                    # feat_prop = self.backbone[module_name](feat_prop)

                feats[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)

            # end for
            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        outputs_b = torch.stack(feats['backward_1'], dim=1).view(-1, c, h, w)
        outputs_f = torch.stack(feats['forward_1'], dim=1).view(-1, c, h, w)

        if self.learnable:
            mask_in = mask.view(-1, 2, h, w)
            masks_b, masks_f = None, None
            outputs = self.fuse(torch.cat([outputs_b, outputs_f, mask_in], dim=1)) + x.reshape(-1, c, h, w)
        else:
            masks_b = torch.stack(masks['backward_1'], dim=1)
            masks_f = torch.stack(masks['forward_1'], dim=1)
            outputs = outputs_f

        return outputs_b.view(b, -1, c, h, w), outputs_f.view(b, -1, c, h, w), \
               outputs.view(b, -1, c, h, w), masks_f


class MCGenerator(BaseNetwork):
    def __init__(self, init_weights=True, model_path=None):
        super(MCGenerator, self).__init__()
        channel = 512
        hidden = 512

        # define layers for encoder and decoder
        input_nc = 7
        output_nc = 3
        ngf = 64
        n_downsampling=3
        n_blocks=9
        norm_layer=nn.BatchNorm2d
        padding_type='reflect'
        activation_layer=nn.ReLU
        up_norm_layer=nn.BatchNorm2d 
        up_activation=nn.ReLU(True)
        init_conv_kwargs={'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False}
        downsample_conv_kwargs={'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False}
        resnet_conv_kwargs={'ratio_gin': 0.75, 'ratio_gout': 0.75, 'enable_lfu': False}
        spatial_transform_layers=None 
        spatial_transform_kwargs={}
        add_out_act='sigmoid' 
        max_features=1024 
        out_ffc=False 
        out_ffc_kwargs={} 

        self.init_pad = nn.ReflectionPad2d(3)
        self.init_conv = FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                                    activation_layer=activation_layer, **init_conv_kwargs)

        self.downsample_modules = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            self.downsample_modules.append(FFC_BN_ACT(min(max_features, ngf * mult),
                                                      min(max_features, ngf * mult * 2),
                                                      kernel_size=3, stride=2, padding=1,
                                                      norm_layer=norm_layer,
                                                      activation_layer=activation_layer,
                                                      **cur_conv_kwargs))
        
        # aux
        self.aux_conv = FFC_BN_ACT(3, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                                    activation_layer=activation_layer, **init_conv_kwargs)

        self.aux_downsample_modules = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            self.aux_downsample_modules.append(FFC_BN_ACT(min(max_features, ngf * mult),
                                                      min(max_features, ngf * mult * 2),
                                                      kernel_size=3, stride=2, padding=1,
                                                      norm_layer=norm_layer,
                                                      activation_layer=activation_layer,
                                                      **cur_conv_kwargs))


        self.resnet_blocks = nn.ModuleList()
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(min(max_features, ngf * (2 ** n_downsampling)),
                                          padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            self.resnet_blocks.append(cur_resblock)

        self.concat_tuple = ConcatTupleLayer()

        self.upsample_modules = nn.ModuleList()
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            self.upsample_modules.extend([
                nn.ConvTranspose2d(min(max_features, ngf * mult),
                                   min(max_features, int(ngf * mult / 2)),
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                up_norm_layer(min(max_features, int(ngf * mult / 2))),
                up_activation
            ])

        if out_ffc:
            self.out_ffc_block = FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                                norm_layer=norm_layer, inline=True, **out_ffc_kwargs)
        else:
            self.out_ffc_block = None

        self.final_pad = nn.ReflectionPad2d(3)
        self.final_conv = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        
        if add_out_act:
            self.out_activation = get_activation('tanh' if add_out_act is True else add_out_act)
        else:
            self.out_activation = None
        self.output_nc = output_nc


        # soft split and soft composition
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }
        self.ss = SoftSplit(channel, hidden, kernel_size, stride, padding)
        self.sc = SoftComp(channel, hidden, kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)        
        
        depths = 8
        num_heads = 4
        window_size = (5, 9)
        pool_size = (4, 4)
        self.transformers = TemporalSparseTransformerBlock(dim=hidden,
                                                n_head=num_heads,
                                                window_size=window_size,
                                                pool_size=pool_size,
                                                depths=depths,
                                                t2t_params=t2t_params)
        
        # legacy, redundant
        self.img_prop_module = BidirectionalPropagation(3, learnable=False)
        self.feat_prop_module = BidirectionalPropagation(128, learnable=True)

        if init_weights:
            self.init_weights()


        if model_path is not None:
            print('Pretrained model has loaded...')
            ckpt = torch.load(model_path, map_location='cpu')
            self.load_state_dict(ckpt, strict=True)

        # print network parameter number
        self.print_network()

    def forward(self, masked_frames, completed_flows=None, masks_in=None, masks_updated=None, num_local_frames=10, interpolation='bilinear', t_dilation=2):
        l_t = num_local_frames
        b, t, _, ori_h, ori_w = masked_frames.size()

        # extracting features
        x = masked_frames.view(b * t, 7, ori_h, ori_w)
        # print('1',x.shape)
        x_content = x[:, :3].clone()
        x = self.init_pad(x)
        x = self.init_conv(x)
        
        # A list to store the sizes after each downsampling
        sizes_after_downsampling = []
        
        for layer in self.downsample_modules:
            sizes_after_downsampling.append(x[0].size())
            x = layer(x)
            # print('downsample',layer, x[0].shape)

        x_aux = self.init_pad(x_content)
        x_aux = self.aux_conv(x_aux)
        aux_downsample_list = []
        for layer in self.aux_downsample_modules:
            aux_downsample_list.append(x_aux)
            x_aux = layer(x_aux)
            # print('aux downsample', layer, x[0].shape)

        # resnet blocks
        for block in self.resnet_blocks:
            x = block(x)
            # print('ffc resnet blocks', block, x[0].shape)

        x = self.concat_tuple(x)

        enc_feat = x

        _, c, h, w = enc_feat.size()
        # feature propagation
        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]
        fold_feat_size = (h, w)

        enc_feat = torch.cat((local_feat, ref_feat), dim=1) 
        # print('temporal_feat',enc_feat.shape)

        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_feat_size)
        # set mask_pool_l to 1
        mask_pool_l = torch.ones(b, l_t, trans_feat.shape[2], trans_feat.shape[3], 1).to(trans_feat.device)
        trans_feat = self.transformers(trans_feat, fold_feat_size, mask_pool_l, t_dilation=t_dilation)
        trans_feat = self.sc(trans_feat, t, fold_feat_size)
        trans_feat = trans_feat.view(b, t, -1, h, w)
        enc_feat = enc_feat + trans_feat

        x = enc_feat.view(-1, c, h, w)
        # print('final most downsample', self.transformers, x.shape)

        # decoder(upsampling)
        for i, layer in enumerate(self.upsample_modules):
            x = layer(x)

            if isinstance(layer, nn.ConvTranspose2d):
                target_size = sizes_after_downsampling.pop()  # Get the corresponding downsample size
                
                # Calculate padding needed to match target size
                diffY = target_size[2] - x.size()[2]
                diffX = target_size[3] - x.size()[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
                if i in [0,3,6]:
                    x = x + self.concat_tuple(aux_downsample_list.pop())
            # print('upsample', layer, x.shape)
            
            
        if self.out_ffc_block:
            x = self.out_ffc_block(x)

        x = self.final_pad(x)
        x = self.final_conv(x)

        if self.out_activation:
            x = self.out_activation(x)

        output = x
        # print('output', self.final_conv, output.shape)

        if self.training:
            output = output.view(b, t, 3, ori_h, ori_w)
        else:
            output = output.view(b, t, 3, ori_h, ori_w)[:, :l_t, ...]
        # print('output', output.shape)
        return output

class MCGenerator_l1_vgg_gan(MCGenerator):
    pass

# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################
class Discriminator(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(3, 5, 5),
                      stride=(1, 2, 2),
                      padding=(1, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


class Discriminator_2D(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator_2D, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(1, 5, 5),
                          stride=(1, 2, 2),
                          padding=(0, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(1, 5, 5),
                      stride=(1, 2, 2),
                      padding=(0, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out

def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module