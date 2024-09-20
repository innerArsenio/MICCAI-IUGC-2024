from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from typing import Optional




class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, mode="seg"):
        super(U_Net, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.mode = mode

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.norm = nn.LayerNorm(1024, eps=1e-6)  # final norm layer
        # self.head = nn.Linear(1024, 2)
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.activation = nn.Sigmoid()

    def forward(self, x, mask_type=False, mode="inference"):
        # Initial encoder forward pass (common for all samples)
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        if mask_type!=False:
            seg_outputs = torch.zeros(len(mask_type), 3, e1.size(-2), e1.size(-1), device=x.device)
        
        if mask_type!=False:
            if mode == "seg" or mode == "inference":
                # Mask for negative samples
                non_negative_mask = [t != 'negative' for t in mask_type]
                non_negative_mask = torch.tensor(non_negative_mask, device=x.device)

                # Only process segmentation for non-negative samples
                if non_negative_mask.any():
                    d5 = self.Up5(e5[non_negative_mask])
                    d5 = torch.cat((e4[non_negative_mask], d5), dim=1)
                    d5 = self.Up_conv5(d5)
                    d4 = self.Up4(d5)
                    d4 = torch.cat((e3[non_negative_mask], d4), dim=1)
                    d4 = self.Up_conv4(d4)
                    d3 = self.Up3(d4)
                    d3 = torch.cat((e2[non_negative_mask], d3), dim=1)
                    d3 = self.Up_conv3(d3)
                    d2 = self.Up2(d3)
                    d2 = torch.cat((e1[non_negative_mask], d2), dim=1)
                    seg_output = self.Up_conv2(d2)
                    seg_output = self.Conv(seg_output)
                    seg_output = seg_output.to(seg_outputs.dtype)
                    seg_outputs[non_negative_mask] = seg_output


            if mode == "cls" or mode == "inference":
                cls_output = self.norm(e5.mean([-2, -1]))  # Compute classification for all samples
                cls_output = self.head(cls_output)
                cls_outputs = cls_output

            if mode == "cls":
                return cls_outputs

            if mode == "inference":
                return cls_outputs, seg_outputs if seg_outputs is not None else torch.zeros_like(e1)

            return seg_outputs
        else:
            if mode == "seg":
                d5 = self.Up5(e5)
                d5 = torch.cat((e4, d5), dim=1)

                d5 = self.Up_conv5(d5)

                d4 = self.Up4(d5)
                d4 = torch.cat((e3, d4), dim=1)
                d4 = self.Up_conv4(d4)

                d3 = self.Up3(d4)
                d3 = torch.cat((e2, d3), dim=1)
                d3 = self.Up_conv3(d3)

                d2 = self.Up2(d3)
                d2 = torch.cat((e1, d2), dim=1)
                d2 = self.Up_conv2(d2)

                out = self.Conv(d2)
                return out
            elif mode == "cls":
                out = self.norm(e5.mean([-2, -1]))
                out = self.head(out)
                return out
                # out = self.activation(out)
            elif mode == "inference":
                out = self.norm(e5.mean([-2, -1]))
                out = self.head(out)  # (B,2)
                cls = out.argmax(dim=-1)
                seg = None
                if cls:
                    d5 = self.Up5(e5)
                    d5 = torch.cat((e4, d5), dim=1)
                    d5 = self.Up_conv5(d5)

                    d4 = self.Up4(d5)
                    d4 = torch.cat((e3, d4), dim=1)
                    d4 = self.Up_conv4(d4)

                    d3 = self.Up3(d4)
                    d3 = torch.cat((e2, d3), dim=1)
                    d3 = self.Up_conv3(d3)

                    d2 = self.Up2(d3)
                    d2 = torch.cat((e1, d2), dim=1)
                    d2 = self.Up_conv2(d2)

                    seg = self.Conv(d2)
                return out, seg
            
            
# class conv_block(nn.Module):
#     """
#     Convolution Block
#     """

#     def __init__(self, in_ch, out_ch):
#         super(conv_block, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True))

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class up_conv(nn.Module):
#     """
#     Up Convolution Block
#     """

#     def __init__(self, in_ch, out_ch):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.up(x)
#         return x


# class U_Net(nn.Module):
#     def __init__(self, in_ch=3, out_ch=3, mode="seg"):
#         super(U_Net, self).__init__()
#         n1 = 64
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

#         self.mode = mode

#         self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(in_ch, filters[0])
#         self.Conv2 = conv_block(filters[0], filters[1])
#         self.Conv3 = conv_block(filters[1], filters[2])
#         self.Conv4 = conv_block(filters[2], filters[3])
#         self.Conv5 = conv_block(filters[3], filters[4])

#         self.Up5 = up_conv(filters[4], filters[3])
#         self.Up_conv5 = conv_block(filters[4], filters[3])

#         self.Up4 = up_conv(filters[3], filters[2])
#         self.Up_conv4 = conv_block(filters[3], filters[2])

#         self.Up3 = up_conv(filters[2], filters[1])
#         self.Up_conv3 = conv_block(filters[2], filters[1])

#         self.Up2 = up_conv(filters[1], filters[0])
#         self.Up_conv2 = conv_block(filters[1], filters[0])

#         self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

#         self.norm = nn.LayerNorm(1024, eps=1e-6)  # final norm layer
#         # self.head = nn.Linear(1024, 2)
#         self.head = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512,128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )
#         self.activation = nn.Sigmoid()

#     def forward(self, x, mode="inference"):
#         e1 = self.Conv1(x)

#         e2 = self.Maxpool1(e1)
#         e2 = self.Conv2(e2)

#         e3 = self.Maxpool2(e2)
#         e3 = self.Conv3(e3)

#         e4 = self.Maxpool3(e3)
#         e4 = self.Conv4(e4)

#         e5 = self.Maxpool4(e4)
#         e5 = self.Conv5(e5)  # out  #(B,1024,32,32)
#         if mode == "seg":
#             d5 = self.Up5(e5)
#             d5 = torch.cat((e4, d5), dim=1)

#             d5 = self.Up_conv5(d5)

#             d4 = self.Up4(d5)
#             d4 = torch.cat((e3, d4), dim=1)
#             d4 = self.Up_conv4(d4)

#             d3 = self.Up3(d4)
#             d3 = torch.cat((e2, d3), dim=1)
#             d3 = self.Up_conv3(d3)

#             d2 = self.Up2(d3)
#             d2 = torch.cat((e1, d2), dim=1)
#             d2 = self.Up_conv2(d2)

#             out = self.Conv(d2)
#         elif mode == "cls":
#             out = self.norm(e5.mean([-2, -1]))
#             out = self.head(out)
#             # out = self.activation(out)
#         elif mode == "inference":
#             out = self.norm(e5.mean([-2, -1]))
#             out = self.head(out)  # (B,2)
#             cls = out.argmax(dim=-1)
#             seg = None
#             if cls:
#                 d5 = self.Up5(e5)
#                 d5 = torch.cat((e4, d5), dim=1)
#                 d5 = self.Up_conv5(d5)

#                 d4 = self.Up4(d5)
#                 d4 = torch.cat((e3, d4), dim=1)
#                 d4 = self.Up_conv4(d4)

#                 d3 = self.Up3(d4)
#                 d3 = torch.cat((e2, d3), dim=1)
#                 d3 = self.Up_conv3(d3)

#                 d2 = self.Up2(d3)
#                 d2 = torch.cat((e1, d2), dim=1)
#                 d2 = self.Up_conv2(d2)

#                 seg = self.Conv(d2)
#             return out, seg
        
#         elif mode == "forward":
#             out = self.norm(e5.mean([-2, -1]))
#             out = self.head(out)  # (B,2)
#             cls = out.argmax(dim=-1)
#             seg = None
#             if True:
#                 d5 = self.Up5(e5)
#                 d5 = torch.cat((e4, d5), dim=1)
#                 d5 = self.Up_conv5(d5)

#                 d4 = self.Up4(d5)
#                 d4 = torch.cat((e3, d4), dim=1)
#                 d4 = self.Up_conv4(d4)

#                 d3 = self.Up3(d4)
#                 d3 = torch.cat((e2, d3), dim=1)
#                 d3 = self.Up_conv3(d3)

#                 d2 = self.Up2(d3)
#                 d2 = torch.cat((e1, d2), dim=1)
#                 d2 = self.Up_conv2(d2)

#                 seg = self.Conv(d2)
#             return out, seg
#         return out
    
    
    
    
    

# class SegResNet(nn.Module):
#     """
#     SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
#     <https://arxiv.org/pdf/1810.11654.pdf>`_.
#     The module does not include the variational autoencoder (VAE).
#     The model supports 2D or 3D inputs.

#     Args:
#         spatial_dims: spatial dimension of the input data. Defaults to 3.
#         init_filters: number of output channels for initial convolution layer. Defaults to 8.
#         in_channels: number of input channels for the network. Defaults to 1.
#         out_channels: number of output channels for the network. Defaults to 2.
#         dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
#         norm_name: feature normalization type, this module only supports group norm,
#             batch norm and instance norm. Defaults to ``group``.
#         num_groups: number of groups to separate the channels into. Defaults to 8.
#         use_conv_final: if add a final convolution block to output. Defaults to ``True``.
#         blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
#         blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
#         upsample_mode: [``"transpose"``, ``"bilinear"``, ``"trilinear"``]
#             The mode of upsampling manipulations.
#             Using the last two modes cannot guarantee the model's reproducibility. Defaults to``trilinear``.

#             - ``transpose``, uses transposed convolution layers.
#             - ``bilinear``, uses bilinear interpolate.
#             - ``trilinear``, uses trilinear interpolate.
#         mode: Mode of the model. Can be "seg" for segmentation, "cls" for classification, or "inference" for combined.
#     """

#     def __init__(
#         self,
#         spatial_dims: int = 2,  # changed default to 2D
#         init_filters: int = 8,
#         in_channels: int = 3,
#         out_channels: int = 3,
#         dropout_prob: Optional[float] = None,
#         norm_name: str = "group",
#         num_groups: int = 8,
#         use_conv_final: bool = True,
#         blocks_down: tuple = (1, 2, 2, 4),
#         blocks_up: tuple = (1, 1, 1),
#         upsample_mode: str = "bilinear",  # changed default to bilinear for 2D
#         mode: str = "seg",  # Added mode parameter
#         num_classes: int = 2,  # For classification head
#     ):
#         super().__init__()

#         assert spatial_dims == 2 or spatial_dims == 3, "spatial_dims can only be 2 or 3."

#         self.spatial_dims = spatial_dims
#         self.init_filters = init_filters
#         self.blocks_down = blocks_down
#         self.blocks_up = blocks_up
#         self.dropout_prob = dropout_prob
#         self.norm_name = norm_name
#         self.num_groups = num_groups
#         self.upsample_mode = upsample_mode
#         self.use_conv_final = use_conv_final
#         self.mode = mode  # Store mode
#         self.convInit = self.get_conv_layer(spatial_dims, in_channels, init_filters)
#         self.down_layers = self._make_down_layers()
#         self.up_layers, self.up_samples = self._make_up_layers()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv_final = self._make_final_conv(out_channels)

#         # Classification head
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1) if spatial_dims == 2 else nn.AdaptiveAvgPool3d(1)
#         self.classifier = nn.Linear(init_filters * 2 ** (len(blocks_down) - 1), num_classes)

#         if dropout_prob:
#             self.dropout = nn.Dropout(dropout_prob)

#     def _make_down_layers(self):
#         down_layers = nn.ModuleList()
#         blocks_down, spatial_dims, filters, norm_name, num_groups = (
#             self.blocks_down,
#             self.spatial_dims,
#             self.init_filters,
#             self.norm_name,
#             self.num_groups,
#         )
#         for i in range(len(blocks_down)):
#             layer_in_channels = filters * 2 ** i
#             pre_conv = (
#                 self.get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
#                 if i > 0
#                 else nn.Identity()
#             )
#             down_layer = nn.Sequential(
#                 pre_conv,
#                 *[
#                     self.ResBlock(spatial_dims, layer_in_channels, norm_name=norm_name, num_groups=num_groups)
#                     for _ in range(blocks_down[i])
#                 ],
#             )
#             down_layers.append(down_layer)
#         return down_layers

#     def _make_up_layers(self):
#         up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
#         upsample_mode, blocks_up, spatial_dims, filters, norm_name, num_groups = (
#             self.upsample_mode,
#             self.blocks_up,
#             self.spatial_dims,
#             self.init_filters,
#             self.norm_name,
#             self.num_groups,
#         )
#         n_up = len(blocks_up)
#         for i in range(n_up):
#             sample_in_channels = filters * 2 ** (n_up - i)
#             up_layers.append(
#                 nn.Sequential(
#                     *[
#                         self.ResBlock(spatial_dims, sample_in_channels // 2, norm_name=norm_name, num_groups=num_groups)
#                         for _ in range(blocks_up[i])
#                     ]
#                 )
#             )
#             up_samples.append(
#                 nn.Sequential(
#                     *[
#                         self.get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
#                         self.get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
#                     ]
#                 )
#             )
#         return up_layers, up_samples

#     def _make_final_conv(self, out_channels: int):
#         return nn.Sequential(
#             self.get_norm_layer(self.spatial_dims, self.init_filters, norm_name=self.norm_name, num_groups=self.num_groups),
#             self.relu,
#             self.get_conv_layer(self.spatial_dims, self.init_filters, out_channels=out_channels, kernel_size=1, bias=True),
#         )

#     def forward(self, x, mode="seg"):
#         x = self.convInit(x)
#         if self.dropout_prob:
#             x = self.dropout(x)

#         down_x = []
#         for i in range(len(self.blocks_down)):
#             x = self.down_layers[i](x)
#             down_x.append(x)
#         down_x.reverse()

#         if mode == "seg" or mode == "inference":
#             for i in range(len(self.blocks_up)):
#                 x = self.up_samples[i](x) + down_x[i + 1]
#                 x = self.up_layers[i](x)

#             if self.use_conv_final:
#                 x = self.conv_final(x)

#         if mode == "cls" or mode == "inference":
#             cls_x = self.global_avg_pool(down_x[0])
#             cls_x = cls_x.view(cls_x.size(0), -1)
#             cls_output = self.classifier(cls_x)
#             if mode == "cls":
#                 return cls_output
#             elif mode == "inference":
#                 return cls_output, x 


#         return x

#     def get_conv_layer(self, spatial_dims, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
#         return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=bias) if spatial_dims == 2 else nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=bias)

#     def get_norm_layer(self, spatial_dims, num_features, norm_name, num_groups):
#         if norm_name == "group":
#             return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
#         elif norm_name == "batch":
#             return nn.BatchNorm2d(num_features) if spatial_dims == 2 else nn.BatchNorm3d(num_features)
#         elif norm_name == "instance":
#             return nn.InstanceNorm2d(num_features) if spatial_dims == 2 else nn.InstanceNorm3d(num_features)
#         else:
#             raise ValueError(f"Unsupported norm type: {norm_name}")

#     def get_upsample_layer(self, spatial_dims, out_channels, upsample_mode="trilinear"):
#         if upsample_mode == "transpose":
#             return nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2) if spatial_dims == 2 else nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
#         elif upsample_mode == "bilinear":
#             return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if spatial_dims == 2 else nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#         elif upsample_mode == "trilinear":
#             return nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#         else:
#             raise ValueError(f"Unsupported upsample mode: {upsample_mode}")

#     class ResBlock(nn.Module):
#         def __init__(self, spatial_dims, in_channels, norm_name="group", num_groups=8):
#             super().__init__()
#             self.conv1 = self.get_conv_layer(spatial_dims, in_channels, in_channels, kernel_size=3)
#             self.norm1 = self.get_norm_layer(spatial_dims, in_channels, norm_name, num_groups)
#             self.relu = nn.ReLU(inplace=True)
#             self.conv2 = self.get_conv_layer(spatial_dims, in_channels, in_channels, kernel_size=3)
#             self.norm2 = self.get_norm_layer(spatial_dims, in_channels, norm_name, num_groups)

#         def forward(self, x):
#             residual = x
#             out = self.conv1(x)
#             out = self.norm1(out)
#             out = self.relu(out)
#             out = self.conv2(out)
#             out = self.norm2(out)
#             out += residual
#             out = self.relu(out)
#             return out

#     # Depending on the provided dimensions, this will determine if it's 2D or 3D
#         def get_conv_layer(self, spatial_dims, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
#             if spatial_dims == 2:
#                 return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=bias)
#             elif spatial_dims == 3:
#                 return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=bias)

#         def get_norm_layer(self, spatial_dims, num_features, norm_name, num_groups):
#             if norm_name == "group":
#                 return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
#             elif norm_name == "batch":
#                 if spatial_dims == 2:
#                     return nn.BatchNorm2d(num_features)
#                 elif spatial_dims == 3:
#                     return nn.BatchNorm3d(num_features)
#             elif norm_name == "instance":
#                 if spatial_dims == 2:
#                     return nn.InstanceNorm2d(num_features)
#                 elif spatial_dims == 3:
#                     return nn.InstanceNorm3d(num_features)
#             else:
#                 raise ValueError(f"Unsupported norm type: {norm_name}")


    
    
    
    


# if __name__ == '__main__':
#     breakpoint()
#     x = torch.randn((12, 3, 512, 512)).cpu()
#     model = SegResNet().cpu()
#     # model.load_state_dict(torch.load("./checkpoints/epoch65_val_acc_0.85322.pth",map_location="cpu"))
#     out = model(x, "seg")
#     breakpoint()
#     # # print(out.shape)
#     # import pickle

#     # with open('model.pickle', 'wb') as f:
#     #     pickle.dump(model, f)
