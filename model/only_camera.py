import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################
############CAMERA ARCHITECTURE###############
##############################################

class Bottleneck_camera(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None, expansion=4):
        super(Bottleneck_camera, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion * planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(residual + out)
        return out


class warmupblock(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size=1, use_bn=True):
        super(warmupblock, self).__init__()
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(in_layer, out_layer, kernel_size,
                               stride=(1, 1), padding=1, bias=(not use_bn))

        self.bn1 = nn.BatchNorm2d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        if self.use_bn:
            x1 = self.bn1(x1)
        x = self.relu(x1)
        return x


class FPN_BackBone_camera(nn.Module):

    def __init__(self, num_block, channels, block_expansion, use_bn=True):
        super(FPN_BackBone_camera, self).__init__()
        self.block_expansion = block_expansion
        self.use_bn = use_bn
        self.warmup = warmupblock(3, 32, kernel_size=3, use_bn=True)
        self.in_planes = 32

        self.conv = nn.Conv2d(self.in_planes, self.in_planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=False)

        # Residuall blocks
        self.block1 = self._make_layer(Bottleneck_camera, planes=channels[0], num_blocks=num_block[0])
        self.block2 = self._make_layer(Bottleneck_camera, planes=channels[1], num_blocks=num_block[1])
        self.block3 = self._make_layer(Bottleneck_camera, planes=channels[2], num_blocks=num_block[2])
        self.block4 = self._make_layer(Bottleneck_camera, planes=channels[3], num_blocks=num_block[3])

    def forward(self, x):
        x = self.warmup(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Backbone
        features = {}
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        features['x0'] = x
        features['x1'] = x1
        features['x2'] = x2
        features['x3'] = x3
        features['x4'] = x4

        return features

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * self.block_expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * self.block_expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * self.block_expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample, expansion=self.block_expansion))
        self.in_planes = planes * self.block_expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1, expansion=self.block_expansion))
            self.in_planes = planes * self.block_expansion
        return nn.Sequential(*layers)  # this *layers will unpack the list


class BasicBlock_UpScaling(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_UpScaling, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class UpScaling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = BasicBlock_UpScaling(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1) #.clone()
        return self.conv(x)


class BEV_decoder(nn.Module):
    def __init__(self, ):
        super(BEV_decoder, self).__init__()
        ##with low params
        # self.up1 = (Up(256, 128))
        # self.up2 = (Up(128, 64))
        # self.up3 = (Up(64, 32))
        # self.up4 = (Up(32, 16))
        self.up1 = (UpScaling(512, 256))
        self.up2 = (UpScaling(256, 128))
        self.up3 = (UpScaling(128, 64))
        self.up4 = (UpScaling(64, 32))

    def forward(self, features):
        T0 = features['x0']
        T1 = features['x1']
        T2 = features['x2']
        T3 = features['x3']
        T4 = features['x4']

        x = self.up1(T4, T3)
        x = self.up2(x, T2)
        x = self.up3(x, T1)
        out = self.up4(x, T0)
        width = out.shape[-1]
        out2 = out #.detach().clone()
        out = out2[:, :, 0:128, 16:int(width - 16)]
        return out


class PolarSegFusionNet_BEV(nn.Module):
    def __init__(self, channels_bev, blocks):

        super(PolarSegFusionNet_BEV, self).__init__()

        self.FPN = FPN_BackBone_camera(num_block=blocks, channels=channels_bev, block_expansion=4, use_bn=True)
        self.BEV_decoder = BEV_decoder()


    def forward(self, x):
        fusion_list_camera = []
        features = self.FPN(x)
        BEV_decoded = self.BEV_decoder(features)
        fusion_list_camera.append(BEV_decoded)
        return fusion_list_camera


class SegmentationHead_onlycam(nn.Module):
    def __init__(self, channels_bev, blocks, segmentation_head, camera_input):
        super(SegmentationHead_onlycam, self).__init__()

        self.segmentation_head = segmentation_head
        self.camera_input = camera_input

        #####################################################
        ########### SEGMENTATION WITH CAMERA INPUT ##########
        #####################################################
        if self.camera_input and self.segmentation_head:
            self.cameraenc = PolarSegFusionNet_BEV(channels_bev=channels_bev, blocks=blocks)
            self.freespace_camera = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1))

    def forward(self, bev_inputs):

        out = {'Segmentation': []}
        out_seg = {'Segmentation_Radar': [], 'Segmentation_Camera': []}

        #####################################################
        ########### SEGMENTATION WITH CAMERA INPUT ##########
        #####################################################
        if self.camera_input and self.segmentation_head:
            fusion_list_camera = self.cameraenc(bev_inputs)
            C = F.interpolate(fusion_list_camera[0], (256, 224))
            out_seg['Segmentation_Camera'] = self.freespace_camera(C)
            out['Segmentation'] = out_seg['Segmentation_Camera']

        return out
