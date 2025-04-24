from networks.unet_3D import unet_3D, unet_3D_scale4
from networks.vnet import VNet
from networks.VoxResNet import VoxResNet
from networks.nnunet import initialize_network
from networks.nnunetljy import UNet_LY


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2, pretrain=False, reconstruct=False, mix_cof=False, mix_num=False):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "unet_3D_scale4":
        net = unet_3D_scale4(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "voxresnet":
        net = VoxResNet(in_chns=in_chns, feature_chns=64,
                        class_num=class_num).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    elif net_type == "nnUNet_LY":
        net = UNet_LY(spatial_dims=3, in_channels=in_chns, out_channels=class_num, strides=(2,2,2,2), pretrain=pretrain, reconstruct=reconstruct, mix_cof=mix_cof, mix_num=mix_num).cuda()
    else:
        net = None
    return net
