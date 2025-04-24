import sys
sys.path.append('/mnt/liangjw/ASA/ASA_Segmentation')
sys.path.append('/mnt/liangjw/PCRLv2/models')
sys.path.append('/mnt/liangjw/TransVW/pytorch')
# from network_architecture.MEDIUMVIT import MEDIUMVIT
# from training.network_training.nnFormerTrainer import nnFormerTrainer
# from models.ynet3d import UNet3D #TransVW
from collections import OrderedDict
import math
from glob import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
from networks.net_factory_3d import net_factory_3d
from torch import nn
import SimpleITK as sitk
from monai.networks.nets import SwinUNETR
# from pcrlv2_model_3d import SegmentationModel
from ModelsGenesis.pytorch.unet3d import UNet3D



def ConvertToMultiChannelBasedOnBratsClassesd(label):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """
    
    result = []
    # merge label 4 and label 1 to construct TC
    result.append(np.where(((label >= 3) | (label == 1)), 1, 0))
    # merge labels 2, 4 and 1 to construct WT
    result.append(np.where(label>0, 1, 0))
    # label 4 is ET
    result.append(np.where(label>=3, 1, 0))
    label = np.stack(result, axis=0)
    return label

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, if_sigmoid=True):
    c, w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(0,0), (wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    cc, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + (ww, hh, dd)).astype(np.float32)
    cnt = np.zeros((ww, hh, dd)).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # ensemble
                    if if_sigmoid:
                        logit = torch.nn.Sigmoid()
                        y = logit(y1)
                    else:
                        y = y1
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.where(score_map>=0.5, 1, 0)

    if add_pad:
        label_map = label_map[:, wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return score_map,label_map



def cal_metric(gt, pred, dsc_smmoth):
    if dsc_smmoth==0:
        if pred.sum() > 0 and gt.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            return np.array([dice, hd95])
        elif pred.sum() == 0 and gt.sum() == 0:
            return np.array([1, 0])
        else:
            return np.array([0, 10])
    else:
        if pred.sum() > 0 and gt.sum() > 20:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            return np.array([dice, hd95])
        elif pred.sum() == 0 and gt.sum() == 0:
            return np.array([1, 0])
        else:
            dice = metric.binary.dc(pred, gt, dsc_smmoth)
            return np.array([dice, 10])





class Dice(nn.Module):
    def __init__(self, n_classes):
        super(Dice, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore_mask):
        target = target.float()
        smooth = 1e-5
        if ignore_mask is not None:
            ignore_mask = ignore_mask.float()
            intersect = torch.sum(score * target * ignore_mask)
            y_sum = torch.sum(target)
            z_sum = torch.sum(score)
        else:
            intersect = torch.sum(score * target)
            y_sum = torch.sum(target)
            z_sum = torch.sum(score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False, ignore_mask=None):
        inputs = torch.tensor(inputs)
        target = torch.tensor(target)
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[i,], target[i,], ignore_mask)
            class_wise_dice.append(1.0 - dice.item())
            
        return  class_wise_dice

# def test_all_case(net, base_dir, test_list="full_test.list", num_classes=3, patch_size=(64, 160, 160), stride_xy=32, stride_z=24, select=['t1','t1ce','t2','flair'], val_num=None,if_sigmoid=False):
#     ref_img = sitk.ReadImage('/mnt/liangjw/SSL4MOD/data/BraTS2021/BraTS2021_00000/BraTS2021_00000_t1_MNI.nii.gz')
#     ori = ref_img.GetOrigin()
#     dir = ref_img.GetDirection()
#     spa = ref_img.GetSpacing()
#     with open(os.path.join(base_dir,test_list), 'r') as f:
#         image_list = f.readlines()
#     if val_num is not None:
#         image_list = image_list[:val_num]
#     image_list = [base_dir + "/{}/{}h5file.h5".format(
#         item.replace('\n', '').split(",")[0], item.replace('\n', '').split(",")[0]) for item in image_list]
#     # total_metric = np.zeros((num_classes, 2))
#     # print("Validation begin")
#     # dsc_M = Dice(3)
#     # tc_dsc = []
#     # wt_dsc = []
#     # et_dsc = []
#     # tc_hd = []
#     # wt_hd = []
#     # et_hd = []
#     for image_path in tqdm(image_list):

#         snap_path = os.path.join('/mnt/liangjw/SSL4MOD/view_MG_result_SSA',image_path.split('/')[6])
#         if not os.path.exists(snap_path):
#             os.mkdir(snap_path)

#         # dsc_metric = []
#         # hd95_metric = []
#         h5f = h5py.File(image_path, 'r')
#         image = []
#         for i in select:
#             if i is not None:
#                 image.append(h5f[i][:])
#                 image_shape = h5f[i][:].shape
#             else:
#                 image.append('None')
#         for i in range(len(image)):
#             if isinstance(image[i],str):
#                 image[i] = np.zeros(shape=image_shape)
                
#         image = np.stack(image,axis=0)
#         # label = h5f['seg'][:]
#         # label = ConvertToMultiChannelBasedOnBratsClassesd(label=label)
#         predict_prob, prediction_label = test_single_case(
#             net, image, stride_xy, stride_z, patch_size, num_classes=num_classes,if_sigmoid=if_sigmoid)
#         # dsc_metric = dsc_M(prediction_label,label)
#         # total_metric[:, 0] += dsc_metric
#         t1ce = sitk.GetImageFromArray(np.float32(image[1,]))
#         t1ce.SetDirection(dir)
#         t1ce.SetOrigin(ori)
#         t1ce.SetSpacing(spa)
#         sitk.WriteImage(t1ce,fileName=os.path.join(snap_path,'t1ce.nii'))
#         # sitk.WriteImage(image=sitk.GetImageFromArray(np.int32(label[0,])),fileName=os.path.join(snap_path,'tc_label.nii'))
#         # sitk.WriteImage(image=sitk.GetImageFromArray(np.int32(label[1,])),fileName=os.path.join(snap_path,'wt_label.nii'))
#         # sitk.WriteImage(image=sitk.GetImageFromArray(np.int32(label[2,])),fileName=os.path.join(snap_path,'et_label.nii'))
#         tc_predict = sitk.GetImageFromArray(np.int32(prediction_label[0,]))
#         tc_predict.SetDirection(dir)
#         tc_predict.SetOrigin(ori)
#         tc_predict.SetSpacing(spa)
#         sitk.WriteImage(tc_predict,fileName=os.path.join(snap_path,'tc_predict.nii'))
#         wt_predict = sitk.GetImageFromArray(np.int32(prediction_label[1,]))
#         wt_predict.SetDirection(dir)
#         wt_predict.SetOrigin(ori)
#         wt_predict.SetSpacing(spa)
#         sitk.WriteImage(wt_predict,fileName=os.path.join(snap_path,'wt_predict.nii'))
#         et_predict = sitk.GetImageFromArray(np.int32(prediction_label[2,]))
#         et_predict.SetDirection(dir)
#         et_predict.SetOrigin(ori)
#         et_predict.SetSpacing(spa)
#         sitk.WriteImage(et_predict,fileName=os.path.join(snap_path,'et_predict.nii'))
#         # sitk.WriteImage(image=sitk.GetImageFromArray(np.int32(prediction_label)), fileName=os.path.join('/mnt/liangjw/SSL4MOD/MICCAI_2024_Ours_Exval/upload_folder',image_path.split('/')[7].split('_')[1]+'.nii'))
#         # np.save(os.path.join('/mnt/liangjw/SSL4MOD/MICCAI_2024_Ours_Exval/npz_fold',image_path.split('/')[7].split('_')[1]+'.npy'),np.int32(prediction_label))

#     #     for i in range(num_classes):
#     #         metric = cal_metric(label[i,:,:,:] == 1, prediction_label[i,] == 1)
#     #         dsc = metric[0]
#     #         hd95 = metric[1]

#     #         if i == 0:
#     #             tc_dsc.append(dsc)
#     #             tc_hd.append(hd95)
#     #         if i == 1:
#     #             wt_dsc.append(dsc)
#     #             wt_hd.append(hd95)
#     #         if i == 2:
#     #             et_dsc.append(dsc)
#     #             et_hd.append(hd95)
            
#     #         hd95_metric.append(hd95)
#     #         total_metric[i, 1] += hd95
#     #         dsc_metric.append(dsc)
#     #         total_metric[i, 0] += dsc
#     #     print('sample %s :\n dice_score : %f dice_tc : %f dice_wt : %f dice_et : %f  hd95 : %f hd95_tc : %f hd95_wt : %f hd95_tc : %f' % (image_path, np.mean(np.array(dsc_metric)), dsc_metric[0], dsc_metric[1], dsc_metric[2], np.mean(np.array(hd95_metric)), hd95_metric[0], hd95_metric[1], hd95_metric[2]))
#     # tc_dsc = np.array(tc_dsc)
#     # wt_dsc = np.array(wt_dsc)
#     # et_dsc = np.array(et_dsc)
#     # tc_hd = np.array(tc_hd)
#     # wt_hd = np.array(wt_hd)
#     # et_hd = np.array(et_hd)
#     # total_metric = total_metric / len(image_list)
#     # print('sample %s :\n dice_score : %f dice_tc : %f dice_wt : %f dice_et : %f  hd95 : %f hd95_tc : %f hd95_wt : %f hd95_tc : %f dice_std : %f dice_tc_std : %f dice_wt_std : %f dice_et_std : %f  hd95_std : %f hd95_tc_std : %f hd95_wt_std : %f hd95_tc_std : %f' % (image_path, total_metric[:, 0].mean(), total_metric[0, 0], total_metric[1, 0], total_metric[2, 0], total_metric[:, 1].mean(), total_metric[0, 1], total_metric[1, 1], total_metric[2, 1],
#     #                                                                                                                                                                                                                                                                     np.std((tc_dsc+wt_dsc+et_dsc)/3), np.std(tc_dsc), np.std(wt_dsc), np.std(et_dsc), np.std((tc_hd+wt_hd+et_hd)/3), np.std(tc_hd), np.std(wt_hd), np.std(et_hd)))
#     print("Validation end")
    

def test_all_case(net, base_dir, test_list="full_test.list", num_classes=3, patch_size=(64, 160, 160), stride_xy=32, stride_z=24, select=['t1','t1ce','t2','flair'], val_num=None, if_sigmoid=True):
    
    with open(os.path.join(base_dir,test_list), 'r') as f:
        image_list = f.readlines()
    if val_num is not None:
        image_list = image_list[:val_num]
    image_list = [base_dir + "/{}/{}h5file.h5".format(
        item.replace('\n', '').split(",")[0], item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes, 2))
    print("Validation begin")
    
    tc_dsc = []
    wt_dsc = []
    et_dsc = []
    tc_hd = []
    wt_hd = []
    et_hd = []
    num=0
    for image_path in tqdm(image_list):
        num += 1

        # snap_path = os.path.join('/mnt/liangjw/SSL4MOD/MICCAI_2024_Ours_Exval',image_path.split('/')[7])
        # if not os.path.exists(snap_path):
        #     os.mkdir(snap_path)

        dsc_metric = []
        hd95_metric = []
        h5f = h5py.File(image_path, 'r')
        image = []
        for i in select:
            if i is not None:
                image.append(h5f[i][:])
                image_shape = h5f[i][:].shape
            else:
                image.append('None')
        for i in range(len(image)):
            if isinstance(image[i],str):
                image[i] = np.zeros(shape=image_shape)
                
        image = np.stack(image,axis=0)
        label = h5f['seg'][:]
        label = ConvertToMultiChannelBasedOnBratsClassesd(label=label)
        predict_prob, prediction_label = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, if_sigmoid=if_sigmoid)
        # dsc_metric = dsc_M(prediction_label,label)
        # total_metric[:, 0] += dsc_metric

        for i in range(num_classes):
            if i<2:
                metric = cal_metric(label[i,:,:,:] == 1, prediction_label[i,] == 1, dsc_smmoth=0)
            else:
                metric = cal_metric(label[i,:,:,:] == 1, prediction_label[i,] == 1, dsc_smmoth=0)
            dsc = metric[0]
            hd95 = metric[1]

            if i == 0:
                tc_dsc.append(dsc)
                tc_hd.append(hd95)
            if i == 1:
                wt_dsc.append(dsc)
                wt_hd.append(hd95)
            if i == 2:
                et_dsc.append(dsc)
                et_hd.append(hd95)
            
            hd95_metric.append(hd95)
            total_metric[i, 1] += hd95
            dsc_metric.append(dsc)
            total_metric[i, 0] += dsc
        print('sample %s :\n ACC_dice : %f dice_score : %f dice_tc : %f dice_wt : %f dice_et : %f  hd95 : %f hd95_tc : %f hd95_wt : %f hd95_tc : %f' % (image_path, (total_metric/num)[:, 0].mean(), np.mean(np.array(dsc_metric)), dsc_metric[0], dsc_metric[1], dsc_metric[2], np.mean(np.array(hd95_metric)), hd95_metric[0], hd95_metric[1], hd95_metric[2]))
    tc_dsc = np.array(tc_dsc)
    wt_dsc = np.array(wt_dsc)
    et_dsc = np.array(et_dsc)
    tc_hd = np.array(tc_hd)
    wt_hd = np.array(wt_hd)
    et_hd = np.array(et_hd)
    total_metric = total_metric / len(image_list)
    print('sample %s :\n dice_score : %f dice_tc : %f dice_wt : %f dice_et : %f  hd95 : %f hd95_tc : %f hd95_wt : %f hd95_tc : %f dice_std : %f dice_tc_std : %f dice_wt_std : %f dice_et_std : %f  hd95_std : %f hd95_tc_std : %f hd95_wt_std : %f hd95_tc_std : %f' % (image_path, total_metric[:, 0].mean(), total_metric[0, 0], total_metric[1, 0], total_metric[2, 0], total_metric[:, 1].mean(), total_metric[0, 1], total_metric[1, 1], total_metric[2, 1],
                                                                                                                                                                                                                                                                        np.std((tc_dsc+wt_dsc+et_dsc)/3), np.std(tc_dsc), np.std(wt_dsc), np.std(et_dsc), np.std((tc_hd+wt_hd+et_hd)/3), np.std(tc_hd), np.std(wt_hd), np.std(et_hd)))
    print("Validation end")
 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = net_factory_3d(net_type='nnUNet_LY', in_chns=4, class_num=3).to(device)
# model = SwinUNETR(img_size=(128,128,128),
#                   in_channels=4,
#                   out_channels=3,
#                   feature_size=48,
#                   use_checkpoint=True,
#                   ).to(device)
# model = SegmentationModel(n_class=3,in_channels=4,norm='gn')#pcrlv2
# model = UNet3D(in_channel=4, n_class=3).to(device) #TransVW
# model = UNet3D(in_channel=4, n_class=3).to(device)#MG
model = torch.nn.DataParallel(model, device_ids=[0])

load_model = torch.load('/mnt/liangjw/SSL4MOD/finetune_balance_Mix_Mask_ReconstructOnly_4mod_210/BraTs2021_Finetune/nnUNet_LY/iter_41999_goodtake_0.883.pth')
model.load_state_dict(load_model['model'])

# model = MEDIUMVIT(in_channels=4,out_channels=3,img_size=(128,128,128),norm_name='instance',window_size=32).cuda()
# model = torch.nn.DataParallel(model, device_ids=[0])
# checkpoint = torch.load('/mnt/liangjw/SSL4MOD/finetune_ASA_META_4mod_142/BraTs2023_META/nnUNet_LY/iter_63899.pth',
#                                 map_location='cpu')
# checkpoint_model = checkpoint['model']
# model.load_state_dict(checkpoint_model)

model.eval()
# test_all_case(net=model,base_dir='/mnt/liangjw/SwinMM/BraTS2021_01scale',test_list='test_path_list.txt',patch_size=(128, 128, 128), stride_xy=64, stride_z=48,select=['t1','t1ce','t2','flair'],if_sigmoid=True)#need
test_all_case(net=model,base_dir='/mnt/liangjw/SSL4MOD/data/BraTS2021',test_list='test_path_list.txt', patch_size=(128, 128, 128),stride_xy=64, stride_z=48, select=['t1','t1ce','t2','flair'],if_sigmoid=True)#need
