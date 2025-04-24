import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import random
import shutil
import sys
import time
import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.brats2019 import (BraTS2019, BraTS2021, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor, ConvertToMultiChannelBasedOnBratsClassesd,
                                   TwoStreamBatchSampler, RandPatchMix, RandPatchPuzzle, Rand_Mask_Mix)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case
import monai.transforms as monai_transforms
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt
import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/mnt/liangjw/SSL4MOD/data/BraTS2021', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BraTs2021_Pre_Train', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='nnUNet_LY', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=800*450, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--sw_batch', type=int, default=2,
                    help='batch_size per sample')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=3e-4,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[128, 128, 128],
                    help='patch size of network input')
parser.add_argument('--split_patch_size', type=list,  default=[16, 16, 16],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=900,
                    help='labeled data')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--resume_path', default='/mnt/liangjw/SSL4MOD/pretrain_balance_model_nnunet_Reconstruct_Coeff_NumCE_b2:4/BraTs2021_Pre_Train/nnUNet_LY/iter_157049_loss_0.159.pth', type=str, help='retrain path')

args = parser.parse_args()


def PL2Loss(input_tensor, output_tensor, brain_tensor):
    L2_calculator = nn.MSELoss()
    l2 = 0
    b,c,w,h,d = input_tensor.shape
    
    for i in range(b):   
        brain = brain_tensor[i,].repeat(1,4,1,1).reshape(16,128,128,128) #need
        l2 = l2 + L2_calculator(input_tensor[i,][torch.where(brain==1)], output_tensor[i,][torch.where(brain==1)])
    l2 = l2/b
    return l2
        
def train(args, snapshot_path):
    original_mri = sitk.ReadImage('/mnt/liangjw/SSL4MOD/data/BraTS2021/BraTS2021_00000/BraTS2021_00000_t1_MNI.nii.gz')
    m_ori = original_mri.GetOrigin()
    m_spa = original_mri.GetSpacing()
    m_dir = original_mri.GetDirection()
    s_ori = original_mri.GetOrigin()
    s_spa = original_mri.GetSpacing()
    s_dir = original_mri.GetDirection()
    iter_num = 0
    start_epoch = 0
    loss_epoch = []
    loss_reconstruct_epoch = []
    loss_mix_cof_epoch = []
    loss_mixt1_Ce_epoch = []
    loss_mixt1ce_Ce_epoch = []
    loss_mixt2_Ce_epoch = []
    loss_mixflair_Ce_epoch = []
    # best_loss = 1e8

    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 3
    coefficient=np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],dtype=np.float32)
    coefficient_sample0 = [[c1,c2,c3,c4] for c1 in coefficient for c2 in coefficient for c3 in coefficient for c4 in coefficient]
    coefficient_sample = [x for x in coefficient_sample0 if (x[0]+x[1]+x[2]+x[3]==1)&(x[0]+x[1]+x[2]<1)]
    coefficient_balance = {}
    coefficient_balance['m1'] = [[0,0,0,1]]
    coefficient_balance['m2'] = []
    coefficient_balance['m3'] = []
    coefficient_balance['m4'] = []
    for c in coefficient_sample:
        cc = 0
        for i in c:
            if i>=0.05:
                cc += 1
        if cc==2:
            coefficient_balance['m2'].append(c)
        if cc==3:
            coefficient_balance['m3'].append(c)
        if cc==4:
            coefficient_balance['m4'].append(c)

    patches_set = [[i,j,k] for i in np.arange(0, args.patch_size[0], args.split_patch_size[0]) for j in np.arange(0, args.patch_size[1], args.split_patch_size[1]) for k in np.arange(0, args.patch_size[2], args.split_patch_size[2])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = net_factory_3d(net_type=args.model, in_chns=4, class_num=num_classes, pretrain=True, reconstruct=True, mix_cof=True, mix_num=True).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0,1])

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(trainable_params,
                                      lr=base_lr)
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight)
    transform1 = transforms.Compose([
                             ConvertToMultiChannelBasedOnBratsClassesd(),
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ])
    transform2 = monai_transforms.Compose(
        [   monai_transforms.RandZoomd(
                keys=["image","brain"],
                prob=0.8,
                max_zoom=1.4,
                min_zoom=0.7,
                mode=("bilinear","nearest"),
            ),
            monai_transforms.RandSpatialCropSamplesd(
                keys=["image","brain"],
                roi_size=(args.patch_size[0],args.patch_size[1],args.patch_size[2]),
                num_samples=args.sw_batch,
                random_size = False
            ),
            monai_transforms.RandRotated(
                range_x=0.5236,
                range_y=0.5236,
                range_z=0.5236,
                keys=["image","brain"],
                mode=("bilinear","nearest"),
                prob=0.8
            ),
            Rand_Mask_Mix(coefficient_sample=coefficient_balance, patches_set=patches_set, input_sizes=args.patch_size, patch_size=args.split_patch_size, mix_num_flag=True),
            monai_transforms.ToTensord(keys=["image","mix_image"], track_meta=False),
        ]
    )
    db_train = BraTS2021(base_dir=train_data_path,
                         split='train_pretrain',
                         num=args.labeled_num,
                         select=['t1','t1ce','t2','flair'],
                         transform=transform2,
                         need_brain_mask=True)

    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    
    if args.resume_path is not None:
        iter_num = int(args.resume_path.split('_')[11])+1
        load_dict = torch.load(args.resume_path)
        start_epoch = load_dict['epoch']
        model.load_state_dict(load_dict['model'])
        optimizer.load_state_dict(load_dict['optimizer'])
        loss_epoch = load_dict['loss']
        loss_reconstruct_epoch = load_dict['loss_reconstruct']
        loss_mix_cof_epoch = load_dict['loss_mix_coff']
        loss_mixt1_Ce_epoch = load_dict['mixt1_ce']
        loss_mixt1ce_Ce_epoch = load_dict['mixt1ce_ce']
        loss_mixt2_Ce_epoch = load_dict['mixt2_ce']
        loss_mixflair_Ce_epoch = load_dict['mixflair_ce']

    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=800)
    elif args.lrschedule == 'poly':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.max_epochs, power=0.9)
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.max_epochs)
    else:
        scheduler = None

    if (args.resume_path is not None)&(scheduler is not None):
            scheduler.step(epoch=start_epoch)
    

    model.train()

    
    reconstruct_loss = nn.MSELoss()
    
    Ce_loss_t1 = nn.CrossEntropyLoss(ignore_index=-1)
    Ce_loss_t1ce = nn.CrossEntropyLoss(ignore_index=-1)
    Ce_loss_t2 = nn.CrossEntropyLoss(ignore_index=-1)
    Ce_loss_flair = nn.CrossEntropyLoss(ignore_index=-1)

    
    logging.info("{} iterations per epoch".format(len(trainloader)))
    


    max_epoch = max_iterations // len(trainloader) + 1
    
    iterator = tqdm(range(start_epoch+1,max_epoch), ncols=70)
    
    for epoch_num in iterator:
        loss_per = 0
        loss_reconstruct_per = 0
        loss_mix_cof_per = 0
        loss_mixt1_Ce_per = 0
        loss_mixt1ce_Ce_per = 0
        loss_mixt2_Ce_per = 0
        loss_mixflair_Ce_per = 0

        for i_batch, sampled_batch in enumerate(trainloader):

            
            if isinstance(sampled_batch, list):
                mix_volume_batch = [x['mix_image'][j,] for x in sampled_batch for j in range(args.batch_size)]
                volume_batch = [x['image'][j,] for x in  sampled_batch for j in range(args.batch_size)]
                mix_cof_batch = [x['mix_cof'][j,] for x in  sampled_batch for j in range(args.batch_size)]
                mix_num_batch = [x['mix_num'][j,] for x in  sampled_batch for j in range(args.batch_size)]
                mix_num_label_batch = [x['mix_num_label'][j,] for x in sampled_batch for j in range(args.batch_size)]
                brain_batch = [x['brain'][j,].squeeze() for x in sampled_batch for j in range(args.batch_size)]

                mix_volume_batch = torch.stack(mix_volume_batch, dim=0).float()
                volume_batch = torch.stack(volume_batch, dim=0).float()
                mix_cof_batch = torch.stack(mix_cof_batch, dim=0)
                mix_num_batch = torch.stack(mix_num_batch, dim=0)
                mix_num_label_batch = torch.stack(mix_num_label_batch, dim=0)
                brain_batch = torch.stack(brain_batch, dim=0)
            else:
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                mix_num_label_batch = torch.stack(mix_num_label_batch, dim=0)

            
                

            volume_batch, mix_volume_batch, mix_cof_batch, mix_num_batch, mix_num_label_batch, brain_batch= volume_batch.cuda(), mix_volume_batch.cuda(), mix_cof_batch.cuda(), mix_num_batch.cuda(), mix_num_label_batch.cuda(), brain_batch.cuda()

            
            outputs_reconstruct, outputs_mix_cof, outputs_mix_num = model(mix_volume_batch)

            loss_ce = 0
            loss_reconstruct = 0
            loss_mix_cof = 0

            if outputs_reconstruct is not None:
                loss_reconstruct = reconstruct_loss(outputs_reconstruct, volume_batch)
                loss_reconstruct_per += loss_reconstruct.item()
            else:
                loss_reconstruct = 0

            if outputs_mix_cof is not None:
                loss_mix_cof = PL2Loss(mix_cof_batch.float(), outputs_mix_cof, brain_batch.float())
                loss_mix_cof_per += loss_mix_cof.item()
            else:
                loss_mix_cof = 0

            if outputs_mix_num is not None:
                loss_mix_num_ce_t1 = Ce_loss_t1(outputs_mix_num[:,0:4,:,:,:], mix_num_batch[:,0,:,:,:].long())
                loss_mix_num_ce_t1ce = Ce_loss_t1ce(outputs_mix_num[:,4:8,:,:,:], mix_num_batch[:,1,:,:,:].long())
                loss_mix_num_ce_t2 = Ce_loss_t2(outputs_mix_num[:,8:12,:,:,:], mix_num_batch[:,2,:,:,:].long())
                loss_mix_num_ce_flair = Ce_loss_flair(outputs_mix_num[:,12:16,:,:,:], mix_num_batch[:,3,:,:,:].long())

                loss_mixt1_Ce_per += loss_mix_num_ce_t1.item()
                loss_mixt1ce_Ce_per += loss_mix_num_ce_t1ce.item()
                loss_mixt2_Ce_per += loss_mix_num_ce_t2.item()
                loss_mixflair_Ce_per += loss_mix_num_ce_flair.item()

                loss_ce = (loss_mix_num_ce_t1 + loss_mix_num_ce_t1ce + loss_mix_num_ce_t2 + loss_mix_num_ce_flair)/4
                
            else:
                loss_mix_num_ce_t1 = 0
                loss_mix_num_ce_t1ce = 0
                loss_mix_num_ce_t2 = 0
                loss_mix_num_ce_flair = 0
                loss_ce = 0
            

            
            loss =  loss_ce + loss_reconstruct + loss_mix_cof
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_per += loss.item()

            logging.info(
                'iteration %d : loss : %f, loss_reconstruct : %f, loss_mix_cof : %f, loss_mixt1_ce: %f, loss_mixt1ce_ce: %f, loss_mixt2_ce: %f, loss_mixflair_ce: %f' %
                (iter_num, loss.item(), loss_reconstruct.item(), loss_mix_cof.item(), 
                 loss_mix_num_ce_t1.item(), loss_mix_num_ce_t1ce.item(), loss_mix_num_ce_t2.item(), loss_mix_num_ce_flair.item()))
           
            
            iter_num = iter_num + 1

        loss_per /= len(trainloader)
        loss_reconstruct_per /= len(trainloader)
        loss_mix_cof_per /= len(trainloader)
        loss_mixt1_Ce_per /= len(trainloader)
        loss_mixt1ce_Ce_per /= len(trainloader)
        loss_mixt2_Ce_per /= len(trainloader)
        loss_mixflair_Ce_per /= len(trainloader)

        loss_epoch.append(loss_per)
        loss_reconstruct_epoch.append(loss_reconstruct_per)
        loss_mix_cof_epoch.append(loss_mix_cof_per)
        loss_mixt1_Ce_epoch.append(loss_mixt1_Ce_per)
        loss_mixt1ce_Ce_epoch.append(loss_mixt1ce_Ce_per)
        loss_mixt2_Ce_epoch.append(loss_mixt2_Ce_per)
        loss_mixflair_Ce_epoch.append(loss_mixflair_Ce_per)

        logging.info(
                'epoch %d : loss : %f, loss_reconstruct : %f, loss_mix_cof : %f, loss_mixt1_ce: %f, loss_mixt1ce_ce: %f, loss_mixt2_ce: %f, loss_mixflair_ce: %f' %
                (epoch_num, loss_per, loss_reconstruct_per, loss_mix_cof_per,
                 loss_mixt1_Ce_per, loss_mixt1ce_Ce_per, loss_mixt2_Ce_per, loss_mixflair_Ce_per))
        
        scheduler.step()
        
                    
        if epoch_num > 0 and (epoch_num + 1) % 50 == 0:
            save_mode_path = os.path.join(snapshot_path,
                                                'iter_{}_loss_{}.pth'.format(
                                                    iter_num-1, round(loss_per, 3)))
                    
            torch.save({"model":model.state_dict(), "optimizer":optimizer.state_dict(), "epoch":epoch_num,
                        "loss":loss_epoch,"loss_reconstruct":loss_reconstruct_epoch,"loss_mix_coff":loss_mix_cof_epoch,
                        "mixt1_ce":loss_mixt1_Ce_epoch,"mixt1ce_ce":loss_mixt1ce_Ce_epoch,"mixt2_ce":loss_mixt2_Ce_epoch,"mixflair_ce":loss_mixflair_Ce_epoch}, save_mode_path)
        
        if iter_num >= max_iterations:
            break


        if iter_num >= max_iterations:
            iterator.close()
            break
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/mnt/liangjw/SSL4MOD/pretrain_balance_model_nnunet_Reconstruct_Coeff_NumCE_b2:4/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
