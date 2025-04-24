import sys
sys.path.append('/raid5/liangjw/SSL4MIS/code/utils')
import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from random import sample
import random
from skimage import exposure
import SimpleITK as sitk




def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For every value v in `labels`, the value in the output will be either 1 or 0. Each vector along the `dim`-th
    dimension has the "one-hot" format, i.e., it has a total length of `num_classes`,
    with a one and `num_class-1` zeros.
    Note that this will include the background label, thus a binary mask should be treated as having two classes.

    Args:
        labels: input tensor of integers to be converted into the 'one-hot' format. Internally `labels` will be
            converted into integers `labels.long()`.
        num_classes: number of output channels, the corresponding length of `labels[dim]` will be converted to
            `num_classes` from `1`.
        dtype: the data type of the output one_hot label.
        dim: the dimension to be converted to `num_classes` channels from `1` channel, should be non-negative number.

    Example:

    For a tensor `labels` of dimensions [B]1[spatial_dims], return a tensor of dimensions `[B]N[spatial_dims]`
    when `num_classes=N` number of classes and `dim=1`.

    .. code-block:: python

        from monai.networks.utils import one_hot
        import torch

        a = torch.randint(0, 2, size=(1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=0)
        print(out.shape)  # torch.Size([2, 2, 2, 2])

        a = torch.randint(0, 2, size=(2, 1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=1)
        print(out.shape)  # torch.Size([2, 2, 2, 2, 2])

    """

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape) #1,128,128,128

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels

class BraTS2019(Dataset):
    """ BraTS2019 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None, if_blank_fill=True):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.txt'
        test_path = self._base_dir+'/val.txt'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/data/{}.h5".format(image_name), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample

class BraTS2021(Dataset):
    """ BraTS2021 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None, select = None, need_brain_mask = True, if_blank_fill=True):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.select = select
        train_path = self._base_dir+'/train_path_list.txt'
        test_path = self._base_dir+'/test_path_list.txt'
        pretrain_path = self._base_dir+'/pretrain_path_list.txt'
        self.split = split
        self.need_brain_mask = need_brain_mask
        self.blank_fill = if_blank_fill
        


        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
            if num is not None:
                self.image_list = self.image_list[:num]

        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'train_pretrain':
            with open(pretrain_path, 'r') as f:
                self.image_list = f.readlines()
            if num is not None:
                self.image_list = self.image_list[:num]
        elif split == 'val_pretrain':
            with open(pretrain_path, 'r') as f:
                self.image_list = f.readlines()
            if num is not None:
                self.image_list = self.image_list[num:]


        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}/{}h5file.h5".format(image_name, image_name), 'r')
        label = h5f['seg'][:]
        img_list = []
        if self.blank_fill:
            for i in self.select:
                if i is not None:
                    img_list.append(h5f[i][:])
                    image_shape = h5f[i][:].shape
                    if self.need_brain_mask:
                        if i == 'flair':
                            brain_mask = np.array([np.where(h5f['flair'][:]>np.min(h5f['flair'][:]),1,0)])
                else:
                    img_list.append('None')

            for i in range(len(img_list)):
                if isinstance(img_list[i],str):
                    img_list[i] = np.zeros(shape=image_shape)

        else:
            for i in self.select:
                img_list.append(h5f[i][:])

        if self.need_brain_mask:
            sample = {'image': np.stack(img_list, axis=0), 'brain': brain_mask} if 'pretrain' in self.split else  {'image': np.stack(img_list, axis=0), 'label': label.astype(np.uint8)}
        else:
            sample = {'image': np.stack(img_list, axis=0)} if 'pretrain' in self.split else  {'image': np.stack(img_list, axis=0), 'label': label.astype(np.uint8)}
        
        if self.transform:
            sample = self.transform(sample)
        return sample
    


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[1] <= self.output_size[0] or label.shape[2] <= self.output_size[1] or label.shape[3] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[1]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[2]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[3]) // 2 + 3, 0)
            image = np.pad(image, [(0,0), (pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(0,0), (pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (c, w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[:, w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[:, w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[1] <= self.output_size[0] or label.shape[2] <= self.output_size[1] or label.shape[3] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[1]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[2]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[3]) // 2 + 3, 0)
            image = np.pad(image, [(0,0), (pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(0,0), (pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(0,0), (pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (c, w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[:, w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[:, w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[:, w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1,2))
        label = np.rot90(label, k, axes=(1,2))
        axis = np.random.randint(1, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (c, w, h, d) = image.shape
        for i in range(d):
            noise = np.clip(self.sigma * np.random.randn(
                image.shape[1], image.shape[2], image.shape[3]), -2*self.sigma, 2*self.sigma)
            noise = noise + self.mu
            image[:,:,:,i] = image[:,:,:,i] + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[1], label.shape[2], label.shape[3]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label[0,:] == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class ConvertToMultiChannelBasedOnBratsClassesd(object):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, sample):

        label = sample['label']
        image = sample['image']
        result = []
        # merge label 4 and label 1 to construct TC
        result.append(np.where(((label >= 3) | (label == 1)), 1, 0))
            # merge labels 2, 4 and 1 to construct WT
        result.append(np.where(label>0, 1, 0))
            # label 4 is ET
        result.append(np.where(label>=3, 1, 0))
        label = np.stack(result, axis=0)
        return {'image': image, 'label': label}

def divide_sample(sample_base, pretrain=900, train=210, test=70):

    sample_list = [os.path.basename(x) for x in os.listdir(sample_base)]
    pretrain_sample = sample(population=sample_list, k=pretrain)

    sample_list = [x for x in sample_list if x not in pretrain_sample]
    train_sample = sample(population=sample_list, k=train)

    sample_list = [x for x in sample_list if x not in train_sample]
    test_sample = sample(population=sample_list, k=test)

    val_sample = [x for x in sample_list if x not in test_sample]

    f = open(os.path.join(sample_base, 'pretrain_path_list.txt'), 'w')
    for name in pretrain_sample:
        f.write(name + "\n")
    f.close()

    f = open(os.path.join(sample_base, 'train_path_list.txt'), 'w')
    for name in train_sample:
        f.write(name + "\n")
    f.close()

    f = open(os.path.join(sample_base, 'val_path_list.txt'), 'w')
    for name in val_sample:
        f.write(name + "\n")
    f.close()

    f = open(os.path.join(sample_base, 'test_path_list.txt'), 'w')
    for name in test_sample:
        f.write(name + "\n")
    f.close()

class RandPatchMix(object):

    def __init__(self, coefficient_sample, patches_set, input_sizes=[128,128,128], patch_size=[8,8,8], mix_rate=0.75, c_base=[0,1,2,3]):
        self.c_base = c_base
        self.patch_size = patch_size
        self.mix_rate = mix_rate
        self.input_sizes = input_sizes
        self.coefficient_sample = coefficient_sample
        self.patches_set = patches_set
    # 输入图像：CWHD，Numpy格式
    def __call__(self, sample):
        if 1==1:
            image = sample['image'].numpy()
            # image_original = sample['image_original'].numpy()
            brain_area = sample['brain'][0,].numpy()

            # pad the sample if necessary
            if (image.shape[1] < self.input_sizes[0]) or (image.shape[2] < self.input_sizes[1]) or (image.shape[3] < \
                    self.input_sizes[2]):
                
                pw = (self.input_sizes[0] - image.shape[1]) 
                ph = (self.input_sizes[1] - image.shape[2])  
                pd = (self.input_sizes[2] - image.shape[3]) 
                pw0 = pw // 2
                ph0 = ph // 2
                pd0 = pd // 2
            
                image = np.pad(image, [(0,0), (pw0, pw-pw0), (ph0, ph-ph0), (pd0, pd-pd0)],
                            mode='minimum')
                # image_original = np.pad(image_original, [(0,0), (pw0, pw-pw0), (ph0, ph-ph0), (pd0, pd-pd0)],
                #             mode='minimum')
                brain_area = np.pad(brain_area, [(pw0, pw-pw0), (ph0, ph-ph0), (pd0, pd-pd0)],
                            mode='constant', constant_values=0)
                
                

            (c, w, h, d) = image.shape

            # patches_set = [[i,j,k] for i in np.arange(0, image.shape[1], self.patch_size[0]) for j in np.arange(0, image.shape[2], self.patch_size[1]) for k in np.arange(0, image.shape[3] ,self.patch_size[2])]
            a = self.patches_set
            random.shuffle(a)
            select_patch = a[0:int(self.mix_rate * len(a))]

            sn = int(len(self.coefficient_sample['m2']+self.coefficient_sample['m3']+self.coefficient_sample['m4'])/3)
            # print(sn)
            w1 = self.patch_size[0]
            h1 = self.patch_size[1]
            d1 = self.patch_size[2]

            mix_cof_label = np.zeros(shape=(c*len(self.c_base),w,h,d)) #16
            mix_num = []
            mix_num_label = np.zeros(shape=(c*len(self.c_base),w,h,d)) #16
            mix_img = []

            # coefficient=[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]

            # coefficient_sample0 = [[c1,c2,c3] for c1 in coefficient for c2 in coefficient for c3 in coefficient]
            
            # coefficient_sample = [x  for x in coefficient_sample0 if x[0]+x[1]+x[2]<=1]

            for cc, index in zip(self.c_base, range(len(self.c_base))):
                channel_index = list(range(c))
                channel_index.remove(cc)
                bias = index * c
                coeff_balance = []
                # coeff_balance = coeff_balance + [self.coefficient_sample['m1'][0] for i in range(int(sn))]
                coeff_balance = coeff_balance + self.coefficient_sample['m2']
                if sn > len(self.coefficient_sample['m2']):
                    coeff_balance = coeff_balance + random.choices(population=self.coefficient_sample['m2'], k=int((sn-len(self.coefficient_sample['m2']))))
                coeff_balance = coeff_balance + random.sample(population=self.coefficient_sample['m3'], k=sn)
                coeff_balance = coeff_balance + random.sample(population=self.coefficient_sample['m4'], k=sn)
                # print(len(coeff_balance))
                random.shuffle(coeff_balance)
                coeff = random.choices(population=coeff_balance, k=len(select_patch))
                

                for p in range(len(select_patch)):
                    
                    c1 = round(coeff[p][0],2)
                    c2 = round(coeff[p][1],2)
                    c3 = round(coeff[p][2],2)
                    c4 = round(coeff[p][3],2)
                    
                    assert round(c1+c2+c3+c4,2) == 1
                    mix_cof_label[channel_index[0]+bias, select_patch[p][0]:select_patch[p][0]+w1, select_patch[p][1]:select_patch[p][1]+h1, select_patch[p][2]:select_patch[p][2]+d1] = c1
                    mix_cof_label[channel_index[1]+bias, select_patch[p][0]:select_patch[p][0]+w1, select_patch[p][1]:select_patch[p][1]+h1, select_patch[p][2]:select_patch[p][2]+d1] = c2
                    mix_cof_label[channel_index[2]+bias, select_patch[p][0]:select_patch[p][0]+w1, select_patch[p][1]:select_patch[p][1]+h1, select_patch[p][2]:select_patch[p][2]+d1] = c3
                mix_cof_label[cc+bias, ] = 1-np.sum(mix_cof_label[bias:bias+c,],axis=0)

                # mix_img_temp = np.sum(np.multiply(image, mix_cof_label[bias:bias+c,:,:,:]), axis=0)
                # mix_img_temp[not_brain_area] = np.min(image[cc,:,:,:])
                mix_img.append(np.sum(np.multiply(image, mix_cof_label[bias:bias+c,:,:,:]), axis=0))

                mix_num.append(np.sum(np.where(mix_cof_label[bias:bias+c,:,:,:]>0.04,1,0), axis=0))
                
            # mix_cof_label = mix_cof_label * np.array([brain_area])
            mix_num = np.stack(mix_num,axis=0)
            mix_img = np.stack(mix_img, axis=0)

            for cc, index in zip(self.c_base, range(len(self.c_base))):
                bias = index * c
                mix_num_label[bias:bias+c,:,:,:] = one_hot(torch.tensor(mix_num[index,:,:,:]).unsqueeze(0),num_classes=5,dim=0).numpy()[1:,]
            sample = {'image': image, 'mix_image': mix_img, 'mix_cof': mix_cof_label, 'mix_num': mix_num * np.array([brain_area]) - 1, 'mix_num_label': mix_num_label, 'brain':np.array([brain_area])}

        return sample


class RandPatchPuzzle(object):

    def __init__(self, patch_size=[16,16,16], input_sizes=[128,128,128]):
        self.patch_size = patch_size
        self.input_sizes = input_sizes
    # 输入图像：CWHD，Numpy格式
    def __call__(self, sample):
        if 1==1:
            image = sample['image'].numpy()
            # image_ori = sample['image_ori']
            # image_original = sample['image_original'].numpy()
            

            # pad the sample if necessary
            if (image.shape[1] < self.input_sizes[0]) or (image.shape[2] < self.input_sizes[1]) or (image.shape[3] < \
                    self.input_sizes[2]):
                
                pw = (self.input_sizes[0] - image.shape[1]) 
                ph = (self.input_sizes[1] - image.shape[2])  
                pd = (self.input_sizes[2] - image.shape[3]) 
                pw0 = pw // 2
                ph0 = ph // 2
                pd0 = pd // 2
            
                image = np.pad(image, [(0,0), (pw0, pw-pw0), (ph0, ph-ph0), (pd0, pd-pd0)],
                            mode='minimum')
                # image_original = np.pad(image_original, [(0,0), (pw0, pw-pw0), (ph0, ph-ph0), (pd0, pd-pd0)],
                #             mode='minimum')
                
            (c, w, h, d) = image.shape

            patches_set = [[i,j,k] for i in np.arange(0, image.shape[1], self.patch_size[0]) for j in np.arange(0, image.shape[2], self.patch_size[1]) for k in np.arange(0, image.shape[3] ,self.patch_size[2])]
            random.shuffle(patches_set)
            sn = int(0.25 * len(patches_set))
            select_patch_t1 = patches_set[0:sn]
            select_patch_t1ce = patches_set[sn:2*sn]
            select_patch_t2 = patches_set[2*sn:3*sn]
            select_patch_flair = patches_set[3*sn:4*sn]

            w1 = self.patch_size[0]
            h1 = self.patch_size[1]
            d1 = self.patch_size[2]
            
            mix_image = np.zeros(shape=(c,w,h,d))
            
            for p in range(sn):
                mix_image[0, select_patch_t1[p][0]:select_patch_t1[p][0]+w1, select_patch_t1[p][1]:select_patch_t1[p][1]+h1, select_patch_t1[p][2]:select_patch_t1[p][2]+d1] = 1
                mix_image[1, select_patch_t1ce[p][0]:select_patch_t1ce[p][0]+w1, select_patch_t1ce[p][1]:select_patch_t1ce[p][1]+h1, select_patch_t1ce[p][2]:select_patch_t1ce[p][2]+d1] = 1
                mix_image[2, select_patch_t2[p][0]:select_patch_t2[p][0]+w1, select_patch_t2[p][1]:select_patch_t2[p][1]+h1, select_patch_t2[p][2]:select_patch_t2[p][2]+d1] = 1
                mix_image[3, select_patch_flair[p][0]:select_patch_flair[p][0]+w1, select_patch_flair[p][1]:select_patch_flair[p][1]+h1, select_patch_flair[p][2]:select_patch_flair[p][2]+d1] = 1

            mix = image * mix_image

            sample = {'image': image, 'mix_image': mix}
            
            
        return sample
    
class Rand_Mask_Mix(object):

    def __init__(self, coefficient_sample, patches_set, patch_size=[16,16,16], input_sizes=[128,128,128], mix_num_flag=True):
        self.patch_size = patch_size
        self.input_sizes = input_sizes
        self.coefficient_sample = coefficient_sample
        self.patches_set = patches_set
        self.mix_num_flag = mix_num_flag
        
    # 输入图像：CWHD，Numpy格式
    def __call__(self, sample):
        if 1==1:
            image = sample['image'].numpy()
            brain_area = sample['brain'][0,].numpy()

            # pad the sample if necessary
            if (image.shape[1] < self.input_sizes[0]) or (image.shape[2] < self.input_sizes[1]) or (image.shape[3] < \
                    self.input_sizes[2]):
                
                pw = (self.input_sizes[0] - image.shape[1]) 
                ph = (self.input_sizes[1] - image.shape[2])  
                pd = (self.input_sizes[2] - image.shape[3]) 
                pw0 = pw // 2
                ph0 = ph // 2
                pd0 = pd // 2
            
                image = np.pad(image, [(0,0), (pw0, pw-pw0), (ph0, ph-ph0), (pd0, pd-pd0)],
                            mode='minimum')
                brain_area = np.pad(brain_area, [(pw0, pw-pw0), (ph0, ph-ph0), (pd0, pd-pd0)],
                            mode='constant', constant_values=0)
                
            (c, w, h, d) = image.shape

            a = self.patches_set
            random.shuffle(a)
            sn = int(0.25 * len(a)) #每个通道显示个数
            cn = int(len(self.coefficient_sample['m1']+self.coefficient_sample['m2']+self.coefficient_sample['m3']+self.coefficient_sample['m4'])/4) #每种个数选择数量

            #选择进行显示的
            select_patch_t1 = a[0:sn]
            select_patch_t1ce = a[sn:2*sn]
            select_patch_t2 = a[2*sn:3*sn]
            select_patch_flair = a[3*sn:4*sn]

            w1 = self.patch_size[0]
            h1 = self.patch_size[1]
            d1 = self.patch_size[2]

            #为每个模态选择混合系数
            coeff = []
            for i in range(4):
                coeff_balance = [self.coefficient_sample['m1'][0] for i in range(int(cn))]
                coeff_balance = coeff_balance + self.coefficient_sample['m2']
                if cn > len(self.coefficient_sample['m2']):
                    coeff_balance = coeff_balance + random.choices(population=self.coefficient_sample['m2'], k=cn-len(self.coefficient_sample['m2']))
                coeff_balance = coeff_balance + random.sample(population=self.coefficient_sample['m3'], k=cn)
                if cn > len(self.coefficient_sample['m3']):
                    coeff_balance = coeff_balance + random.choices(population=self.coefficient_sample['m3'], k=cn-len(self.coefficient_sample['m3']))
                coeff_balance = coeff_balance + random.sample(population=self.coefficient_sample['m4'], k=cn)
                if cn > len(self.coefficient_sample['m4']):
                    coeff_balance = coeff_balance + random.choices(population=self.coefficient_sample['m4'], k=cn-len(self.coefficient_sample['m4']))
                random.shuffle(coeff_balance)#这是一个均衡的混合系数池子
                coeff.append(random.choices(population=coeff_balance, k=sn))

            mix_image = np.zeros(shape=(c,w,h,d))
            mix_coff_image = np.zeros(shape=(4*c,w,h,d))
            mix_num_label = np.zeros(shape=(4*c,w,h,d))
            mix = []
            mix_num = []
            
            #mix image 是图像的mask
            repeat_num = self.patch_size[0]*self.patch_size[1]*self.patch_size[2]
            for p in range(sn):
                mix_coff_image[[1,2,3],select_patch_t1[p][0]:select_patch_t1[p][0]+w1, select_patch_t1[p][1]:select_patch_t1[p][1]+h1, select_patch_t1[p][2]:select_patch_t1[p][2]+d1] = np.repeat(coeff[0][p][0:3],repeat_num).reshape((3,self.patch_size[0],self.patch_size[1],self.patch_size[2]))
                mix_coff_image[0,select_patch_t1[p][0]:select_patch_t1[p][0]+w1, select_patch_t1[p][1]:select_patch_t1[p][1]+h1, select_patch_t1[p][2]:select_patch_t1[p][2]+d1] = np.repeat(coeff[0][p][3],repeat_num).reshape((self.patch_size[0],self.patch_size[1],self.patch_size[2]))

                mix_coff_image[[4,6,7],select_patch_t1ce[p][0]:select_patch_t1ce[p][0]+w1, select_patch_t1ce[p][1]:select_patch_t1ce[p][1]+h1, select_patch_t1ce[p][2]:select_patch_t1ce[p][2]+d1] = np.repeat(coeff[1][p][0:3],repeat_num).reshape((3,self.patch_size[0],self.patch_size[1],self.patch_size[2]))
                mix_coff_image[5,select_patch_t1ce[p][0]:select_patch_t1ce[p][0]+w1, select_patch_t1ce[p][1]:select_patch_t1ce[p][1]+h1, select_patch_t1ce[p][2]:select_patch_t1ce[p][2]+d1] = np.repeat(coeff[1][p][3],repeat_num).reshape((self.patch_size[0],self.patch_size[1],self.patch_size[2]))

                mix_coff_image[[8,9,11], select_patch_t2[p][0]:select_patch_t2[p][0]+w1, select_patch_t2[p][1]:select_patch_t2[p][1]+h1, select_patch_t2[p][2]:select_patch_t2[p][2]+d1] = np.repeat(coeff[2][p][0:3],repeat_num).reshape((3,self.patch_size[0],self.patch_size[1],self.patch_size[2]))
                mix_coff_image[10, select_patch_t2[p][0]:select_patch_t2[p][0]+w1, select_patch_t2[p][1]:select_patch_t2[p][1]+h1, select_patch_t2[p][2]:select_patch_t2[p][2]+d1] = np.repeat(coeff[2][p][3],repeat_num).reshape((self.patch_size[0],self.patch_size[1],self.patch_size[2]))

                mix_coff_image[[12,13,14], select_patch_flair[p][0]:select_patch_flair[p][0]+w1, select_patch_flair[p][1]:select_patch_flair[p][1]+h1, select_patch_flair[p][2]:select_patch_flair[p][2]+d1] = np.repeat(coeff[3][p][0:3],repeat_num).reshape((3,self.patch_size[0],self.patch_size[1],self.patch_size[2]))
                mix_coff_image[15, select_patch_flair[p][0]:select_patch_flair[p][0]+w1, select_patch_flair[p][1]:select_patch_flair[p][1]+h1, select_patch_flair[p][2]:select_patch_flair[p][2]+d1] = np.repeat(coeff[3][p][3],repeat_num).reshape((self.patch_size[0],self.patch_size[1],self.patch_size[2]))
            

            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[0,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t1_2_t1_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[1,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t1_2_t1ce_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[2,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t1_2_t2_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[3,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t1_2_flair_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[4,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t1ce_2_t1_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[5,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t1ce_2_t1ce_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[6,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t1ce_2_t2_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[7,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t1ce_2_flair_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[8,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t2_2_t1_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[9,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t2_2_t1ce_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[10,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t2_2_t2_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[11,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t2_2_flair_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[12,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','flair_2_t1_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[13,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','flair_2_t1ce_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[14,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','flair_2_t2_coff.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(mix_coff_image[15,]),os.path.join('/mnt/liangjw/SSL4MOD/check_view','flair_2_flair_coff.nii'))


            mix_image[0,] = np.where(np.sum(mix_coff_image[0:4,], axis=0)>=0.04, 1, 0)
            mix_image[1,] = np.where(np.sum(mix_coff_image[4:8,], axis=0)>=0.04, 1, 0)
            mix_image[2,] = np.where(np.sum(mix_coff_image[8:12,], axis=0)>=0.04, 1, 0)
            mix_image[3,] = np.where(np.sum(mix_coff_image[12:16,], axis=0)>=0.04, 1, 0)

            for i in range(4):
                mix.append(np.sum(np.multiply(image, mix_coff_image[4*i:4*i+4,:,:,:]), axis=0))
                if self.mix_num_flag:
                    mix_num.append(np.sum(np.where(mix_coff_image[4*i:4*i+4,:,:,:]>=0.04,1,0), axis=0))
                
            # mix_cof_label = mix_cof_label * np.array([brain_area])
            if self.mix_num_flag:
                mix_num = np.stack(mix_num,axis=0)

            mix = np.stack(mix, axis=0)

            # sitk.WriteImage(sitk.GetImageFromArray(np.int32(mix_num[0,])),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t1num.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(np.int32(mix_num[1,])),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t1cenum.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(np.int32(mix_num[2,])),os.path.join('/mnt/liangjw/SSL4MOD/check_view','t2num.nii'))
            # sitk.WriteImage(sitk.GetImageFromArray(np.int32(mix_num[3,])),os.path.join('/mnt/liangjw/SSL4MOD/check_view','flairnum.nii'))

            if self.mix_num_flag:
                for i in range(4):
                    mix_num_label[4*i:4*i+4,:,:,:] = one_hot(torch.tensor(mix_num[i,:,:,:]).unsqueeze(0),num_classes=5,dim=0).numpy()[1:,]
            if self.mix_num_flag:
                sample = {'image': image, 'mix_image': mix, 'mix_cof': mix_coff_image, 'mix_num': mix_num * np.multiply(brain_area, mix_image) - 1, 'mix_num_label': mix_num_label, 'brain':np.array([np.multiply(brain_area, mix_image)])}
            else:
                sample = {'image': image, 'mix_image': mix, 'mix_cof': mix_coff_image, 'brain':np.array([np.multiply(brain_area, mix_image)])}

        return sample

                

class Clip(object):

    def __init__(self, percent):
        self.percent = percent

    def __call__(self, sample):
        image = sample['image']
        channel = image.shape[0]
        for i in range(channel):
            cdf = exposure.cumulative_distribution(image)
            high_boundary = cdf[1][cdf[0] <= (1-self.percent)][-1]
            low_boundary = cdf[1][cdf[0] >= self.percent][0]
            image[i,] = np.clip(image[i,], low_boundary, high_boundary)
        return {'image':image}
        
        

if __name__ == '__main__':
    # import sys
    # sys.path.append('/raid5/liangjw/SSL4MIS/code/utils')
    # import SimpleITK as sitk
    # from model_genesis import *
    from torch.utils.data import DataLoader
    import monai.transforms as monai_transforms

    # config = models_genesis_config()

    # original_mri = sitk.ReadImage('/raid5/liangjw/Glioma/fix_dataset_2021/BraTS2021_00000/BraTS2021_00000_t1_MNI.nii.gz')
    # m_ori = original_mri.GetOrigin()
    # m_spa = original_mri.GetSpacing()
    # m_dir = original_mri.GetDirection()

    # transform = monai_transforms.Compose(
    #     [   monai_transforms.SpatialPadd(keys=["image"],
    #                                      spatial_size=(128,128,128),
    #                                      mode='edge'),
    #         monai_transforms.CenterSpatialCropd(
    #             keys=["image"],
    #             roi_size=[128,128,128]
    #         ),
    #         ModelGenesis_Aug(config=config),

    #         monai_transforms.ToTensord(keys=["image","label"], track_meta=False),
    #     ]
    # )
    transform2 = monai_transforms.Compose(
        [   
            monai_transforms.RandZoomd(
                keys=["image"],
                prob=0.8,
                max_zoom=1.4,
                min_zoom=0.7,
                mode=("bilinear")
                ),
            monai_transforms.RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=(128,128,128),
                num_samples=1,
                random_size = False
            ),
            monai_transforms.RandRotated(
                range_x=0.5236,
                range_y=0.5236,
                range_z=0.5236,
                keys=["image"],
                mode=("bilinear"),
                prob=0.8
            ),
            Rand_Mask_Mix(patch_size=[32,32,32]),
            monai_transforms.ToTensord(keys=["image","mix_image"], track_meta=False),
        ]
    )

    db_train = BraTS2021(base_dir='/mnt/liangjw/SSL4MOD/data/BraTS2021',
                         split='train_pretrain',
                         num=None,
                         select=['t1','t1ce','t2','flair'],
                         transform=transform2,
                         need_brain_mask=False)
    
    trainloader = DataLoader(db_train, batch_size=1, shuffle=True,
                             num_workers=4, pin_memory=True)
    
    for i_batch, sampled_batch in enumerate(trainloader):
        mix_volume_batch = sampled_batch[0]['mix_image']
        volume_batch = sampled_batch[0]['image']
        volume_batch_ori = sampled_batch[0]['image_ori']
        snapshot_path_check = os.path.join('/mnt/liangjw/SSL4MOD/check_view','sample_'+str(i_batch))
        if not os.path.exists(snapshot_path_check):
            os.mkdir(snapshot_path_check)
        example1_0 = sitk.GetImageFromArray(mix_volume_batch[0,0,:,:,:])
        example1_1 = sitk.GetImageFromArray(mix_volume_batch[0,1,:,:,:])
        example1_2 = sitk.GetImageFromArray(mix_volume_batch[0,2,:,:,:])
        # example1.SetOrigin(m_ori)
        # example1.SetSpacing(m_spa)
        # example1.SetDirection(m_dir)
        example1_3 = sitk.GetImageFromArray(mix_volume_batch[0,3,:,:,:])
        # example1_2.SetOrigin(m_ori)
        # example1_2.SetSpacing(m_spa)
        # example1_2.SetDirection(m_dir)
        
        sitk.WriteImage(example1_0,os.path.join(snapshot_path_check,'t1_mix'+'.nii.gz'))
        sitk.WriteImage(example1_1,os.path.join(snapshot_path_check,'t1ce_mix'+'.nii.gz'))
        sitk.WriteImage(example1_2,os.path.join(snapshot_path_check,'t2_mix'+'.nii.gz'))
        sitk.WriteImage(example1_3,os.path.join(snapshot_path_check,'flair_mix'+'.nii.gz'))
       

        example2_0 = sitk.GetImageFromArray(volume_batch[0,0,:,:,:])
        # example5.SetOrigin(m_ori)
        # example5.SetSpacing(m_spa)
        # example5.SetDirection(m_dir)
        
        example2_1 = sitk.GetImageFromArray(volume_batch[0,1,:,:,:])
        example2_2 = sitk.GetImageFromArray(volume_batch[0,2,:,:,:])
        example2_3 = sitk.GetImageFromArray(volume_batch[0,3,:,:,:])
        # example7.SetOrigin(m_ori)
        # example7.SetSpacing(m_spa)
        # example7.SetDirection(m_dir)
        
        sitk.WriteImage(example2_0,os.path.join(snapshot_path_check,'aug_t1'+'.nii.gz'))
        sitk.WriteImage(example2_1,os.path.join(snapshot_path_check,'aug_t1ce'+'.nii.gz'))
        sitk.WriteImage(example2_2,os.path.join(snapshot_path_check,'aug_t2'+'.nii.gz'))
        sitk.WriteImage(example2_3,os.path.join(snapshot_path_check,'aug_flair'+'.nii.gz'))


        example3_0 = sitk.GetImageFromArray(volume_batch_ori[0,0,:,:,:])
        # example5.SetOrigin(m_ori)
        # example5.SetSpacing(m_spa)
        # example5.SetDirection(m_dir)
        
        example3_1 = sitk.GetImageFromArray(volume_batch_ori[0,1,:,:,:])
        example3_2 = sitk.GetImageFromArray(volume_batch_ori[0,2,:,:,:])
        example3_3 = sitk.GetImageFromArray(volume_batch_ori[0,3,:,:,:])
        # example7.SetOrigin(m_ori)
        # example7.SetSpacing(m_spa)
        # example7.SetDirection(m_dir)
        
        sitk.WriteImage(example3_0,os.path.join(snapshot_path_check,'original_t1'+'.nii.gz'))
        sitk.WriteImage(example3_1,os.path.join(snapshot_path_check,'original_t1ce'+'.nii.gz'))
        sitk.WriteImage(example3_2,os.path.join(snapshot_path_check,'original_t2'+'.nii.gz'))
        sitk.WriteImage(example3_3,os.path.join(snapshot_path_check,'original_flair'+'.nii.gz'))
        print(i_batch)
    # a = torch.tensor([[[1,3,4],[2,2,3]],[[1,1,3],[2,3,4]],[[1,3,4],[1,1,1]],[[2,2,2],[4,4,4]]])
    # b = one_hot(labels=a.unsqueeze(0),num_classes=5,dim=0)
    # print(a)
    # print(b[1:])