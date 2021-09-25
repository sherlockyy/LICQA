from pathlib import Path
from PIL import Image
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F


def get_dist_type(img):
    codec_ids = {'FactMSE': 0, 'HyperMSE': 1, 'HierarchMSE': 2, 'HierarchMSIM': 3, 'CAEntropyMSIM': 4, 'FRICwRNN': 5,
                 'HiFiC': 6, 'WebP': 7, 'VVC': 8, 'JPEG': 9}
    codec = img.parent.name
    return codec_ids[codec]


def get_transform(args, phase='train'):
    trans = []
    if args.resize:
        trans.append(Resize(args.resize))

    if phase == 'train':
        if args.crop_train:
            trans.append(RandomCrop(args.crop_train))
        if args.hori_flip:
            trans.append(RandomHorizontalFlip(args.hori_flip))
    else:
        if args.crop_test:
            trans.append(CenterCrop(args.crop_test))

    trans.extend([ToTensor(), Normalize(args.mean, args.std)])

    if args.multi_crop_size:
        if phase == 'train':
            if args.multi_crop_size != args.crop_train:
                trans.append(RandomCropPatch(args.multi_crop_size, args.multi_crop_num))
        else:
            trans.append(NonOverlappingCropPatch(args.multi_crop_size))
    return Compose(trans)


class Data:
    def __init__(self, args):
        print('Loading data...')
        self.loader_train = None
        if not args.test_only:
            ts = get_transform(args, phase='train')
            dataset_train = IQADatabase(args.data_dir, args.data_train, ts, args.mos_norm)
            self.loader_train = DataLoader(
                dataset=dataset_train,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.device.type == 'cpu',
                num_workers=args.n_threads,
            )
            print(f'Training set {len(dataset_train)} samples')

            ts = get_transform(args, phase='val')
            dataset_val = IQADatabase(args.data_dir, args.data_val, ts, args.mos_norm)
            self.loader_val = DataLoader(
                dataset=dataset_val,
                batch_size=1,
                shuffle=False,
                pin_memory=not args.device.type == 'cpu',
                num_workers=args.n_threads,
            )
            print(f'Val set {len(dataset_val)} samples')

        ts = get_transform(args, phase='test')
        dataset_test = IQADatabase(args.data_dir, args.data_test, ts, args.mos_norm)
        self.loader_test = DataLoader(
                dataset=dataset_test,
                batch_size=1,
                shuffle=False,
                pin_memory=not args.device.type == 'cpu',
                num_workers=args.n_threads,
            )
        print(f'Test set {len(dataset_test)} samples')


class IQADatabase(Dataset):
    def __init__(self, data_root, train_file, transform, mos_norm):
        data_root = Path(data_root)
        self.transform = transform
        self.ref_imgs = []
        self.dst_imgs = []
        self.dst_types = []
        self.mos = []
        self.dst_ref = {}

        lines = []
        for train_file_ in train_file:
            with open(train_file_, 'r') as f:
                lines_ = f.readlines()
                lines.extend(lines_)

        for i in range(len(lines)):
            ref, dst, _, s = lines[i].strip().split(',')
            ref_path = data_root / ref
            dst_path = data_root / dst

            self.dst_types.append(get_dist_type(dst_path))
            self.dst_imgs.append(dst_path)
            self.mos.append(float(s))
            if ref_path not in self.ref_imgs:
                self.ref_imgs.append(ref_path)
            self.dst_ref[i] = self.ref_imgs.index(ref_path)
        
        self.mos = torch.tensor(self.mos, dtype=torch.float32)
        if mos_norm:
            min_val, max_val = mos_norm
            self.mos = (self.mos - min_val) / (max_val - min_val)

    def __len__(self) -> int:
        return len(self.dst_imgs)

    def __getitem__(self, item):
        ref_index = self.dst_ref[item]
        ref_img = self.ref_imgs[ref_index]
        dst_img = self.dst_imgs[item]
        dst_type = self.dst_types[item]
        mos = self.mos[item]

        ref_img = Image.open(ref_img).convert('RGB')
        dst_img = Image.open(dst_img).convert('RGB')
        ref_img, dst_img = self.transform(ref_img, dst_img)
        return ref_img, dst_img, dst_type, mos


class ValDatabase(Dataset):
    def __init__(self, data_root, ref_dst, transform):
        data_root = Path(data_root)
        self.transform = transform
        self.ref_imgs, self.dst_imgs = ref_dst

    def __len__(self) -> int:
        return len(self.dst_imgs)

    def __getitem__(self, item):
        ref_img = self.ref_imgs[item]
        dst_img = self.dst_imgs[item]

        ref_img = Image.open(ref_img).convert('RGB')
        dst_img = Image.open(dst_img).convert('RGB')
        ref_img, dst_img = self.transform(ref_img, dst_img)
        return ref_img, dst_img


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size
        self.interpolation = interpolation

    def __call__(self, img1, img2):
        img1 = F.resize(img1, self.size, self.interpolation)
        img2 = F.resize(img2, self.size, self.interpolation)
        return img1, img2


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    @staticmethod
    def get_params(img, size):
        w, h = img.size
        th, tw = size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img1, img2):
        i, j, h, w = self.get_params(img1, self.size)
        img1 = F.crop(img1, i, j, h, w)
        img2 = F.crop(img2, i, j, h, w)
        return img1, img2


class CenterCrop(object):
    """Crop from center of image in a sample.
    Args:
        size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    @staticmethod
    def get_params(img, size):
        w, h = img.size
        th, tw = size
        if w == tw and h == th:
            return 0, 0, h, w

        i = (h - th) // 2
        j = (w - tw) // 2
        return i, j, th, tw

    def __call__(self, img1, img2):
        i, j, h, w = self.get_params(img1, self.size)
        img1 = F.crop(img1, i, j, h, w)
        img2 = F.crop(img2, i, j, h, w)
        return img1, img2


class NonOverlappingCropPatch(object):
    """
    NonOverlapping crop input 3-D tensors (C, H, W) to patches
    Args:
        size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, size):
        assert isinstance(size, int)
        self.size = size

    def __call__(self, img1, img2):
        c, h, w = img1.shape
        patches1 = ()
        patches2 = ()
        stride = self.size
        for i in range(0, h - stride + 1, stride):
            for j in range(0, w - stride + 1, stride):
                patch1 = img1[:, i:i+stride, j:j+stride]
                patch2 = img2[:, i:i+stride, j:j+stride]
                patches1 = patches1 + (patch1,)
                patches2 = patches2 + (patch2,)
        return torch.stack(patches1), torch.stack(patches2)


class RandomCropPatch(object):
    """
    Random crop input 3-D tensors (C, H, W) to patches
    Args:
        size (tuple or int): Desired output size. If int, square crop
            is made.
        num (int): Number of cropped patches.
    """
    def __init__(self, size, num):
        assert isinstance(size, int)
        self.size = size
        self.patch_num = num

    def __call__(self, img1, img2):
        c, h, w = img1.shape
        patches1 = ()
        patches2 = ()
        stride = self.size
        for i in range(self.patch_num):
            w1 = np.random.randint(low=0, high=w-stride+1)
            h1 = np.random.randint(low=0, high=h-stride+1)

            patch1 = img1[:, h1:h1+stride, w1:w1+stride]
            patch2 = img2[:, h1:h1+stride, w1:w1+stride]
            patches1 = patches1 + (patch1,)
            patches2 = patches2 + (patch2,)
        return torch.stack(patches1), torch.stack(patches2)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img1, img2):
        if random.random() < self.p:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)
        return img1, img2


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor1, tensor2):
        tensor1 = F.normalize(tensor1, self.mean, self.std, self.inplace)
        tensor2 = F.normalize(tensor2, self.mean, self.std, self.inplace)
        return tensor1, tensor2


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    """
    def __call__(self, pic1, pic2):
        pic1 = F.to_tensor(pic1)
        pic2 = F.to_tensor(pic2)
        return pic1, pic2
