"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

from PIL import Image
import torch
import numpy as np
import glob
import torchvision
import random
from torch.utils.data import DataLoader, Dataset


class KittiDepthDataset(Dataset):
    def __init__(self, data_path, gt_path, setname='train', transform=None, norm_factor=256, invert_depth=False,
                 rgb_dir=None, rgb2gray=False, fill_depth = False, flip = False, blind = False):
        self.data_path = data_path
        self.gt_path = gt_path
        self.setname = setname
        self.transform = transform
        self.norm_factor = norm_factor
        self.invert_depth = invert_depth
        self.rgb_dir = rgb_dir
        self.rgb2gray = rgb2gray
        self.fill_depth = fill_depth
        self.flip = flip
        self.blind = blind

        self.data = list(sorted(glob.iglob(self.data_path + "/**/*.png", recursive=True)))
        self.gt = list(sorted(glob.iglob(self.gt_path + "/**/*.png", recursive=True)))

        assert (len(self.gt) == len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        # Check if Data filename is equal to GT filename
        if self.setname == 'train' or self.setname == 'val':
            data_path = self.data[item].split(self.setname)[1]
            gt_path = self.gt[item].split(self.setname)[1]

            assert (data_path[0:25] == gt_path[0:25])  # Check folder name

            data_path = data_path.split('image')[1]
            gt_path = gt_path.split('image')[1]

            assert (data_path == gt_path)  # Check filename

            # Set the certainty path
            sep = str(self.data[item]).split('data_depth_velodyne')

        elif self.setname == 'selval':
            data_path = self.data[item].split('00000')[1]
            gt_path = self.gt[item].split('00000')[1]
            assert (data_path == gt_path)
            # Set the certainty path
            sep = str(self.data[item]).split('/velodyne_raw/')



        # Read images and convert them to 4D floats
        data = Image.open(str(self.data[item]))

        gt = Image.open(str(self.gt[item]))



        # Read RGB images
        if self.setname == 'train' or self.setname == 'val':
            gt_path = str(self.gt[item])
            idx = gt_path.find('2011')
            day_dir = gt_path[idx:idx + 10]
            idx2 = gt_path.find('groundtruth')
            fname = gt_path[idx2 + 12:]
            rgb_path = self.rgb_dir + '/' + day_dir + '/' + gt_path[idx:idx + 26] + '/' + fname[
                                                                                          :8] + '/data/' + fname[9:]
            rgb = Image.open(rgb_path)


        elif self.setname == 'selval':
            data_path = str(self.data[item])
            idx = data_path.find('velodyne_raw')
            fname = data_path[idx + 12:]
            idx2 = fname.find('velodyne_raw')
            rgb_path = data_path[:idx] + 'image' + fname[:idx2] + 'image' + fname[idx2 + 12:]
            rgb = Image.open(rgb_path)
        elif self.setname == 'test':
            data_path = str(self.data[item])
            idx = data_path.find('velodyne_raw')
            fname = data_path[idx + 12:]
            rgb_path = data_path[:idx] + 'image/' + fname
            rgb = Image.open(rgb_path)





        if self.rgb2gray:
            t = torchvision.transforms.Grayscale(1)
            rgb = t(rgb)

        W, H = data.size





        # Apply transformations if given
        if self.transform is not None:
            data = self.transform(data)
            gt = self.transform(gt)
            rgb = self.transform(rgb)
        if self.transform is None and self.setname == 'train':
            crop_lt_u = random.randint(0, W - 1216)
            crop_lt_v = random.randint(0, H - 352)
            data = data.crop((crop_lt_u, crop_lt_v, crop_lt_u+1216, crop_lt_v+352))
            gt = gt.crop((crop_lt_u, crop_lt_v, crop_lt_u + 1216, crop_lt_v + 352))
            rgb = rgb.crop((crop_lt_u, crop_lt_v, crop_lt_u + 1216, crop_lt_v + 352))


        if self.flip and random.randint(0, 1) and self.setname == 'train':
            data = data.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)


        # Convert to numpy
        data = np.array(data, dtype=np.float16)
        gt = np.array(gt, dtype=np.float16)


        #blind
        if self.blind and (self.setname == 'train'):
            blind_start = random.randint(100, H - 50)
            data[blind_start:blind_start+50, :] = 0

        # define the certainty
        C = (data > 0).astype(float)


        # Normalize the data
        data = data / self.norm_factor  # [0,1]
        gt = gt / self.norm_factor

        # Expand dims into Pytorch format
        data = np.expand_dims(data, 0)
        gt = np.expand_dims(gt, 0)
        C = np.expand_dims(C, 0)

        # Convert to Pytorch Tensors
        data = torch.tensor(data, dtype=torch.float)
        gt = torch.tensor(gt, dtype=torch.float)
        C = torch.tensor(C, dtype=torch.float)

        # Convert depth to disparity
        if self.invert_depth:
            data[data == 0] = -1
            data = 1 / data
            data[data == -1] = 0

            gt[gt == 0] = -1
            gt = 1 / gt
            gt[gt == -1] = 0

        # Convert RGB image to tensor
        rgb = np.array(rgb, dtype=np.float16)
        rgb /= 255
        if self.rgb2gray:
            rgb = np.expand_dims(rgb, 0)
        else:
            rgb = np.transpose(rgb, (2, 0, 1))
        rgb = torch.tensor(rgb, dtype=torch.float)

        return data, C, gt, item, rgb
