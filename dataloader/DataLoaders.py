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

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from dataloader.KittiDepthDataset import KittiDepthDataset
import random
import glob
num_worker = 8

def KittiDataLoader(params):
    # Input images are 16-bit, but only 15-bits are utilized, so we normalized the data to [0:1] using a normalization factor
    norm_factor = params['data_normalize_factor']
    invert_depth = params['invert_depth']
    ds_dir = params['dataset_dir']
    rgb_dir = params['rgb_dir'] if 'rgb_dir' in params else None
    rgb2gray = params['rgb2gray'] if 'rgb2gray' in params else False
    fill_depth = params['fill_depth'] if 'fill_depth' in params else False
    flip = params['flip'] if ('flip' in params) else False
    dataset = params['dataset'] if 'dataset' in params else 'KittiDepthDataset'
    num_worker = 8

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}

    ###### Training Set ######
    train_data_path = os.path.join(ds_dir, 'data_depth_velodyne/train')
    train_gt_path = os.path.join(ds_dir, 'data_depth_annotated/train')

    if params['transform_type'] == 'center':
        train_transform = transforms.Compose([transforms.CenterCrop((352, 1216))])
    else:
        train_transform = None

    image_datasets['train'] = eval(dataset)(train_data_path, train_gt_path, setname='train',
                                                transform=train_transform, norm_factor=norm_factor,
                                                invert_depth=invert_depth,
                                                rgb_dir=rgb_dir, rgb2gray=rgb2gray, fill_depth=fill_depth, flip=flip)

    # Select the desired number of images from the training set
    if params['train_on'] != 'full':
        image_datasets['train'].data = image_datasets['train'].data[0:params['train_on']]  # file directions
        image_datasets['train'].gt = image_datasets['train'].gt[0:params['train_on']]

    dataloaders['train'] = DataLoader(image_datasets['train'], shuffle=True, batch_size=params['train_batch_sz'],
                                      num_workers=num_worker)
    dataset_sizes['train'] = {len(image_datasets['train'])}

    ###### Validation Set ######
    val_data_path = os.path.join(ds_dir, 'data_depth_velodyne/val')
    val_gt_path = os.path.join(ds_dir, 'data_depth_annotated/val')

    val_transform = transforms.Compose([transforms.CenterCrop((352, 1216))])

    image_datasets['val'] = eval(dataset)(val_data_path, val_gt_path, setname='val', transform=val_transform,
                                              norm_factor=norm_factor, invert_depth=invert_depth,
                                              rgb_dir=rgb_dir, rgb2gray=rgb2gray, fill_depth=fill_depth, flip=flip)
    dataloaders['val'] = DataLoader(image_datasets['val'], shuffle=False, batch_size=params['val_batch_sz'],
                                    num_workers=num_worker)
    dataset_sizes['val'] = {len(image_datasets['val'])}

    ###### Selected Validation set ######
    selval_data_path = os.path.join(ds_dir, 'depth_selection/val_selection_cropped/velodyne_raw')
    selval_gt_path = os.path.join(ds_dir, 'depth_selection/val_selection_cropped/groundtruth_depth')

    image_datasets['selval'] = eval(dataset)(selval_data_path, selval_gt_path, setname='selval', transform=None,
                                                 norm_factor=norm_factor, invert_depth=invert_depth,
                                                 rgb_dir=rgb_dir, rgb2gray=rgb2gray, fill_depth=fill_depth, flip=flip)

    dataloaders['selval'] = DataLoader(image_datasets['selval'], shuffle=False, batch_size=params['test_batch_sz'],
                                       num_workers=num_worker)
    dataset_sizes['selval'] = {len(image_datasets['selval'])}

    ###### Selected test set ######
    test_data_path = os.path.join(ds_dir, 'depth_selection/test_depth_completion_anonymous/velodyne_raw')
    test_gt_path = os.path.join(ds_dir, 'depth_selection/test_depth_completion_anonymous/velodyne_raw')

    image_datasets['test'] = eval(dataset)(test_data_path, test_gt_path, setname='test', transform=None,
                                               norm_factor=norm_factor, invert_depth=invert_depth,
                                               rgb_dir=rgb_dir, rgb2gray=rgb2gray, fill_depth=fill_depth)

    dataloaders['test'] = DataLoader(image_datasets['test'], shuffle=False, batch_size=params['test_batch_sz'],
                                     num_workers=num_worker)
    dataset_sizes['test'] = {len(image_datasets['test'])}

    print(dataset_sizes)

    return dataloaders, dataset_sizes







