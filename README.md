# msg_chn_wacv20
This repository contains the network configurations (in PyTorch) of our paper "[A Multi-Scale Guided Cascade Hourglass Network for Depth Completion](http://openaccess.thecvf.com/content_WACV_2020/papers/Li_A_Multi-Scale_Guided_Cascade_Hourglass_Network_for_Depth_Completion_WACV_2020_paper.pdf)" by Ang Li, Zejian Yuan, Yonggen Ling, Wanchao Chi, Shenghao Zhang and Chong Zhang.

## Introduction
Depth completion, a task to estimate the dense depth map from sparse measurement under the guidance from the
high-resolution image, is essential to many computer vision applications. Most previous methods building on fully convolutional networks can not handle diverse patterns in the depth map efficiently and effectively. We propose a multi-scale guided cascade hourglass network to tackle this problem. Structures at different levels are captured by specialized hourglasses in the cascade network with sparse inputs in various sizes. An encoder extracts multiscale features from color image to provide deep guidance
for all the hourglasses. A multi-scale training strategy further activates the effect of cascade stages. With the role of
each sub-module divided explicitly, we can implement components with simple architectures. Extensive experiments show that our lightweight model achieves competitive results compared with state-of-the-art in KITTI depth completion benchmark, with low complexity in run-time.

<p align="center">
  <img src="https://github.com/anglixjtu/msg_chn_wacv20/blob/master/demo/video5.gif" alt="photo not available" height="50%">
</p>


## Dependency
- Python 3.5
- Pytorch 1.1.0

## Network
The implementation of our network is in 'network.py'. It takes the sparse depth and the rgb image (normalized to 0~1) as inputs， outputs the predictions from the last, the second, and the first sub-network in sequence. The output from the last network ('output_d11') is used for final test.

    Inputs: input_d, input_rgb
    Outputs: output_d11， output_d12， output_d14
             # outputs from the last, the second, and the first sub-network

※NOTE: We recently improve the accuracy by adding the skip connections between the depth encoders and the depth decoders at the previous stage. This vision of network has 32 channels rather than 64 channels in our paper. The 32-channel network performs similarly on the test test, but has a much smaller number of parameters and a shorter run time. You can find more details in [Results](#results)

## Training

## Results
