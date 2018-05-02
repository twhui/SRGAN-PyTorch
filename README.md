# SRResNet-pyTorch

This repository contains the pyTorch re-implementation of <strong>SRResNet</strong> in the paper <a href="https://arxiv.org/abs/1609.04802">Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network</a>, CVPR17. 

We closely followed the network structure, training strategy and training set as the orignal SRResNet. We also implemented <strong>subpixel convolution layer</strong> as <a href="https://arxiv.org/abs/1609.05158">Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network</a>, CVPR16. <a href="https://github.com/waihokwok">My collaborator</a> also shares contribution to this repository.

# Results in terms of PSNR and SSIM
</ul>
<table>
<thead>
<tr>
<th align="center">Dataset</th>
<th align="center">Custom</th>
<th align="center">CVPR17</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">Set5</td>
<td align="center">31.9678/0.9007</td>
<td align="center">32.05/0.9019</td>
</tr>
<tr>
<td align="center">Set14</td>
<td align="center">28.5809/0.7972</td>
<td align="center">28.49/0.8184</td>
</tr>
<tr>
<td align="center">BSD100</td>
<td align="center">27.5784/0.7538</td>
<td align="center">27.58/0.7620</td>
</tr>
<tr>
<td align="center">Urban100</td>
<td align="center">26.0479/0.7954</td>
<td align="center">-</td>
</tr>  
</tbody></table>

# Dependencies
pytorch 0.2 or above

python 3.5

# Training
CUDA_VISIBLE_DEVICES=0 python ./train.py --option ./options/train/SRResNet/SRResNet_x4.json

# Testing
CUDA_VISIBLE_DEVICES=0 python ./test.py --option ./options/test/SRResNet/SRResNet_x4.json

# Trained model
A trained model (16 residual blocks) is provided in </code>trained_models/latest_G.pth</code>.

# SRGAN-pyTorch
Coming soon
