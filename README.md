# SRGAN-PyTorch

This repository contains the unoffical pyTorch implementation of <strong>SRGAN</strong> and also <strong>SRResNet</strong> in the paper <a href="https://arxiv.org/abs/1609.04802">Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network</a>, CVPR17. 

We closely followed the network structure, training strategy and training set as the orignal SRGAN and SRResNet. We also implemented <strong>subpixel convolution layer</strong> as <a href="https://arxiv.org/abs/1609.05158">Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network</a>, CVPR16. <a href="https://github.com/waihokwok">My collaborator</a> also shares contribution to this repository.

# License and Citation
All code and other materials (including but not limited to the tables) are provided for research purposes only and without any warranty. Any commercial use requires our consent. If our work helps your research or you use any parts of the code in your research, please acknowledge it appropriately:

<pre><code>@InProceedings{ledigsrgan17,    
 author = {Christian Ledig and Lucas Theis and Ferenc Husz&aacuter and Jose Caballero and Andrew Cunningham and Alejandro Acosta and Andrew Aitken and Alykhan Tejani and Johannes Totz and Zehan Wang and Wenzhe Shi},    
 title  = {Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network},    
 booktitle  = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},    
 pages = {4681--4690},  
 year = {2017}}
 </code></pre>

<pre><code>@misc{SRGAN-pyTorch,
  author = {Tak-Wai Hui and Wai-Ho Kwok},
  title = {SRGAN-PyTorch: A PyTorch Implementation of Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/twhui/SRGAN-PyTorch}}
}</code></pre>

# Results of SRGAN in terms of PSNR and SSIM
</ul>
<table>
<thead>
<tr>
<th align="center">Dataset</th>
<th align="center">Our</th>
<th align="center">CVPR17</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">Set5</td>
<td align="center">29.4490 / 0.8542</td>
<td align="center">29.40 / 0.8472</td>
</tr>
<tr>
<td align="center">Set14</td>
<td align="center">26.0677 / 0.7153</td>
<td align="center">26.02 / 0.7397</td>
</tr>
<tr>
<td align="center">BSD100</td>
<td align="center">24.8665 / 0.6594</td>
<td align="center">25.16 / 0.6688</td>
</tr>
<tr>
<td align="center">Urban100</td>
<td align="center">23.9434 / 0.7277</td>
<td align="center">-</td>
</tr>  
</tbody></table>

# Results of SRResNet in terms of PSNR and SSIM
</ul>
<table>
<thead>
<tr>
<th align="center">Dataset</th>
<th align="center">Our</th>
<th align="center">CVPR17</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">Set5</td>
<td align="center">31.9678 / 0.9007</td>
<td align="center">32.05 / 0.9019</td>
</tr>
<tr>
<td align="center">Set14</td>
<td align="center">28.5809 / 0.7972</td>
<td align="center">28.49 / 0.8184</td>
</tr>
<tr>
<td align="center">BSD100</td>
<td align="center">27.5784 / 0.7538</td>
<td align="center">27.58 / 0.7620</td>
</tr>
<tr>
<td align="center">Urban100</td>
<td align="center">26.0479 / 0.7954</td>
<td align="center">-</td>
</tr>  
</tbody></table>

# Dependencies
pytorch 0.3+, python 3.5, python-box, scikit-image, numpy

# Training set
We used a subset of Imagenet dataset ILSVRC2016_CLS-LOC.tar.gz for training our models. The subset can be found in <code>/subset.txt</code> 

# Training
<pre><code>CUDA_VISIBLE_DEVICES=0 python ./train.py --option ./options/train/SRResNet/SRResNet_x4.json</code></pre>
<pre><code>CUDA_VISIBLE_DEVICES=0 python ./train.py --option ./options/train/SRGAN/SRGAN_x4.json</code></pre>

# Testing
<pre><code>CUDA_VISIBLE_DEVICES=0 python ./test.py --option ./options/test/SRResNet/SRResNet_x4.json</code></pre>
<pre><code>CUDA_VISIBLE_DEVICES=0 python ./test.py --option ./options/test/SRGAN/SRGAN_x4.json</code></pre>

The upsampled images will be generated in <code>/home/twhui/Projects/SRGAN/results/MODEL_NAME/test_images</code>. 
A text file that contains PSNR and SSIM results will be generated in <code>/home/twhui/Projects/SRGAN/results/MODEL_NAME/log</code>. MODEL_NAME = SRResNet_x4 or SRGAN_x4.

# Trained models
The trained models (16 residual blocks) of <a href="https://drive.google.com/file/d/1BRRfis9HEWccJJsIEgPg0Ou3zV-3gVq5/view">SRResNet</a> and <a href="https://drive.google.com/file/d/1vAtPLGbdyt--SZQxUl0YKPRQgu6-kR6v/view">SRGAN</a> are available.
