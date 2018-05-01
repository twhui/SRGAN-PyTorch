# SRResNet-pyTorch

This repository contains the pyTorch implementation of <strong>SRResNet</strong> in the paper <a href="https://arxiv.org/abs/1609.04802"><strong>Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network </strong></a>.

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
