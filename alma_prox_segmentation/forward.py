import os
import time
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn

def get_args():
    return {
        "classes": 19,
        "crop_h": 449,
        "crop_w": 449,
        "scales": [1.0],
        "base_size": 1024
    }

mean = [0.485, 0.456, 0.406]

cv2.ocl.setUseOpenCL(False)

args = get_args()

def net_process(model, image, target, device, attack=None):
    input = image.float()
    target = target.long()

    input = input.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    if attack is not None:
        adver_input = attack(
            model=model,
            inputs=input,
            labels=target,
            targeted=False
        )

        adver_input.clamp(min=0, max=1)

        with torch.no_grad():
            output = model(adver_input)
    else:
        with torch.no_grad():
            output = model(input)

    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape

    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)

    output = F.softmax(output, dim=1)
    output = output[0]

    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)

    return output
