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

cv2.ocl.setUseOpenCL(False)

def predict(model, image, target, device, attack=None):
    input = image.float()
    target = target.long()

    input = input.to(device)
    target = target.to(device)

    print("FORWARD:", input.shape)
    print("FORWARD:", target.shape)

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

    if attack is not None:
        return output, adver_input
    
    return output
