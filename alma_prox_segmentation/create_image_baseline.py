import transforms as transform
from dataset import SemData
from forward import predict
from model import load_model
from dataset import SemDataSplit

import numpy as np
import torch
import cv2

from adv_lib.attacks.segmentation import (
    alma_prox as alma_prox_seg
)

from functools import partial

def get_alma_prox(
    norm = float('inf'),
    num_steps = 500,
    alpha = 0.8,
    lr_reduction = 0.1,
    init_lr_distance = 16,
    scale_min = 0.05,
    scale_max = 1,
    rho_init = 0.01,
    mu_init=1e-4,
    scale_init = None,
    constraint_masking=True,
    mask_decay=True
):
    
    if norm == float('inf'):
        _log.warning('Divided init_lr_distance by 255')
        init_lr_distance = init_lr_distance / 255
        
    attack = partial(alma_prox_seg, norm=norm, num_steps=num_steps, α=alpha, lr_reduction=lr_reduction, ρ_init=rho_init,
                     μ_init=mu_init, init_lr_distance=init_lr_distance, scale_min=scale_min, scale_max=scale_max,
                     scale_init=scale_init, constraint_masking=constraint_masking, mask_decay=mask_decay)
    name = f'ALMA_prox_L{norm}_{num_steps}'
    return attack, name

def get_cityscapes_resized(root="", size=None, split="", num_images=None, batch_size=1):
    val_transform = transform.Compose(
        [transform.ToTensor(),]
    )

    image_list_path = root + "/" + split + ".txt"  
    
    dataset = SemDataSplit(
        split=split,
        data_root=root,
        data_list=image_list_path,
        transform=val_transform,
        num_of_images=num_images
    )

    return dataset

def model_prediction(input_, target_, model, device):
    logits_arr = []
    labels_arr = []

    for k in range(len(input_)):
        input = input_[k].to(device)
        target = target_[k].to(device)

        print(input.shape)

        input = input.reshape(1, *input.shape)
        target = target.reshape(1, *target.shape)

        pred = predict(
            model=model,
            image=input,
            target=target,
            device=device,
            attack=None
        )

        logits_arr.append(pred)
        labels_arr.append(target)

    logits = torch.zeros(19, 898, 1796)
    label = torch.zeros(1, 898, 1796)

    d = 0

    for x in range(2):
        for y in range(4):
            logits[:, x*449:(x+1)*449, y*449:(y+1)*449] = logits_arr[d]
            label[:, x*449:(x+1)*449, y*449:(y+1)*449] = labels_arr[d]
            d += 1

    pred = logits.reshape(19, 898, 1796)
    pred = torch.argmax(pred, dim=0)

    pred = pred.cpu()
    label = label.cpu()

    return pred, label

def test():
    device = "cuda:2"
    model = load_model("/models/cityscapes/pspnet/sat/train_epoch_400.pth", device).eval()

    attack, name = get_alma_prox()
    
    dataset_ = get_cityscapes_resized(
        root="./data/cityscapes/",
        size=None,
        split="val",
        num_images=1,
        batch_size=1
    )

    input_n, target_n = dataset_.__getitem__(1)
    pred, label = model_prediction(input_n, target_n, model, device)

    print(pred)
    print(label)

    print(pred.shape)
    print(label.shape)

    label = label[0]

    print((label == pred).sum() / ((898*1796) - (label==255).sum()))

if __name__ == "__main__":
    test()