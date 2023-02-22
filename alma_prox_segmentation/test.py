import transforms as transform
from dataset import SemData
from forward import predict
from model import load_model
from dataset import SemDataSplit

import numpy as np
import torch
import cv2

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

device = "cuda:2"
model = load_model("/models/cityscapes/pspnet/ddcat/train_epoch_400.pth", device).eval()

dataset_ = get_cityscapes_resized(
    root="./data/cityscapes/",
    size=None,
    split="val",
    num_images=1,
    batch_size=1
)

input, target = dataset_.__getitem__(1)

input = input[0].to(device)
target = target[0].to(device)

print("INPUT-SHAPE:", input.shape)
print("TARGET-SHAPE:", target.shape)

pred = predict(
    model=model,
    image=input,
    target=target,
    device=device,
    attack=None
)

pred = torch.argmax(pred, dim=0)

print(pred)
print(target)

print(pred.shape)
print(target.shape)

target = target[0]

print((target == pred).sum() / ((449*449) - (target==255).sum()))
