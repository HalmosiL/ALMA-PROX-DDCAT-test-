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

input_, target_ = dataset_.__getitem__(1)

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

pred = logits.reshape(19, 898, 1796).to(device)
pred = torch.argmax(pred, dim=0)

print(pred)
print(label)

print(pred.shape)
print(label.shape)

label = label[0]

print((label == pred).sum() / ((449*449) - (label==255).sum()))
