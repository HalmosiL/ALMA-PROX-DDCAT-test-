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

    loader = torch.utils.data.DataLoader(   
        dataset=SemDataSplit(
            split=split,
            data_root=root,
            data_list=image_list_path,
            transform=val_transform,
            num_of_images=num_images
        ),
        batch_size=1,
        num_workers=1,
        pin_memory=True
    )

    return loader

device = "cuda:2"
model = load_model("/models/cityscapes/pspnet/ddcat/train_epoch_400.pth", device).eval()

loader = get_cityscapes_resized(
    root="./data/cityscapes/",
    size=None,
    split="val",
    num_images=1,
    batch_size=1
)

input, target = next(iter(loader))

input = input[0].to(device)
target = target[0].to(device)

pred = predict(
    model=model,
    image=input,
    target=target,
    device=device,
    attack=None
)

print(input)

pred = torch.argmax(pred, dim=0)

print(pred)
print(target)

print(pred.shape)
print(target.shape)

target = target[0]

print((target == pred).sum() / ((449*449) - (target==255).sum()))
