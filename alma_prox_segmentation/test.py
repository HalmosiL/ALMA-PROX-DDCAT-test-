import transforms as transform
from dataset import SemData

import numpy as np
import torch
import cv2

def get_cityscapes_resized(root="", size=None, split="", num_images=1, batch_size=1):
    val_transform = transform.Compose(
        [transform.ToTensor()]
    )

    image_list_path = root + "/" + split + ".txt"

    loader = torch.utils.data.DataLoader(   
        dataset=SemData(
            split=split,
            data_root=root,
            data_list=image_list_path,
            transform=val_transform,
            num_of_images=num_images
        ),
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle = False
    )

    return loader

device = "cuda:2"
model = load_model("/models/cityscapes/pspnet/ddcat/train_epoch_400.pth", device)

loader = get_cityscapes_resized(
    root="./data/cityscapes/",
    size=None,
    split="val",
    num_images=1,
    batch_size=1
)

input, target, label_path = next(iter(loader))
