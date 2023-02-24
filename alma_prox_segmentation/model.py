from network import PSPNet_DDCAT, PSPNet

import torch.nn as nn
import torch

def load_model(path, device, mode="ddcat"):
    print(device, " | ", mode)

    if mode == "ddcat":
        model = PSPNet_DDCAT(
            layers=50,
            classes=19,
            zoom_factor=8,
            pretrained=False 
        )

    else:
        model = PSPNet(
            layers=50,
            bins=(1, 2, 3, 6),
            dropout=0.1,
            classes=19,
            zoom_factor=8,
            use_ppm=True,
            criterion=nn.CrossEntropyLoss(ignore_index=255),
            BatchNorm=nn.BatchNorm2d,
            pretrained=False
        )

    model = model.to(device)

    model = torch.nn.DataParallel(model, device_ids=[device]).to(device)
    model.load_state_dict(torch.load(path, map_location=device)["state_dict"], strict=False)

    return model
