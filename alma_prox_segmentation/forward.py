import torch
import torch.nn.functional as F

def predict(model, image, target, device, attack=None):
    input = image.float()
    target = target.long()

    input = input.to(device)
    target = target.to(device)

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
            output_normal = model(input)
        
        print(output.shape)
        
        output_ = F.softmax(output, dim=1)
        output_ = output_[0]
        
        print(output_.shape)
        print(target.shape)
        
        output_ = output_.argmax(0)
        
        print(output_.shape)
        
        output_normal = F.softmax(output_normal, dim=0)
        output_normal = output_normal[0]
        
        print(output_.shape)
      
        output_normal = output_normal.argmax(0)
        
        print((target == output_).sum() / (449*449))
        print((target == output_normal).sum() / (449*449))
        
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
