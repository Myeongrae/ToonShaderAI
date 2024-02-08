import torch
import sys
import numpy as np
from PIL import Image
from src.model import StyleEncoder
from src.fileIO import load_img_tensor

if __name__ == "__main__" :

    if len(sys.argv) < 2 :
        print("Error : image file path is needed")
        sys.exit()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load image
    file_path = sys.argv[1]
    
    img = Image.open(file_path)
    
    # crop and resize
    h, w = img.size
    if h > w : 
        img = img.crop(((h-w)//2, 0, (h+w)//2, w))
    elif w > h :
        img = img.crop((0, (w-h)//2, h, (h+w)//2))

    img = img.resize((128, 128), Image.LANCZOS)

    img = torch.tensor(np.array(img)/255., dtype=torch.float32, device=device)
    img = img[..., :3].permute(2, 0, 1).unsqueeze(0)

    # encode image
    net = StyleEncoder(weight_norm=True, pretrained="model/pretrained.safetensors").to(device)
    std, mean, c1, c2 = net(img)
    
    # print result
    print(f'Style Std : {std.squeeze().tolist()}')
    print(f'Style Mean : {mean.squeeze().tolist()}')
    print(f'Directional Light Color : {c1.squeeze().tolist()}')
    print(f'Ambient Light Color : {c2.squeeze().tolist()}')
