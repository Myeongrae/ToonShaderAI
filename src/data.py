import torch 
import numpy as np
import os
from torch.utils.data import Dataset
from .fileIO import load_img_tensor

class GBufferDataset(Dataset):
    """G buffer dataset."""

    def __init__(self, dir):
        """
        Arguments:
            dir (string): Dataset directory.
        """

        self.file_list = np.array([os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))], dtype=str)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])

        sample = {}
        for k, v in data.items() :
            sample[k] = torch.from_numpy(v).type(torch.float32)
            if sample[k].dim() == 3 :
                sample[k] = sample[k].permute((2, 0, 1))

        return sample

class ImageDataset(Dataset):
    """simple image dataset."""

    def __init__(self, img_dir):
        """
        Arguments:
            img_dir (string): Directory with all the input images.
        """

        self.img_list = np.array([os.path.join(img_dir, k) for k in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, k))], dtype=str)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        sample = {'image' : load_img_tensor(self.img_list[idx]).permute(2, 0, 1)}

        return sample
    
@torch.no_grad()
def phong_shading(data, random=True, device='cuda') :
    g_alpha = data['g_position'][:, 3:]
    b, _, h, w = g_alpha.size()

    g_pos = data['g_position'][:, :3]
    g_norm = torch.nn.functional.normalize(data['g_normal'], dim=1)
    g_view = data['g_view'] # normalized
    g_albedo = data['g_albedo']
    shadow = data['g_highlight'][:, 0:1]
    hlt1 = data['g_highlight'][:, 1:2]
    hlt2 = data['g_highlight'][:, 2:3]

    if random :
        g_lgt = torch.nn.functional.normalize(torch.randn((b, 3), device=g_pos.device))
        lgt_c = torch.rand((b, 3), device=g_pos.device) * 0.8 + 0.2
        shad_c = torch.rand((b, 3), device=g_pos.device) * 0.7
    else : 
        g_lgt = data['g_lgt']
        lgt_c = data['light_color']
        shad_c = data['light_shadow']

    g_lgt = g_lgt[..., None, None].repeat((1, 1, h, w))
    lgt_c = lgt_c[..., None, None].repeat((1, 1, h, w))
    shad_c = shad_c[..., None, None].repeat((1, 1, h, w))

    dot_ln = (g_lgt*g_norm).sum(dim=1, keepdim=True)
    dot_vn = (g_view*g_norm).sum(dim=1, keepdim=True)
    refl = g_lgt - 2*dot_ln*g_norm
    dot_rv = (g_view*refl).sum(dim=1, keepdim=True)

    diff = torch.where(dot_ln < 0., 0., dot_ln)
    spec = torch.where(dot_ln > 0., torch.where(dot_rv > 0., torch.pow(dot_rv, 8), 0.) , 0.)
    spec2 = torch.where(dot_ln > 0., torch.where(dot_rv > 0., torch.pow(dot_rv, 32), 0.) , 0.)

    diff = (diff + shadow) / 2
    spec = torch.where(hlt2 > 0, spec2, spec*shadow) + 0.2*hlt1

    kd = g_albedo
    ks = g_albedo*2
    ka = g_albedo

    lgt_d = (1-shad_c + lgt_c) * 0.5

    shaded = ka*shad_c + kd * diff* lgt_d + ks*spec*lgt_c
    shaded = torch.where(torch.abs(dot_vn) < 1e-4, 1.0, shaded)
    shaded = torch.clip(shaded, 0., 1.)

    g_buffer = torch.concat([g_alpha, g_albedo, torch.zeros_like(dot_vn), dot_ln, dot_rv, shadow, hlt1, hlt2, lgt_c, shad_c], dim=1).to(device)
    img = torch.concat([shaded, g_alpha], dim=1).to(device)

    return g_buffer, img

@torch.no_grad()
def prepare_data(data, device='cuda') :
    return phong_shading(data, True, device)
