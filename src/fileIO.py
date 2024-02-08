import torch
import numpy as np
from safetensors.torch import save_model
from PIL import Image
from .model import ToonShaderStyle

def load_img_tensor(filePath) :
    '''
        load image as torch tensor
    '''
    img = np.array(Image.open(filePath))
    return torch.tensor(img/255., dtype=torch.float32)

@torch.no_grad()
def weight_to_str(weight : torch.Tensor, var_name : str, dtype='glsl') :
    dtype_set = {
        'glsl' : {
            'major' : 'column',
            'float' : 'float',
            'vec3' : 'vec3',
            'mat3' : 'mat3',
            'float_declare' : 'const float',
            'vec3_declare' : 'const vec3',
            'mat3_declare' : 'const mat3'
        },

        'hlsl' : {
            'major' : 'row',
            'float' : 'float',
            'vec3' : 'float3',
            'mat3' : 'float3x3',
            'float_declare' : 'static const float',
            'vec3_declare' : 'static const float3',
            'mat3_declare' : 'static const float3x3',
        },

        'webgl2' : {
            'major' : 'column',
            'float' : 'float',
            'vec3' : 'vec3',
            'mat3' : 'mat3',
            'float_declare' : 'const float',
            'vec3_declare' : 'const vec3',
            'mat3_declare' : 'const mat3'
        },
    }

    assert dtype_set.get(dtype) is not None
    dset = dtype_set[dtype]

    if weight.dim() == 1 :
        if weight.shape[0] == 1 :
            return f"{dset['float_declare']} {var_name} = {weight.item()};\n"
        elif weight.shape[0] == 3 :
            return f"{dset['vec3_declare']} {var_name} = {dset['vec3']}({weight[0]}, {weight[1]}, {weight[2]});\n"
        else :
            assert False

    elif weight.dim() == 2 :
        assert weight.shape[0] == 3 and weight.shape[1] % 3 == 0

        out_str = ''
        for n in range(weight.shape[1] // 3) :
            if weight.shape[1] == 3 :
                head = f"{dset['mat3_declare']} {var_name} = {dset['mat3']}("
            else :
                head = f"{dset['mat3_declare']} {var_name}_{n} = {dset['mat3']}("
            # row major writing
            if dset['major'] == 'row' :
                for i in range(3) :
                    for j in range(3*n, 3*(n+1)):
                        head += f'{weight[i, j]:.8f}, '
            # column major writing
            elif dset['major'] == 'column' :
                for i in range(3*n, 3*(n+1)) :
                    for j in range(3) :
                        head += f'{weight[j, i]:.8f}, '
            out_str += head[:-2] + ');\n'
        return out_str

    else :
        return ''


@torch.no_grad()
def export_weights_lines(net : ToonShaderStyle, dtype='glsl', weight_norm=False) :
    # 
    # since [ torch.nn.utils.weight_norm ] is deprecated, weight_norm is always False
    #
    
    module = {'conv_lgt' : net.convLgt, 'conv_shad' : net.convShad}
    out_str = ''
    for key, layer in module.items() :
        layers = {layer[0].conv[0] : weight_norm, 
                  layer[1].conv[0] : weight_norm,
                  layer[1].conv[2] : weight_norm,
                  layer[2] : False}
        i = 0
        for this_layer, wn in layers.items() : 
            if wn :
                weight_g = this_layer.weight_g.squeeze()
                weight_v = this_layer.weight_v.squeeze()
                weight = weight_g * weight_v / (torch.norm(weight_v) + 1e-12)
                out_str += weight_to_str(weight, f'{key}_{i}_weight', dtype)
            else : 
                out_str += weight_to_str(this_layer.weight.squeeze(), f'{key}_{i}_weight', dtype)
            out_str += weight_to_str(this_layer.bias, f'{key}_{i}_bias', dtype)
            i += 1

    return out_str

@torch.no_grad()
def export_weights_shader(file_out:str, pretrained:str, weight_norm:bool=False, format:str='glsl') :
    net = ToonShaderStyle.load_pretrained(pretrained, weight_norm)

    template_head = f"data/templates/{format}_head.txt"
    template_tail = f"data/templates/{format}_tail.txt"

    with open(file_out, 'w') as f :
        with open(template_head, 'r') as head:
            f.writelines(head.readlines())

        f.write('\n' + export_weights_lines(net, format) + '\n')

        with open(template_tail, 'r') as tail:
            f.writelines(tail.readlines())

@torch.no_grad()
def safetensorFromCkpt(net, ckpt, file_out, key='net') : 
    net.load_state_dict(torch.load(ckpt)[key])
    save_model(net, file_out)