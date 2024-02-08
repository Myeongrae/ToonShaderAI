import torch
from torchvision import models
from collections import OrderedDict
from safetensors.torch import load_model

# ================================================================================================================================
# 
#   1x1 Convolution Layer with Leaky ReLU activation layer
#
# ================================================================================================================================

class Conv1x1(torch.nn.Module) :
    def __init__(self, in_channels, out_channels, negative_slope=0.01, weight_norm=False) -> None:
        super(Conv1x1, self).__init__()
        # 
        # deprecated : torch.nn.utils.weight_norm 
        # wn = lambda x : torch.nn.utils.weight_norm(x, dim=None) if weight_norm else x
        #
        wn = lambda x : torch.nn.utils.parametrizations.weight_norm(x, dim=None) if weight_norm else x
        self.conv = torch.nn.Sequential(OrderedDict([
            ('conv', wn(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))),
            ('ReLU', torch.nn.LeakyReLU(negative_slope=negative_slope)),
        ]))

    def forward(self, x) :
        return self.conv(x)

# ================================================================================================================================
# 
#   1x1 Convolution Residual Block
#
# ================================================================================================================================
    
class Conv1x1Residual(torch.nn.Module) :
    def __init__(self, channels, negative_slope=0.01, weight_norm=False) -> None:
        super(Conv1x1Residual, self).__init__()
        
        # 
        # deprecated : torch.nn.utils.weight_norm 
        # wn = lambda x : torch.nn.utils.weight_norm(x, dim=None) if weight_norm else x
        #
        wn = lambda x : torch.nn.utils.parametrizations.weight_norm(x, dim=None) if weight_norm else x
        self.conv = torch.nn.Sequential(OrderedDict([
            ('conv1', wn(torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0))),
            ('ReLU', torch.nn.LeakyReLU(negative_slope=negative_slope)),
            ('conv2', wn(torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0))),
        ]))

    def forward(self, x) :
        return self.conv(x) + x
    

# ================================================================================================================================
# 
#   ToonShaderStyle
#
# ================================================================================================================================
class ToonShaderStyle(torch.nn.Module) :
    def load_pretrained(pretrained:str, weight_norm:bool=False) : 
        net = ToonShaderStyle(weight_norm=weight_norm)
        if pretrained[-11:] == 'safetensors' :
            load_model(net, pretrained)
        else : 
            net.load_state_dict(torch.load(pretrained)['net']) 
        return net

    def __init__(self, negative_slope= 0.01, weight_norm=False) -> None:
        super(ToonShaderStyle, self).__init__()
        
        self.convLgt = torch.nn.Sequential(OrderedDict([
            ('conv1', Conv1x1(6, 3, negative_slope, weight_norm)),
            ('conv2', Conv1x1Residual(3, negative_slope, weight_norm)),
            ('conv_vanila', torch.nn.Conv2d(3, 3, 1, 1, 0)),
        ]))

        self.convShad = torch.nn.Sequential(OrderedDict([
            ('conv1', Conv1x1(6, 3, negative_slope, weight_norm)),
            ('conv2', Conv1x1Residual(3, negative_slope, weight_norm)),
            ('conv_vanila', torch.nn.Conv2d(3, 1, 1, 1, 0)),
        ]))

        self.feature = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(1920, 512)),
            ('relu1', torch.nn.LeakyReLU(0.01, True)),
            ('fc2', torch.nn.Linear(512, 128)),
            ('relu2', torch.nn.LeakyReLU(0.01, True)),
            ('fc3', torch.nn.Linear(128, 32)),
            ('relu3', torch.nn.LeakyReLU(0.01, True)),
            ('fc4', torch.nn.Linear(32, 12)),
        ]))

    def lighting(self, g_buffer, feature) :
        x = g_buffer[:, 4:-6]

        for i in range(0, len(self.convLgt)-1) :
            x = self.convLgt[i](x)
            x = x * torch.exp(0.5*feature[:, :3, None, None]) + feature[:, 3:, None, None]
        z = self.convLgt[-1](x)
        return z

    def shading(self, z, g_buffer) :
        c_albedo = g_buffer[:, 1:4]
        c_light = g_buffer[:, -6:]
        z = torch.sigmoid(z).unsqueeze(1).repeat((1, 3, 1, 1, 1))
        z = torch.concat([z, c_albedo[:, :, None, ...], c_light[:, :3, None, ...], c_light[:, 3:, None, ...]], dim=2).reshape((-1, 6) + z.shape[-2:])
        z = self.convShad(z).reshape((-1, 3, 1) + z.shape[-2:]).squeeze(2)
        return z

    def forward(self, g_buffer, feature_vector = None) :
        self.reg = 0
        if feature_vector is None :
            feature = torch.zeros((g_buffer.shape[0], 6), device=g_buffer.device)
        else :
            feature = self.feature(feature_vector)
            g_buffer = torch.concat([g_buffer[:, :-6], torch.sigmoid(feature[:, -6:, None, None]).repeat((1, 1, ) + g_buffer.shape[-2:])], dim=1)
            feature = feature[:, :6]

        z = self.lighting(g_buffer, feature)
        color = self.shading(z, g_buffer)
        return torch.concat([torch.sigmoid(color), g_buffer[:, :1]], dim=1)

# ================================================================================================================================
# 
#   VGG-19 Feature Extractor
#
# ================================================================================================================================
class VGGFeatureExtractor(torch.nn.Module) :
    def __init__(self, layers=None) -> None:
        super(VGGFeatureExtractor, self).__init__()
        cnn = models.vgg19(weights='IMAGENET1K_V1').features.eval()
        for param in cnn.parameters():
            param.requires_grad = False

        self.model = torch.nn.Sequential()

        i = 0
        for layer in cnn.children():
            if isinstance(layer, torch.nn.ReLU) :
                self.model.add_module(str(i), torch.nn.ReLU(inplace=False))
            else :
                self.model.add_module(str(i), layer)
            
            i += 1
            if i > 29 : 
                break 
        
        self.mean = torch.nn.parameter.Parameter(torch.tensor([[[[0.485]], [[0.456]], [[0.406]]]]), requires_grad=False)
        self.std = torch.nn.parameter.Parameter(torch.tensor([[[[0.229]], [[0.224]], [[0.225]]]]), requires_grad=False)

        if layers is None : 
            self.layers = {
                '1' : 'relu1_1',
                '6' : 'relu2_1',
                '11' : 'relu3_1',
                '20' : 'relu4_1', 
            }
        else :
            self.layers = layers

    def forward(self, x) :
        x = (x - self.mean) / self.std
        features = {}
        for name, layer in self.model._modules.items() :
            x = layer(x)
            if name in self.layers :
                features[self.layers[name]] = x

        return features

# ================================================================================================================================
# 
#   Style Encoder 
#
# ================================================================================================================================    
class StyleEncoder(torch.nn.Module) :
    def __init__(self, weight_norm=False, pretrained=None) -> None:
        super(StyleEncoder, self).__init__()
        self.vgg = VGGFeatureExtractor()
        toonShader = ToonShaderStyle.load_pretrained(pretrained, weight_norm)

        self.decoder = toonShader.feature

    def forward(self, x) :
        features = self.vgg(x)
        mean = []
        std = []
        for feature in features.values() :
            mean.append(torch.mean(feature, dim=(-1, -2)))
            std.append(torch.std(feature, dim=(-1, -2)))

        style = self.decoder(torch.concat(mean+std, dim=1))
        
        #
        # explicitly define slicing dimensions to avoid onnx exporter automatically set dimension as UINT64_MAX 
        # ([:, -3:] => [:, -3:12]) 
        #
        return torch.exp(0.5*style[:, :3]), style[:, 3:6], torch.sigmoid(style[:, -6:-3]), torch.sigmoid(style[:, -3:12])