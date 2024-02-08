import torch

from src.model import StyleEncoder
from src.fileIO import export_weights_shader

if __name__ == "__main__" :

    net = StyleEncoder(weight_norm=True, pretrained="model/pretrained.safetensors")

    dummy_input = torch.randn(1, 3, 128, 128)
    input_names = ["input_0"]
    output_names = ["style_std", "style_mean", "light_color", "ambient_color"]
    torch.onnx.export(net, dummy_input, "model/model.onnx", verbose=False, input_names=input_names, output_names=output_names)

    print("exporting onnx is finished!")

    export_weights_shader('examples/glsl/example_glsl_frag.glsl', 'model/pretrained.safetensors', True, 'glsl')
    export_weights_shader('examples/hlsl/example_hlsl_frag.hlsl', 'model/pretrained.safetensors', True, 'hlsl')
    export_weights_shader('examples/webgl2/example_webgl2.js', 'model/pretrained.safetensors', True, 'webgl2')
    
    print("writing example shaders is finished!")
