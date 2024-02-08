#version 430

uniform mat4 matProj;
uniform mat4 matView;
uniform mat4 matGeo;

layout(location = 0) in vec3 inputVS0;
layout(location = 0) out vec4 outputVS0;
layout(location = 1) out vec3 outputVS1;
layout(location = 1) in vec3 inputVS1;
layout(location = 2) out vec3 outputVS2;
layout(location = 3) out vec2 outputVS3;
layout(location = 2) in vec2 inputVS2;

void main()
{
    gl_Position = ((matProj * matView) * matGeo) * vec4(inputVS0, 1.0);
    outputVS0 = matGeo * vec4(inputVS0, 1.0);
    outputVS1 = transpose(inverse(mat3(matGeo[0].xyz, matGeo[1].xyz, matGeo[2].xyz))) * normalize(inputVS1);
    outputVS2 = transpose(mat3(matView[0].xyz, matView[1].xyz, matView[2].xyz)) * (-matView[3].xyz);
    outputVS3 = vec2(inputVS2.x, -inputVS2.y);
}

