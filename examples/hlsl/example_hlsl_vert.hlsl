struct VSInput
{
    float3 Position;
    float3 Normal;
    float2 TexCoord;
};

struct VSOutput
{
    float4 Position;
    float3 WorldPos;
    float3 Normal;
    float3 ViewPos;
    float2 TexCoord;
};

static const VSOutput _24 = { 0.0f.xxxx, 0.0f.xxx, 0.0f.xxx, 0.0f.xxx, 0.0f.xx };

cbuffer cbPerFrame : register(b0)
{
    column_major float4x4 _30_matProj : packoffset(c0);
    column_major float4x4 _30_matView : packoffset(c4);
    column_major float4x4 _30_matGeo : packoffset(c8);
};

uniform float4 gl_HalfPixel;

static float4 gl_Position;
static float3 vin_Position;
static float3 vin_Normal;
static float2 vin_TexCoord;
static float3 _entryPointOutput_WorldPos;
static float3 _entryPointOutput_Normal;
static float3 _entryPointOutput_ViewPos;
static float2 _entryPointOutput_TexCoord;

struct SPIRV_Cross_Input
{
    float3 vin_Position : TEXCOORD0;
    float3 vin_Normal : TEXCOORD1;
    float2 vin_TexCoord : TEXCOORD2;
};

struct SPIRV_Cross_Output
{
    float3 _entryPointOutput_WorldPos : TEXCOORD0;
    float3 _entryPointOutput_Normal : TEXCOORD1;
    float3 _entryPointOutput_ViewPos : TEXCOORD2;
    float2 _entryPointOutput_TexCoord : TEXCOORD3;
    float4 gl_Position : POSITION;
};

VSOutput _main(VSInput vin)
{
    VSOutput vout = _24;
    vout.Position = mul(mul(mul(float4(vin.Position, 1.0f), _30_matGeo), _30_matView), _30_matProj);
    vout.WorldPos = float3(mul(vin.Position, float3x4(_30_matGeo[0], _30_matGeo[1], _30_matGeo[2])).xyz);
    vout.Normal = mul(vin.Normal, float3x3(_30_matGeo[0].xyz, _30_matGeo[1].xyz, _30_matGeo[2].xyz));
    vout.ViewPos = -mul(float3(_30_matView[3].xyz), transpose(float3x3(_30_matView[0].xyz, _30_matView[1].xyz, _30_matView[2].xyz)));
    vout.TexCoord = float2(vin.TexCoord.x, -vin.TexCoord.y);
    return vout;
}

void vert_main()
{
    VSInput vin;
    vin.Position = vin_Position;
    vin.Normal = vin_Normal;
    vin.TexCoord = vin_TexCoord;
    VSInput param = vin;
    VSOutput flattenTemp = _main(param);
    gl_Position = flattenTemp.Position;
    _entryPointOutput_WorldPos = flattenTemp.WorldPos;
    _entryPointOutput_Normal = flattenTemp.Normal;
    _entryPointOutput_ViewPos = flattenTemp.ViewPos;
    _entryPointOutput_TexCoord = flattenTemp.TexCoord;
    gl_Position.x = gl_Position.x - gl_HalfPixel.x * gl_Position.w;
    gl_Position.y = gl_Position.y + gl_HalfPixel.y * gl_Position.w;
}

SPIRV_Cross_Output main(SPIRV_Cross_Input stage_input)
{
    vin_Position = stage_input.vin_Position;
    vin_Normal = stage_input.vin_Normal;
    vin_TexCoord = stage_input.vin_TexCoord;
    vert_main();
    SPIRV_Cross_Output stage_output;
    stage_output.gl_Position = gl_Position;
    stage_output._entryPointOutput_WorldPos = _entryPointOutput_WorldPos;
    stage_output._entryPointOutput_Normal = _entryPointOutput_Normal;
    stage_output._entryPointOutput_ViewPos = _entryPointOutput_ViewPos;
    stage_output._entryPointOutput_TexCoord = _entryPointOutput_TexCoord;
    return stage_output;
}
