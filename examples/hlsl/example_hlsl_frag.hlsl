// =====================================================================================================================
// 	binidng variables
// =====================================================================================================================

struct PSInput
{
	float4 Position : SV_POSITION;
	float3 WorldPos : TEXCOORD0;
	float3 Normal : TEXCOORD1;
	float3 ViewPos : TEXCOORD2;
	float2 TexCoord : TEXCOORD3;
};

cbuffer cbLight : register(b0) 
{
	float3 LightPos;
	float3 LightColor;
	float3 AmbientColor;
}

Texture2D<float4> texAlbedo : register(t0);
Texture2D<float4> texMask : register(t1);
sampler samp : register(s0);

static const float3 target_style_std = float3(1.0, 1.0, 1.0);
static const float3 target_style_mean = float3(0.0, 0.0, 0.0);

// =====================================================================================================================
// 	ToonShaderAI parameters
// =====================================================================================================================

static const float3x3 conv_lgt_0_weight_0 = float3x3(0.00000000, -0.15168414, 0.29677373, 0.00000000, 0.08816471, 0.07736473, -0.00803071, 0.01686509, -0.00339539);
static const float3x3 conv_lgt_0_weight_1 = float3x3(-0.42641106, -0.66182077, 0.03244254, -0.95069349, -0.61783767, -0.16375503, 0.00003925, 0.00081890, -0.00006972);
static const float3 conv_lgt_0_bias = float3(0.8055619597434998, 0.5941834449768066, -0.2303735911846161);
static const float3x3 conv_lgt_1_weight = float3x3(1.63268578, 0.92085719, -0.02657001, -0.00025563, 0.00388690, -0.00553063, -0.00016434, 0.00006069, 0.00019203);
static const float3 conv_lgt_1_bias = float3(0.17507442831993103, -0.06554356962442398, -0.022941825911402702);
static const float3x3 conv_lgt_2_weight = float3x3(0.86393136, -0.00077882, -0.00001402, 1.40284622, 0.00035306, 0.00086880, -0.45821241, 0.00454504, -0.00017212);
static const float3 conv_lgt_2_bias = float3(0.6411716938018799, 0.3183196187019348, -0.07949614524841309);
static const float3x3 conv_lgt_3_weight = float3x3(0.17398299, 0.57659459, 0.12003411, 0.44896966, 0.94531846, 0.00023569, 0.27768105, 0.70338953, -0.26372793);
static const float3 conv_lgt_3_bias = float3(0.36807695031166077, -0.0722191259264946, 0.41702306270599365);
static const float3x3 conv_shad_0_weight_0 = float3x3(0.15342245, 0.40318379, 0.40889031, 0.44183522, 0.55213690, 0.49199867, 0.00307558, -0.00027415, 0.00176936);
static const float3x3 conv_shad_0_weight_1 = float3x3(0.68825370, -0.90781307, 0.01890765, -0.68958551, -0.41224211, -0.71897894, -0.00147964, 0.00264597, 0.00197219);
static const float3 conv_shad_0_bias = float3(0.16848771274089813, 0.28855130076408386, -0.579033613204956);
static const float3x3 conv_shad_1_weight = float3x3(-0.00000232, -0.00000212, -0.00000002, -0.00019028, 0.00028210, -0.00000071, 0.01873682, 1.53645313, -0.00179048);
static const float3 conv_shad_1_bias = float3(-0.14506810903549194, -0.29552969336509705, -0.03612350672483444);
static const float3x3 conv_shad_2_weight = float3x3(0.00063310, 0.00102125, -1.22484183, -0.00044743, 0.00275843, 0.88119119, -0.00023480, 0.09293491, 0.45697182);
static const float3 conv_shad_2_bias = float3(0.3446977436542511, -0.0002613725373521447, -0.16102707386016846);
static const float3 conv_shad_3_weight = float3(1.108639121055603, -0.794776439666748, -0.4122093617916107);
static const float conv_shad_3_bias = 0.6914172172546387;


// =====================================================================================================================
// 	helper functions
// =====================================================================================================================

float3 leakyReLU(float3 v) 
{
	float x = v.x > 0 ? v.x : 0.01*v.x;
	float y = v.y > 0 ? v.y : 0.01*v.y;
	float z = v.z > 0 ? v.z : 0.01*v.z;
	return float3(x, y, z);
};

float3x3 leakyReLU(float3x3 m) 
{
	return float3x3(leakyReLU(m[0]), leakyReLU(m[1]), leakyReLU(m[2]));
}

float3 sigmoid(float3 v) 
{
	return 1.0 / (1.0 + exp(-v));
}

float3 toonShaderAI(float3 light, float3 normal, float3 view, 
					float occlusion, float emission, float shininess, 
					float3 albedo, float3 light_color, float3 ambient_color,
					float3 style_std, float3 style_mean) {

										
	float3 L = normalize(light);
	float3 N = normalize(normal);
	float3 V = normalize(view);
	float3 R = reflect(L, N);
	
	float3 lgt = float3(0, dot(L, N), dot(R, V));
	float3 mask = float3(occlusion, emission, shininess);
	
	// lighting module
	float3 feature = leakyReLU(mul(conv_lgt_0_weight_0, lgt) + mul(conv_lgt_0_weight_1, mask) + conv_lgt_0_bias);
    feature = feature * style_std + style_mean;
    
    float3 feature_conv = leakyReLU(mul(conv_lgt_1_weight, feature) + conv_lgt_1_bias);
    feature = mul(conv_lgt_2_weight, feature_conv) + conv_lgt_2_bias + feature;
    feature = feature * style_std + style_mean;
    
    feature = sigmoid(mul(conv_lgt_3_weight, feature) + conv_lgt_3_bias);
    
	// coloring module
    float3x3 features = transpose(float3x3(feature, feature, feature));
    float3x3 colors = float3x3(albedo.rgb, light_color, ambient_color);
	
	colors = leakyReLU(mul(conv_shad_0_weight_0, features) + mul(conv_shad_0_weight_1, colors) + transpose(float3x3(conv_shad_0_bias, conv_shad_0_bias, conv_shad_0_bias)));
    float3x3 colors_conv = leakyReLU(mul(conv_shad_1_weight, colors) + transpose(float3x3(conv_shad_1_bias, conv_shad_1_bias, conv_shad_1_bias)));
    colors = mul(conv_shad_2_weight, colors_conv) + transpose(float3x3(conv_shad_2_bias, conv_shad_2_bias, conv_shad_2_bias)) + colors;
    float3 color = sigmoid(mul(conv_shad_3_weight, colors) + float3(conv_shad_3_bias, conv_shad_3_bias, conv_shad_3_bias));
	
	return color;
}

// =====================================================================================================================
// 	main function
// =====================================================================================================================

float4 main(PSInput pin) : SV_TARGET
{
	float4 albedo = texAlbedo.Sample(samp, pin.TexCoord);
	float3 mask = texMask.Sample(samp, pin.TexCoord).rgb;	
	
	float3 color = toonShaderAI(
		LightPos,
		pin.Normal,
		pin.ViewPos - pin.WorldPos,
		mask.r,
		mask.g,
		mask.b,
		albedo.rgb,
		LightColor,
		AmbientColor,
		target_style_std,
		target_style_mean
	);
	
	return float4(color, albedo.a);
}