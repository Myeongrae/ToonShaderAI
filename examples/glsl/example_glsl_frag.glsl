#version 450

// =====================================================================================================================
// 	binding variables
// =====================================================================================================================

layout(binding = 0) uniform sampler2D samplerAlbedo;
layout(binding = 1) uniform sampler2D samplerMasking;

struct Light {
	vec4 lightPos;
	vec4 lightColor;
	vec4 ambientColor;
	vec4 padding;
};
uniform Light light;

layout(location = 0) in vec4 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inViewPos;
layout(location = 3) in vec2 inTexCoords;

out vec4 outColor;

const vec3 target_style_std = vec3(1.0, 1.0, 1.0);
const vec3 target_style_mean = vec3(0.0, 0.0, 0.0);

// =====================================================================================================================
// 	ToonShaderAI parameters
// =====================================================================================================================

const mat3 conv_lgt_0_weight_0 = mat3(0.00000000, 0.00000000, -0.00803071, -0.15168414, 0.08816471, 0.01686509, 0.29677373, 0.07736473, -0.00339539);
const mat3 conv_lgt_0_weight_1 = mat3(-0.42641106, -0.95069349, 0.00003925, -0.66182077, -0.61783767, 0.00081890, 0.03244254, -0.16375503, -0.00006972);
const vec3 conv_lgt_0_bias = vec3(0.8055619597434998, 0.5941834449768066, -0.2303735911846161);
const mat3 conv_lgt_1_weight = mat3(1.63268578, -0.00025563, -0.00016434, 0.92085719, 0.00388690, 0.00006069, -0.02657001, -0.00553063, 0.00019203);
const vec3 conv_lgt_1_bias = vec3(0.17507442831993103, -0.06554356962442398, -0.022941825911402702);
const mat3 conv_lgt_2_weight = mat3(0.86393136, 1.40284622, -0.45821241, -0.00077882, 0.00035306, 0.00454504, -0.00001402, 0.00086880, -0.00017212);
const vec3 conv_lgt_2_bias = vec3(0.6411716938018799, 0.3183196187019348, -0.07949614524841309);
const mat3 conv_lgt_3_weight = mat3(0.17398299, 0.44896966, 0.27768105, 0.57659459, 0.94531846, 0.70338953, 0.12003411, 0.00023569, -0.26372793);
const vec3 conv_lgt_3_bias = vec3(0.36807695031166077, -0.0722191259264946, 0.41702306270599365);
const mat3 conv_shad_0_weight_0 = mat3(0.15342245, 0.44183522, 0.00307558, 0.40318379, 0.55213690, -0.00027415, 0.40889031, 0.49199867, 0.00176936);
const mat3 conv_shad_0_weight_1 = mat3(0.68825370, -0.68958551, -0.00147964, -0.90781307, -0.41224211, 0.00264597, 0.01890765, -0.71897894, 0.00197219);
const vec3 conv_shad_0_bias = vec3(0.16848771274089813, 0.28855130076408386, -0.579033613204956);
const mat3 conv_shad_1_weight = mat3(-0.00000232, -0.00019028, 0.01873682, -0.00000212, 0.00028210, 1.53645313, -0.00000002, -0.00000071, -0.00179048);
const vec3 conv_shad_1_bias = vec3(-0.14506810903549194, -0.29552969336509705, -0.03612350672483444);
const mat3 conv_shad_2_weight = mat3(0.00063310, -0.00044743, -0.00023480, 0.00102125, 0.00275843, 0.09293491, -1.22484183, 0.88119119, 0.45697182);
const vec3 conv_shad_2_bias = vec3(0.3446977436542511, -0.0002613725373521447, -0.16102707386016846);
const vec3 conv_shad_3_weight = vec3(1.108639121055603, -0.794776439666748, -0.4122093617916107);
const float conv_shad_3_bias = 0.6914172172546387;


// =====================================================================================================================
// 	helper functions
// =====================================================================================================================

vec3 leakyReLU(vec3 v) {
    float x = v.x > 0 ? v.x : 0.01*v.x;
    float y = v.y > 0 ? v.y : 0.01*v.y;
    float z = v.z > 0 ? v.z : 0.01*v.z;
    return vec3(x, y, z);
}


mat3 leakyReLU(mat3 m) {
    return mat3(leakyReLU(m[0]), leakyReLU(m[1]), leakyReLU(m[2]));
}

vec3 sigmoid(vec3 v) {
    return 1. / (1. + exp(-v));
}

vec3 toonShaderAI(vec3 light, vec3 normal, vec3 view, 
					float occlusion, float emission, float shininess, 
					vec3 albedo, vec3 light_color, vec3 ambient_color,
					vec3 style_std, vec3 style_mean) {

										
	vec3 L = normalize(light);
	vec3 N = normalize(normal);
	vec3 V = normalize(view);
	vec3 R = reflect(L, N);
	
	vec3 lgt = vec3(0, dot(L, N), dot(R, V));
	vec3 mask = vec3(occlusion, emission, shininess);
	
	// lighting module
	vec3 feature = conv_lgt_0_weight_0 * lgt + conv_lgt_0_weight_1 * mask + conv_lgt_0_bias;
    feature = leakyReLU(feature);
    feature = feature * style_std + style_mean;
    
    vec3 feature_conv = conv_lgt_1_weight * feature + conv_lgt_1_bias;
    feature_conv = leakyReLU(feature_conv);
    feature_conv = conv_lgt_2_weight * feature_conv + conv_lgt_2_bias;
    feature = feature + feature_conv;
    feature = feature*style_std + style_mean;
    
	feature = conv_lgt_3_weight * feature + conv_lgt_3_bias;
    feature = sigmoid(feature);
	
	// coloring module
	mat3 feature_mat = mat3(feature, feature, feature);
    mat3 color_mat = transpose(mat3(albedo, light_color, ambient_color));
	
	color_mat = conv_shad_0_weight_1 * color_mat + mat3(conv_shad_0_bias, conv_shad_0_bias, conv_shad_0_bias);
    color_mat = conv_shad_0_weight_0 * feature_mat + color_mat; 
    color_mat = leakyReLU(color_mat);
    
    mat3 color_mat_conv = conv_shad_1_weight * color_mat + mat3(conv_shad_1_bias, conv_shad_1_bias, conv_shad_1_bias);
    color_mat_conv = leakyReLU(color_mat_conv);
    color_mat_conv = conv_shad_2_weight * color_mat_conv + mat3(conv_shad_2_bias, conv_shad_2_bias, conv_shad_2_bias);
    color_mat = color_mat + color_mat_conv;
    
    vec3 color = conv_shad_3_weight * color_mat + vec3(conv_shad_3_bias, conv_shad_3_bias, conv_shad_3_bias);
	color = sigmoid(color);
	
	return color;
}

// =====================================================================================================================
// 	main function
// =====================================================================================================================

void main() {

    vec4 albedo = texture(samplerAlbedo, inTexCoords);
    vec3 mask = texture(samplerMasking, inTexCoords).rgb;

	vec3 color = toonShaderAI(
		light.lightPos.xyz,
		inNormal,
		inViewPos - inWorldPos.xyz,
		mask.r,
		mask.g,
		mask.b,
		albedo.rgb,
		light.lightColor.rgb,
		light.ambientColor.rgb,
		target_style_std,
		target_style_mean 
	);

    outColor = vec4(color, albedo.a);
}