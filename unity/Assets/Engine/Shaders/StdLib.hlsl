// Because this framework is supposed to work with the legacy render pipelines AND scriptable render
// pipelines we can't use Unity's shader libraries (some scriptable pipelines come with their own
// shader lib). So here goes a minimal shader lib only used for post-processing to ensure good
// compatibility with all pipelines.

#ifndef UNITY_POSTFX_STDLIB
#define UNITY_POSTFX_STDLIB

// -----------------------------------------------------------------------------
// API macros

#if defined(SHADER_API_D3D11)
    #include "API/D3D11.hlsl"
    #define DEBUG_APP
#elif defined(SHADER_API_D3D12)
    #include "API/D3D12.hlsl"
    #define DEBUG_APP
#elif defined(SHADER_API_VULKAN) || defined(SHADER_API_SWITCH)
    #include "API/Vulkan.hlsl"
#elif defined(SHADER_API_METAL)
    #include "API/Metal.hlsl"
#else
    #include "API/OpenGL.hlsl"
#endif
// -----------------------------------------------------------------------------
// Constants
#if defined(SHADER_STAGE_FRAGMENT)||defined(SHADER_STAGE_COMPUTE)
#define FLOAT		half
#define FLOAT2		half2
#define FLOAT3		half3
#define FLOAT4		half4
#define FLOAT2x2	half2x2
#define FLOAT3x3	half3x3
#define FLOAT4x4	half4x4
#else
#define FLOAT		float
#define FLOAT2		float2
#define FLOAT3		float3
#define FLOAT4		float4
#define FLOAT2x2	float2x2
#define FLOAT3x3	float3x3
#define FLOAT4x4	float4x4
#endif

static const FLOAT4 ONES =  FLOAT4(1.0, 1.0, 1.0, 1.0);
static const FLOAT4 ZEROES = (FLOAT4)0.0;
#define HALF_MAX        65504.0
#define EPSILON         1.0e-4
#define PI              3.14159265359
#define TWO_PI          6.28318530718
#define FOUR_PI         12.56637061436
#define INV_PI          0.31830988618
#define INV_TWO_PI      0.15915494309
#define INV_FOUR_PI     0.07957747155
#define HALF_PI         1.57079632679
#define INV_HALF_PI     0.636619772367

#define FLT_EPSILON     1.192092896e-07 // Smallest positive number, such that 1.0 + FLT_EPSILON != 1.0
#define FLT_MIN         1.175494351e-38 // Minimum representable positive FLOATing-point number
#define FLT_MAX         3.402823466e+38 // Maximum representable FLOATing-point number

#define s2(a, b)				temp = a; a = min(a, b); b = max(temp, b);
#define mn3(a, b, c)			s2(a, b); s2(a, c);
#define mx3(a, b, c)			s2(b, c); s2(a, c);

#define mnmx3(a, b, c)				mx3(a, b, c); s2(a, b);                                   // 3 exchanges
#define mnmx4(a, b, c, d)			s2(a, b); s2(c, d); s2(a, c); s2(b, d);                   // 4 exchanges
#define mnmx5(a, b, c, d, e)		s2(a, b); s2(c, d); mn3(a, c, e); mx3(b, d, e);           // 6 exchanges
#define mnmx6(a, b, c, d, e, f) 	s2(a, d); s2(b, e); s2(c, f); mn3(a, b, c); mx3(d, e, f); // 7 exchanges

#if defined(SHADER_API_D3D11) || defined(SHADER_API_D3D12) || defined(SHADER_API_VULKAN)|| defined(SHADER_API_METAL)|| defined(SHADER_API_GLES3)
#define INITIALIZE_OUTPUT(type,name) name = (type)0;
#define DECLARE_OUTPUT(type,name) type name;INITIALIZE_OUTPUT(type,name)
#else
#define INITIALIZE_OUTPUT(type,name)
#define DECLARE_OUTPUT(type,name) type name;
#endif

 #define INSTANCED_ARRAY_SIZE  250

#define INSTANCING_BUFFER_START(buf)      CBUFFER_START(Instancing_##buf) struct {
#define INSTANCING_BUFFER_END(arr)        } arr##Array[INSTANCED_ARRAY_SIZE]; CBUFFER_END
#define DEFINE_INSTANCED_PROP(type, var)  type var;
#define ACCESS_INSTANCED_PROP(arr, var)   arr##Array[instanceID].var
// -----------------------------------------------------------------------------
// Compatibility functions

#if (SHADER_TARGET < 50 && !defined(SHADER_API_PSSL))
inline FLOAT rcp(FLOAT value)
{
    return 1.0 / value;
}

inline FLOAT2 rcp(FLOAT2 value)
{
    return 1.0 / value;
}

inline FLOAT3 rcp(FLOAT3 value)
{
    return 1.0 / value;
}

inline FLOAT4 rcp(FLOAT4 value)
{
    return 1.0 / value;
}

#endif



#if defined(SHADER_API_GLES)
#define mad(a, b, c) (a * b + c)
#endif

#ifndef INTRINSIC_MINMAX3
inline FLOAT Min3(FLOAT a, FLOAT b, FLOAT c)
{
    return min(min(a, b), c);
}

inline FLOAT2 Min3(FLOAT2 a, FLOAT2 b, FLOAT2 c)
{
    return min(min(a, b), c);
}

inline FLOAT3 Min3(FLOAT3 a, FLOAT3 b, FLOAT3 c)
{
    return min(min(a, b), c);
}

inline FLOAT4 Min3(FLOAT4 a, FLOAT4 b, FLOAT4 c)
{
    return min(min(a, b), c);
}

inline FLOAT Min3(FLOAT3 x) 
{ 
	return min(x.x, min(x.y, x.z)); 
}

inline FLOAT Max3(FLOAT a, FLOAT b, FLOAT c)
{
    return max(max(a, b), c);
}

inline FLOAT2 Max3(FLOAT2 a, FLOAT2 b, FLOAT2 c)
{
    return max(max(a, b), c);
}

inline FLOAT3 Max3(FLOAT3 a, FLOAT3 b, FLOAT3 c)
{
    return max(max(a, b), c);
}

inline FLOAT4 Max3(FLOAT4 a, FLOAT4 b, FLOAT4 c)
{
    return max(max(a, b), c);
}

inline FLOAT Max3(FLOAT3 x) 
{ 
	return max(x.x, max(x.y, x.z)); 
}
#endif // INTRINSIC_MINMAX3

// https://twitter.com/SebAaltonen/status/878250919879639040
// madd_sat + madd
inline FLOAT FastSign(FLOAT x)
{
    return saturate(x * FLT_MAX + 0.5) * 2.0 - 1.0;
}

inline FLOAT2 FastSign(FLOAT2 x)
{
    return saturate(x * FLT_MAX + 0.5) * 2.0 - 1.0;
}

inline FLOAT3 FastSign(FLOAT3 x)
{
    return saturate(x * FLT_MAX + 0.5) * 2.0 - 1.0;
}

inline FLOAT4 FastSign(FLOAT4 x)
{
    return saturate(x * FLT_MAX + 0.5) * 2.0 - 1.0;
}

// Using pow often result to a warning like this
// "pow(f, e) will not work for negative f, use abs(f) or conditionally handle negative values if you expect them"
// PositivePow remove this warning when you know the value is positive and avoid inf/NAN.
inline FLOAT PositivePow(FLOAT base, FLOAT power)
{
    return pow(max(abs(base), FLOAT(FLT_EPSILON)), power);
}

inline FLOAT2 PositivePow(FLOAT2 base, FLOAT2 power)
{
    return pow(max(abs(base), FLOAT2(FLT_EPSILON, FLT_EPSILON)), power);
}

inline FLOAT3 PositivePow(FLOAT3 base, FLOAT3 power)
{
    return pow(max(abs(base), FLOAT3(FLT_EPSILON, FLT_EPSILON, FLT_EPSILON)), power);
}

inline FLOAT4 PositivePow(FLOAT4 base, FLOAT4 power)
{
    return pow(max(abs(base), FLOAT4(FLT_EPSILON, FLT_EPSILON, FLT_EPSILON, FLT_EPSILON)), power);
}

// NaN checker
// /Gic isn't enabled on fxc so we can't rely on isnan() anymore
inline bool IsNan(FLOAT x)
{
    // For some reason the following tests outputs "internal compiler error" randomly on desktop
    // so we'll use a safer but slightly slower version instead :/
    //return (x <= 0.0 || 0.0 <= x) ? false : true;
    return (x < 0.0 || x > 0.0 || x == 0.0) ? false : true;
}

inline bool AnyIsNan(FLOAT2 x)
{
    return IsNan(x.x) || IsNan(x.y);
}

inline bool AnyIsNan(FLOAT3 x)
{
    return IsNan(x.x) || IsNan(x.y) || IsNan(x.z);
}

inline bool AnyIsNan(FLOAT4 x)
{
    return IsNan(x.x) || IsNan(x.y) || IsNan(x.z) || IsNan(x.w);
}


inline FLOAT Square(FLOAT x)
{
	return x*x;
}

inline FLOAT2 Square(FLOAT2 x)
{
	return x*x;
}

inline FLOAT3 Square(FLOAT3 x)
{
	return x*x;
}

inline FLOAT4 Square(FLOAT4 x)
{
	return x*x;
}
 
inline FLOAT Pow4(FLOAT x)
{
    FLOAT x2 = x*x;
    
    return x2*x2;
}

inline FLOAT Pow5(FLOAT x)
{
	FLOAT xx = x*x;
	return xx * xx * x;
}

inline FLOAT2 Pow5(FLOAT2 x)
{
	FLOAT2 xx = x*x;
	return xx * xx * x;
}

inline FLOAT3 Pow5(FLOAT3 x)
{
	FLOAT3 xx = x*x;
	return xx * xx * x;
}

inline FLOAT4 Pow5(FLOAT4 x)
{
	FLOAT4 xx = x*x;
	return xx * xx * x;
}
inline FLOAT Pow8(FLOAT x)
{
    FLOAT x2 = x*x;
    FLOAT x4 = x2*x2;

    return x4*x4;
}

inline FLOAT Pow16(FLOAT x)
{
    FLOAT x2 = x*x;
    FLOAT x4 = x2*x2;
    FLOAT x8 = x4*x4;

    return x8*x8;
}
inline FLOAT Pow10(in FLOAT x)
{
	FLOAT x2 = x*x;
	FLOAT x4 = x2*x2;

	return x4*x4*x2;
}

inline FLOAT2 Pow10(in FLOAT2 x)
{
	FLOAT2 x2 = x*x;
	FLOAT2 x4 = x2*x2;

	return x4*x4*x2;
}

inline FLOAT3 Pow10(in FLOAT3 x)
{
	FLOAT3 x2 = x*x;
	FLOAT3 x4 = x2*x2;

	return x4*x4*x2;
}

inline FLOAT4 Pow10(in FLOAT4 x)
{
	FLOAT4 x2 = x*x;
	FLOAT4 x4 = x2*x2;

	return x4*x4*x2;
}

inline FLOAT LerpStep(FLOAT a,FLOAT b,FLOAT t)
{
    FLOAT m = b - a;
    m = (m <= 0.0 ? m : 1e-5);
    return saturate((t - a) * rcp(m));
}

inline FLOAT2 LerpStep(FLOAT2 a,FLOAT2 b,FLOAT2 t)
{
    FLOAT2 m = b - a;
    m = (m <= 0.0 ? m : 1e-5);
    return saturate((t - a) * rcp(m));
}

inline FLOAT3 LerpStep(FLOAT3 a,FLOAT3 b,FLOAT3 t)
{
    FLOAT3 m = b - a;
    m = (m <= 0.0 ? m : 1e-5);
    return saturate((t - a) * rcp(m));
}

inline FLOAT4 LerpStep(FLOAT4 a,FLOAT4 b,FLOAT4 t)
{
    FLOAT4 m = b - a;
    m = (m <= 0.0 ? m : 1e-5);
    return saturate((t - a) * rcp(m));
}

// Clamp the base, so it's never <= 0.0f (INF/NaN).
inline FLOAT ClampedPow(FLOAT X, FLOAT Y)
{
	return pow(max(abs(X), 0.000001f), Y);
}
inline FLOAT2 ClampedPow(FLOAT2 X, FLOAT2 Y)
{
	return pow(max(abs(X), FLOAT2(0.000001f, 0.000001f)), Y);
}
inline FLOAT3 ClampedPow(FLOAT3 X, FLOAT3 Y)
{
	return pow(max(abs(X), FLOAT3(0.000001f, 0.000001f, 0.000001f)), Y);
}
inline FLOAT4 ClampedPow(FLOAT4 X, FLOAT4 Y)
{
	return pow(max(abs(X), FLOAT4(0.000001f, 0.000001f, 0.000001f, 0.000001f)), Y);
}
/**
* Use this function to compute the pow() in the specular computation.
* This allows to change the implementation depending on platform or it easily can be replaced by some approxmation.
*/
inline FLOAT PhongShadingPow(FLOAT X, FLOAT Y)
{
	// The following clamping is done to prevent NaN being the result of the specular power computation.
	// Clamping has a minor performance cost.

	// In HLSL pow(a, b) is implemented as exp2(log2(a) * b).

	// For a=0 this becomes exp2(-inf * 0) = exp2(NaN) = NaN.

	// As seen in #TTP 160394 "QA Regression: PS3: Some maps have black pixelated artifacting."
	// this can cause severe image artifacts (problem was caused by specular power of 0, lightshafts propagated this to other pixels).
	// The problem appeared on PlayStation 3 but can also happen on similar PC NVidia hardware.

	// In order to avoid platform differences and rarely occuring image atrifacts we clamp the base.

	// Note: Clamping the exponent seemed to fix the issue mentioned TTP but we decided to fix the root and accept the
	// minor performance cost.

	return ClampedPow(X, Y);
}

inline FLOAT CosLike(FLOAT x)
{
    FLOAT z = abs(frac(x)-0.5)*2;
    z = 1 - z * z; 
    z = 1 - z * z; 
    z = (z - 0.5) * 2; 
    return z;
}

inline FLOAT2 CosLike(FLOAT2 x)
{
    FLOAT2 z = abs(frac(x)-0.5)*2;
    z = 1 - z * z; 
    z = 1 - z * z; 
    z = (z - 0.5) * 2; 
    return z;
}

inline FLOAT3 CosLike(FLOAT3 x)
{
    FLOAT3 z = abs(frac(x)-0.5)*2;
    z = 1 - z * z; 
    z = 1 - z * z; 
    z = (z - 0.5) * 2; 
    return z;
}

inline FLOAT4 CosLike(FLOAT4 x)
{
    FLOAT4 z = abs(frac(x)-0.5)*2;
    z = 1 - z * z; 
    z = 1 - z * z; 
    z = (z - 0.5) * 2; 
    return z;
}

inline FLOAT3 DecodeHDR(FLOAT4 data, FLOAT3 decodeInstructions)
{
	// Take into account texture alpha if decodeInstructions.w is true(the alpha value affects the RGB channels)
	FLOAT alpha = decodeInstructions.z * (data.a - 1.0) + 1.0;

#   if defined(UNITY_USE_NATIVE_HDR)
	    return decodeInstructions.x * data.rgb; // Multiplier for future HDRI relative to absolute conversion.
#   else
	    return (decodeInstructions.x * alpha) * data.rgb;
#   endif
}

inline float4 EncodeFloatRGBA( float v )
{
    float4 kEncodeMul = float4(1.0, 255.0, 65025.0, 16581375.0);
    float kEncodeBit = 1.0/255.0;
    float4 enc = kEncodeMul * v;
    enc = frac (enc);
    enc -= enc.yzww * kEncodeBit;
    return enc;
}

inline float DecodeFloatRGBA( float4 enc )
{
    float4 kDecodeDot = float4(1.0, 1/255.0, 1/65025.0, 1/16581375.0);
    return dot( enc, kDecodeDot );
}
// -----------------------------------------------------------------------------
// Std unity data

//per view
FLOAT4x4 glstate_matrix_projection;
#define UNITY_MATRIX_P glstate_matrix_projection
FLOAT4x4 unity_CameraProjection;
FLOAT4x4 unity_MatrixVP;
FLOAT4x4 unity_MatrixV;
FLOAT4x4 unity_WorldToCamera;
FLOAT3 _WorldSpaceCameraPos;
FLOAT3 _GameViewWorldSpaceCameraPos;
FLOAT4 _WorldSpaceLightPos0;
FLOAT4 _ProjectionParams;         // x: 1 (-1 flipped), y: near,     z: far,       w: 1/far

FLOAT4 _LightColor0;
FLOAT4 unity_LightmapST;
//FLOAT4 custom_LightmapST;

FLOAT4 unity_AmbientSky;
FLOAT4 unity_AmbientEquator;
FLOAT4 unity_AmbientGround;
FLOAT4 unity_IndirectSpecColor;

FLOAT4 unity_ColorSpaceLuminance;
FLOAT4 unity_DeltaTime;           // x: dt,             y: 1/dt,     z: smoothDt,  w: 1/smoothDt
FLOAT4 unity_OrthoParams;         // x: width,          y: height,   z: unused,    w: ortho ? 1 : 0
FLOAT4 _ZBufferParams;            // x: 1-far/near,     y: far/near, z: x/far,     w: y/far
FLOAT4 _ScreenParams;             // x: width,          y: height,   z: 1+1/width, w: 1+1/height
FLOAT4 _Time;                     // x: t/20,           y: t,        z: t*2,       w: t*3
FLOAT4 _SinTime;                  // x: sin(t/20),      y: sin(t),   z: sin(t*2),  w: sin(t*3)
FLOAT4 _CosTime;                  // x: cos(t/20),      y: cos(t),   z: cos(t*2),  w: cos(t*3)


//per draw
FLOAT4x4 unity_ObjectToWorld;
FLOAT4x4 unity_WorldToObject;
FLOAT4 unity_WorldTransformParams;

#ifdef _LOCAL_WORLD_OFFSET
    // FLOAT4x4 custom_ObjectToWorld;
    // FLOAT4x4 custom_WorldToObject;

    FLOAT4x4 custom_MatrixVP;
    #define _objectToWorld unity_ObjectToWorld
    #define _worldToObject unity_WorldToObject
    #define _matrixVP unity_MatrixVP
#else//!_LOCAL_WORLD_OFFSET
    #define _objectToWorld unity_ObjectToWorld
    #define _worldToObject unity_WorldToObject
    #define _matrixVP unity_MatrixVP
#endif//_LOCAL_WORLD_OFFSET


// -----------------------------------------------------------------------------
// Std functions

// Z buffer depth to linear 0-1 depth
// Handles orthographic projection correctly
FLOAT Linear01Depth(FLOAT z)
{
    FLOAT isOrtho = unity_OrthoParams.w;
    FLOAT isPers = 1.0 - unity_OrthoParams.w;
    z *= _ZBufferParams.x;
    return (1.0 - isOrtho * z) / (isPers * z + _ZBufferParams.y);
}

inline FLOAT LinearEyeDepth(FLOAT z)
{
    return rcp(_ZBufferParams.z * z + _ZBufferParams.w);
}

// Clamp HDR value within a safe range
inline FLOAT3 SafeHDR(FLOAT3 c)
{
    return min(c, HALF_MAX);
}

inline FLOAT4 SafeHDR(FLOAT4 c)
{
    return min(c, HALF_MAX);
}

// Decode normals stored in _CameraDepthNormalsTexture
FLOAT3 DecodeViewNormalStereo(FLOAT4 enc4)
{
    FLOAT kScale = 1.7777;
    FLOAT3 nn = enc4.xyz * FLOAT3(2.0 * kScale, 2.0 * kScale, 0) + FLOAT3(-kScale, -kScale, 1);
    FLOAT g = 2.0 / dot(nn.xyz, nn.xyz);
    FLOAT3 n;
    n.xy = g * nn.xy;
    n.z = g - 1.0;
    return n;
}

// Interleaved gradient function from Jimenez 2014
// http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
FLOAT GradientNoise(FLOAT2 uv)
{
    uv = floor(uv * _ScreenParams.xy);
    FLOAT f = dot(FLOAT2(0.06711056, 0.00583715), uv);
    return frac(52.9829189 * frac(f));
}

// Vertex manipulation
inline FLOAT2 TransformTriangleVertexToUV(FLOAT2 vertex)
{
    FLOAT2 uv = (vertex + 1.0) * 0.5;
    return uv;
}

// Brightness function
inline FLOAT Brightness(FLOAT3 c)
{
	return Max3(c);
}

inline FLOAT4 ComputeGrabScreenPos (FLOAT4 pos) 
{
#if UNITY_UV_STARTS_AT_TOP
    FLOAT scale = -1.0;
#else//!UNITY_UV_STARTS_AT_TOP
    FLOAT scale = 1.0;
#endif//UNITY_UV_STARTS_AT_TOP
    FLOAT4 o = pos * 0.5f;
    o.xy = FLOAT2(o.x, o.y*scale) + o.w;
    o.zw = pos.zw;
    return o;
}

inline FLOAT4 ComputeScreenPos(FLOAT4 pos)
{
    FLOAT4 o = pos * 0.5f;
    o.xy = FLOAT2(o.x, o.y*_ProjectionParams.x) + o.w;
    o.zw = pos.zw;
    return o;
}

// -----------------------------------------------------------------------------
// Default vertex shaders

struct AttributesDefault
{
    FLOAT3 vertex : POSITION;
};

struct VaryingsDefault
{
	FLOAT4 vertex : SV_POSITION;
	FLOAT2 texcoord : TEXCOORD0;
	//FLOAT2 texcoordStereo : TEXCOORD1;
};

VaryingsDefault VertDefault(AttributesDefault v)
{
    VaryingsDefault o;
    o.vertex = FLOAT4(v.vertex.xy, 0.0, 1.0);
    o.texcoord = TransformTriangleVertexToUV(v.vertex.xy);

#if UNITY_UV_STARTS_AT_TOP
    o.texcoord = o.texcoord * FLOAT2(1.0, -1.0) + FLOAT2(0.0, 1.0);
#endif

    //o.texcoordStereo = TransformStereoScreenSpaceTex(o.texcoord, 1.0);

    return o;
}

FLOAT4 _UVTransform; // xy: scale, wz: translate

VaryingsDefault VertUVTransform(AttributesDefault v)
{
    VaryingsDefault o;
    o.vertex = FLOAT4(v.vertex.xy, 0.0, 1.0);
    o.texcoord = TransformTriangleVertexToUV(v.vertex.xy) * _UVTransform.xy + _UVTransform.zw;
    //o.texcoord = TransformStereoScreenSpaceTex(o.texcoord, 1.0);
    return o;
}

#define TRANSFORM_TEX(tex,name) (tex.xy * name##_ST.xy + name##_ST.zw)

#endif // UNITY_POSTFX_STDLIB
