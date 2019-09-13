
/**
* Common.cginc: Common fun and paramaters.
*/
#include "../StdLib.hlsl"
#include "../Colors.hlsl"
#ifndef PBS_COMMON_INCLUDE
#define PBS_COMMON_INCLUDE

#define TANGENTTOWORLD0					TEXCOORD10
#define TANGENTTOWORLD2					TEXCOORD11

#if defined(_WIND_EFFECT)||defined(_ATLASINCOLOR)
	#ifndef _VERTEX_COLOR
		#define _VERTEX_COLOR
	#endif
#endif
inline FLOAT4 TransformWorldToClipPos(in FLOAT4 WorldPosition)
{
	return mul(_matrixVP, WorldPosition);
}

// inline FLOAT4 TransformWorldToViewPos(in FLOAT4 WorldPosition)
// {
// 	return mul(unity_MatrixV, WorldPosition);
// }

inline FLOAT4 TransformObjectToClipPos(in FLOAT3 pos)
{
	// More efficient than computing M*VP matrix product
	return mul(_matrixVP, mul(_objectToWorld, FLOAT4(pos, 1.0)));
}

inline FLOAT InRange(FLOAT2 uv, FLOAT4 ranges)
{
	FLOAT2 value = step(ranges.xy, uv) * step(uv, ranges.zw);
	return value.x * value.y;
}

inline FLOAT3 SafeNormalize(float3 inVec)
{
    FLOAT dp3 = max(0.001f, dot(inVec, inVec));
    return inVec * rsqrt(dp3);
}

inline FLOAT3 UnpackNormal(FLOAT2 packednormal)
{
	FLOAT3 normal;
	normal.xy = packednormal.xy * 2 - 1;
	normal.z = sqrt(1 - saturate(dot(normal.xy, normal.xy)));
	return normal;
}

inline void CalcWorldNormal(FLOAT3x3 TangentToWorld, FLOAT2 Normal,out FLOAT3 WorldNormal,out FLOAT3 TangentSpaceNormal)
{
	FLOAT3 worldTangent = TangentToWorld[0].xyz;
	FLOAT3 worldBinormal = TangentToWorld[1].xyz;
	FLOAT3 worldNormal = TangentToWorld[2].xyz;

	TangentSpaceNormal = UnpackNormal(Normal);
	WorldNormal = normalize(worldTangent * TangentSpaceNormal.x + worldBinormal * TangentSpaceNormal.y + worldNormal * TangentSpaceNormal.z);
}

inline void CalcWorldNormal2(FLOAT3x3 TangentToWorld, FLOAT3 Normal,out FLOAT3 WorldNormal,out FLOAT3 TangentSpaceNormal)
{
	FLOAT3 worldTangent = TangentToWorld[0].xyz;
	FLOAT3 worldBinormal = TangentToWorld[1].xyz;
	FLOAT3 worldNormal = TangentToWorld[2].xyz;

	TangentSpaceNormal = Normal;
	WorldNormal = normalize(worldTangent * TangentSpaceNormal.x + worldBinormal * TangentSpaceNormal.y + worldNormal * TangentSpaceNormal.z);
}

FLOAT AOMultiBounce( FLOAT3 Color, FLOAT AO, FLOAT AOBias )
{
	FLOAT gray = Luminance(Color);
	FLOAT a =  2.0404 * gray - 0.3324;
	FLOAT b = -4.7951 * gray + 0.6417;
	FLOAT c =  2.7552 * gray + 0.6903;
	FLOAT AO2 = AO*AO;
	FLOAT AO3 = AO2*AO;
	gray =  max(AO, (a*AO3 + b*AO2 + c*AO));
// gray =  max( AO, ( ( AO * a + b ) * AO + c ) * AO );
	return gray + 1 - AOBias;
}

#define BLOCKER_SEARCH_NUM_SAMPLES 8
#define PCF_NUM_SAMPLES 8

static FLOAT2 g_ShadowPoissonDisk[8] = 
{
	FLOAT2(0.02971195f, -0.8905211f),
	FLOAT2(0.2495298f, 0.732075f),
	FLOAT2(-0.3469206f, -0.6437836f),
	FLOAT2(-0.01878909f, 0.4827394f),
	FLOAT2(-0.2725213f, -0.896188f),
	FLOAT2(-0.6814336f, 0.6480481f),
	FLOAT2(0.4152045f, -0.2794172f),
	FLOAT2(0.1310554f, 0.2675925f),
};
TEXTURE2D_SAMPLER2D(_ShadowMapTex);
 FLOAT4 _ShadowMapTex_TexelSize;

FLOAT PenumbraSize(FLOAT zReceiver, FLOAT zBlocker)
{
	return (zReceiver - zBlocker) / zBlocker;
}

void FindBlocker(out FLOAT avgBlockerDepth, out FLOAT numBlockers, FLOAT2 uv, FLOAT zReceiver)
{
	FLOAT blockerSum = 0.0f;
	numBlockers = 0;
	
	UNITY_UNROLL
	for(int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; ++i)
	{
		FLOAT2 offset = g_ShadowPoissonDisk[i] * 1 * _ShadowMapTex_TexelSize.xy;

		FLOAT4 shadowMap = SAMPLE_TEXTURE2D(_ShadowMapTex, uv + offset);
		FLOAT shadowMapDepth = DecodeFloatRGBA(shadowMap);
		//FLOAT shadowMapDepth = g_ShadowDepthTexture.SampleLevel(CLAMPTRILINEAR, uv + offset, 0);
		
		if (shadowMapDepth < zReceiver)
		{
			blockerSum += shadowMapDepth;
			numBlockers++;
		}
	}
	
	avgBlockerDepth = blockerSum / numBlockers;
}

FLOAT PCFFilter(FLOAT2 uv, FLOAT zReceiver, FLOAT filterRadiusUV)
{
	FLOAT sum = 0.0f;

	UNITY_UNROLL
	for(int i = 0; i < PCF_NUM_SAMPLES; ++i)
	{
		FLOAT2 offset = g_ShadowPoissonDisk[i] * filterRadiusUV * _ShadowMapTex_TexelSize.xy;
		FLOAT4 shadowMap = SAMPLE_TEXTURE2D(_ShadowMapTex, uv + offset);
		FLOAT d = DecodeFloatRGBA(shadowMap);
		// FLOAT d = g_ShadowDepthTexture.SampleLevel(CLAMPTRILINEAR, uv + offset, 0);
		sum += d >= zReceiver;
	}
	
	return sum / PCF_NUM_SAMPLES;
}

FLOAT PCSSFilter(FLOAT3 coords)
{
	FLOAT2 uv = coords.xy;
	FLOAT zReceiver = coords.z;//saturate(coords.z);
	
	FLOAT avgBlockerDepth = 0.0f;
	FLOAT numBlockers = 0.0f;
	
	FLOAT4 shadowMap = SAMPLE_TEXTURE2D(_ShadowMapTex, coords.xy);
	avgBlockerDepth = DecodeFloatRGBA(shadowMap);
	FindBlocker(avgBlockerDepth, numBlockers, uv, zReceiver);
	if (numBlockers < 1)
		return 1.0f;
	
	FLOAT penumbraRatio = PenumbraSize(zReceiver, avgBlockerDepth);
	FLOAT filterRadiusUV = penumbraRatio * 1;
	FLOAT pcf = PCFFilter(uv, zReceiver, filterRadiusUV);
	return pcf;
}

FLOAT4 _SmoothClamp;
FLOAT PCF8Filter(FLOAT3 coords)
{
	FLOAT shadow = 0;
	FLOAT filterSize = _ShadowMapTex_TexelSize.x* _SmoothClamp.z;
	FLOAT2 smoothClamp = coords.z*(FLOAT2(1,1) + _SmoothClamp.xy);
	UNITY_UNROLL
	for(int x = 0;x < 8; ++x)
	{
		FLOAT2 uv = coords.xy + g_ShadowPoissonDisk[x] * filterSize;
		FLOAT4 shadowMap = SAMPLE_TEXTURE2D(_ShadowMapTex, uv.xy);
		FLOAT d = shadowMap.r;//DecodeFloatRGBA(shadowMap);
		FLOAT s = smoothstep(smoothClamp.x,smoothClamp.y,d);
		shadow += s;
	}
	shadow /= 8.0;
	
	shadow = saturate((shadow) * _SmoothClamp.w);
	FLOAT2 pos = coords.xy * 2 - 1;
	shadow *= 1 - smoothstep(0.7,1,length(pos));

	return shadow;
}
// FLOAT4     _ParallaxParam;
// int4       _ParallaxCount;


#endif //PBS_COMMON_INCLUDE