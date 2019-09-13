// Copyright 2018- PWRD, Inc. All Rights Reserved.

//#include "BRDF.hlsl" 
//#include "SH.hlsl"
//#include "DebugHead.hlsl"
//#include "../Colors.hlsl"

#ifndef PBS_ADDLIGHTING_INCLUDE
#define PBS_ADDLIGHTING_INCLUDE

#include "LightingHead.hlsl"

#if defined(DEBUG_APP)

	#if defined(SHADER_API_D3D11) || defined(SHADER_API_METAL)
		#define ATTEN_CHANNEL r
	#else
		#define ATTEN_CHANNEL a
	#endif

	#if defined(SHADER_API_D3D11) || defined(SHADER_API_D3D11_9X) || defined(UNITY_COMPILER_HLSLCC)
		#if defined(SHADER_API_D3D11_9X)
			#define DECLARE_SHADOWMAP(tex) Texture2D tex : register(t15); SamplerComparisonState sampler##tex : register(s15)
			#define DECLARE_TEXCUBE_SHADOWMAP(tex) TextureCube tex : register(t15); SamplerComparisonState sampler##tex : register(s15)
		#else
			#define DECLARE_SHADOWMAP(tex) Texture2D tex; SamplerComparisonState sampler##tex
			#define DECLARE_TEXCUBE_SHADOWMAP(tex) TextureCube tex; SamplerComparisonState sampler##tex
		#endif
		#define SAMPLE_SHADOW(tex,coord) tex.SampleCmpLevelZero (sampler##tex,(coord).xy,(coord).z)
		#define SAMPLE_SHADOW_PROJ(tex,coord) tex.SampleCmpLevelZero (sampler##tex,(coord).xy/(coord).w,(coord).z/(coord).w)
		#define SAMPLE_TEXCUBE_SHADOW(tex,coord) tex.SampleCmpLevelZero (sampler##tex,(coord).xyz,(coord).w)
	#elif defined(UNITY_COMPILER_HLSL2GLSL) && defined(SHADOWS_NATIVE)
		#define DECLARE_SHADOWMAP(tex) sampler2DShadow tex
		#define DECLARE_TEXCUBE_SHADOWMAP(tex) samplerCUBEShadow tex
		#define SAMPLE_SHADOW(tex,coord) shadow2D (tex,(coord).xyz)
		#define SAMPLE_SHADOW_PROJ(tex,coord) shadow2Dproj (tex,coord)
		#define SAMPLE_TEXCUBE_SHADOW(tex,coord) ((texCUBE(tex,(coord).xyz) < (coord).w) ? 0.0 : 1.0)
	#else
		#define DECLARE_SHADOWMAP(tex) sampler2D_float tex
		#define DECLARE_TEXCUBE_SHADOWMAP(tex) samplerCUBE_float tex
		#define SAMPLE_SHADOW(tex,coord) ((SAMPLE_DEPTH_TEXTURE(tex,(coord).xy) < (coord).z) ? 0.0 : 1.0)
		#define SAMPLE_SHADOW_PROJ(tex,coord) ((SAMPLE_DEPTH_TEXTURE_PROJ(tex,UNITY_PROJ_COORD(coord)) < ((coord).z/(coord).w)) ? 0.0 : 1.0)
		#define SAMPLE_TEXCUBE_SHADOW(tex,coord) ((SAMPLE_DEPTH_CUBE_TEXTURE(tex,(coord).xyz) < (coord).w) ? 0.0 : 1.0)
	#endif

	#if defined(_ADDLIGHTING)
		FLOAT4x4 unity_WorldToLight;
	#endif

	#if defined(SHADOWS_DEPTH) && defined(SPOT)||defined(SHADOWS_SCREEN)&&defined(UNITY_NO_SCREENSPACE_SHADOWS)
			FLOAT4x4 unity_WorldToShadow[4];
	#endif
//#ifdef DIRECTIONAL
//#define UNITY_LIGHT_ATTENUATION(destName, input, worldPos) fixed destName = UNITY_SHADOW_ATTENUATION(input, worldPos);
//#endif
//
	FLOAT4 _LightPositionRange;
	FLOAT4 _LightProjectionParams;

	#if defined (SHADOWS_CUBE)
		DECLARE_TEXCUBE_SHADOWMAP(_ShadowMapTexture);
	#else
		DECLARE_SHADOWMAP(_ShadowMapTexture);
	#endif
	FLOAT4 _LightShadowData;	
	#ifdef POINT
		TEXTURE2D_SAMPLER2D(_LightTexture0);
	#endif

	#ifdef SPOT
		TEXTURE2D_SAMPLER2D(_LightTexture0);
		TEXTURE2D_SAMPLER2D(_LightTextureB0);				
	#endif


	#ifdef POINT_COOKIE
		TEXCUBE_SAMPLERCUBE(_LightTexture0);
		TEXTURE2D_SAMPLER2D(_LightTextureB0);
	#endif

	#ifdef DIRECTIONAL_COOKIE
		TEXTURE2D_SAMPLER2D(_LightTexture0);
	#endif


	#if defined(SHADOWS_SCREEN)		
		#if defined(UNITY_NO_SCREENSPACE_SHADOWS)
		#endif//UNITY_NO_SCREENSPACE_SHADOWS
	#endif//SHADOWS_SCREEN
	
#endif//DEBUG_APP

inline FLOAT3 GetLightDir(FFragData FragData)
{
	FLOAT3 dir = GetLightDir0();
#ifdef _ADDLIGHTING
	#ifdef DIRECTIONAL
		return dir;
	#else
		return normalize(dir - FragData.WorldPosition);
	#endif
#else
	return dir;
#endif
}

inline FLOAT GetLightingAtten(FFragData FragData)
{
#ifdef _ADDLIGHTING
	#if defined(DEBUG_APP)
		#if defined(LIGHTMAP_ON)||defined(_CUSTOM_LIGHTMAP_ON)
			return 0;
		#else
			#ifdef POINT
				FLOAT3 lightCoord = mul(unity_WorldToLight, FLOAT4(FragData.WorldPosition, 1)).xyz;
				return SAMPLE_TEXTURE2D(_LightTexture0, dot(lightCoord, lightCoord).rr).ATTEN_CHANNEL;
			#endif

			#ifdef SPOT
				FLOAT4 lightCoord = mul(unity_WorldToLight, FLOAT4(FragData.WorldPosition, 1));
				FLOAT spotCookie = SAMPLE_TEXTURE2D(_LightTexture0, lightCoord.xy / lightCoord.w + 0.5).w;
				FLOAT spotAttenuate = SAMPLE_TEXTURE2D(_LightTextureB0, dot(lightCoord.xyz, lightCoord.xyz).xx).ATTEN_CHANNEL;
				return (lightCoord.z > 0) * spotCookie * spotAttenuate;
			#endif

			#ifdef POINT_COOKIE
				FLOAT3 lightCoord = mul(unity_WorldToLight, FLOAT4(FragData.WorldPosition, 1)).xyz;
				return SAMPLE_TEXTURE2D(_LightTextureB0, dot(lightCoord, lightCoord).rr).ATTEN_CHANNEL*SAMPLE_TEXCUBE(_LightTexture0, lightCoord).w;
			#endif

			#ifdef DIRECTIONAL_COOKIE
				FLOAT2 lightCoord = mul(unity_WorldToLight, FLOAT4(FragData.WorldPosition, 1)).xy;
				return SAMPLE_TEXTURE2D(_LightTexture0, lightCoord).w;
			#endif
		#endif
	#endif

	return 1;
#else//!DEBUG_APP
	return 1;
#endif//DEBUG_APP
}

FLOAT GetAddShadow(FFragData FragData)
{
#if defined(DEBUG_APP)
	#if defined(SHADOWS_SCREEN)
		#if defined(UNITY_NO_SCREENSPACE_SHADOWS)
			FLOAT4 shadowCoord = mul(unity_WorldToShadow[0], FLOAT4(FragData.WorldPosition, 1));
			FLOAT shadow = SAMPLE_SHADOW(_ShadowMapTexture, shadowCoord.xyz).r;
			shadow = _LightShadowData.r + shadow * (1 - _LightShadowData.r);
			return shadow;
		#endif//UNITY_NO_SCREENSPACE_SHADOWS
	
	#endif//SHADOWS_SCREEN

	#ifdef _ADDLIGHTING
		#if defined(SHADOWS_DEPTH) && defined(SPOT)
			FLOAT4 shadowCoord = mul(unity_WorldToShadow[0], FLOAT4(FragData.WorldPosition, 1));
			FLOAT shadow = SAMPLE_SHADOW_PROJ(_ShadowMapTexture, shadowCoord);
			return lerp(_LightShadowData.r, 1.0f, shadow);
		#endif//SHADOWS_DEPTH&&SPOT

		#if defined (SHADOWS_CUBE)
			FLOAT3 vec = FragData.WorldPosition - _LightPositionRange.xyz;
			FLOAT3 absVec = abs(vec);
			FLOAT dominantAxis = max(max(absVec.x, absVec.y), absVec.z);
			dominantAxis = max(0.00001, dominantAxis - _LightProjectionParams.z);
			dominantAxis *= _LightProjectionParams.w;
			FLOAT mydist = -_LightProjectionParams.x + _LightProjectionParams.y / dominantAxis;
			mydist = 1.0 - mydist;
			FLOAT z = 1.0 / 128.0;
			FLOAT4 shadowVals;
			shadowVals.x = SAMPLE_TEXCUBE_SHADOW(_ShadowMapTexture, FLOAT4(vec + FLOAT3(z, z, z), mydist));
			shadowVals.y = SAMPLE_TEXCUBE_SHADOW(_ShadowMapTexture, FLOAT4(vec + FLOAT3(-z, -z, z), mydist));
			shadowVals.z = SAMPLE_TEXCUBE_SHADOW(_ShadowMapTexture, FLOAT4(vec + FLOAT3(-z, z, -z), mydist));
			shadowVals.w = SAMPLE_TEXCUBE_SHADOW(_ShadowMapTexture, FLOAT4(vec + FLOAT3(z, -z, -z), mydist));
			FLOAT shadow = dot(shadowVals, 0.25);
			return lerp(_LightShadowData.r, 1.0, shadow);
		#endif//SHADOWS_CUBE

	#endif//_ADDLIGHTING

#endif//DEBUG_APP
	return 1;
}
#ifdef _VOXEL_LIGHT
//now only point lights
struct LightInfo
{
	FLOAT4 lightPos;//xyz:world pos w:range
	FLOAT4 lightColor;//xyz:color w: not use
};
struct LightHeadIndex
{
	uint blockStartIndex;
	FLOAT minY;
};
StructuredBuffer<LightInfo> _LightInfos;
StructuredBuffer<uint> _LightIndex;//count+index0,index1...
StructuredBuffer<uint> _VerticalBlockIndex;
StructuredBuffer<LightHeadIndex> _LightIndexHead;
uint _LineBlockCount;//default 20
// FLOAT _GridSize;
FLOAT3 _WorldChunkOffset;
// TEXTURE2D_2(_LightIndexTex0);
// TEXTURE2D_2(_LightIndexTex1);
// TEXTURE2D_2(_LightIndexTex2);
// TEXTURE2D_2(_LightIndexTex3);
// TEXTURE2D_2(_LightIndexTex4);
// TEXTURE2D_2(_LightIndexTex5);
// TEXTURE2D_2(_LightIndexTex6);
// TEXTURE2D_2(_LightIndexTex7);
// TEXTURE2D_2(_LightIndexTex8);

FLOAT3 GetVoxelLighting(FFragData FragData,FMaterialData MaterialData,FLightingData LightingData)
{
	FLOAT3 color = 0;
	FLOAT2 pos = FragData.WorldPosition.xz - _WorldChunkOffset.xy;
	uint2 xzIndex = floor(pos*_WorldChunkOffset.zz);
	uint headIndexOffset = xzIndex.x + xzIndex.y * _LineBlockCount;
	LightHeadIndex headIndex = _LightIndexHead[headIndexOffset];
	UNITY_BRANCH
	if(headIndex.minY < 1000)
	{
		uint verticalBlockIndex = floor((max(0,FragData.WorldPosition.y - headIndex.minY))*_WorldChunkOffset.z);
		uint blockOffset = _VerticalBlockIndex[headIndex.blockStartIndex + verticalBlockIndex];
		UNITY_BRANCH
		if(blockOffset < 1000000)
		{

			uint lightCount = _LightIndex[blockOffset];
			UNITY_LOOP
			for (uint i = 0; i < lightCount; i++)
			{
				uint lightIndex =  _LightIndex[blockOffset+i+1];
				LightInfo li = _LightInfos[lightIndex];

				FLOAT3 dir =  li.lightPos.xyz - FragData.WorldPosition.xyz;
				FLOAT3 ndir = normalize(dir);
				FLOAT3 ndotl = saturate(dot(MaterialData.WorldNormal, ndir));
				FLOAT distanceSqr = dot(dir,dir);
				FLOAT lightRadiusMask = distanceSqr * li.lightPos.w * li.lightPos.w;
				lightRadiusMask = saturate(1 - lightRadiusMask*lightRadiusMask);
				// lightRadiusMask *= lightRadiusMask;		
				FLOAT3 lighting = li.lightColor.xyz*ndotl*rcp(1+distanceSqr)*16*lightRadiusMask;
				color += Diffuse_Burley( LightingData.DiffuseColor,MaterialData.Roughness,LightingData.NdotC,ndotl,LightingData.VdotH)*lighting;
				// FLOAT h = SafeNormalize(FragData.CameraVector + ndir);
				// FLOAT ndoth = saturate(dot(MaterialData.WorldNormal, h));
				// color += LightingData.SpecularColor * CalcSpecular(MaterialData.Roughness, ndoth, h, MaterialData.WorldNormal)*lighting;
				// color +=lighting;
			}
		}
	}
	// LightInfo li = _LightInfos[0];

	// FLOAT3 dir =  li.lightPos.xyz - FragData.WorldPosition.xyz;
	// FLOAT3 ndotl = saturate(dot(MaterialData.WorldNormal, normalize(dir)));

	// FLOAT distanceSqr = dot(dir,dir);
	// FLOAT lightRadiusMask = distanceSqr * li.lightPos.w * li.lightPos.w;
	// lightRadiusMask = saturate(1 - lightRadiusMask);
	// //lightRadiusMask *= lightRadiusMask;		
	// lighting += li.lightColor.xyz*ndotl*rcp(1+distanceSqr)*16*lightRadiusMask;
	return color;
}

#endif//_VOXEL_LIGHT
#endif //PBS_ADDLIGHTING_INCLUDE