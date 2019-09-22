#include "MaterialTemplate.hlsl" 
#include "Lighting.hlsl" 
#include "Fog.hlsl"
#include "Debug.hlsl" 

#ifndef PBS_PIXEL_INCLUDE
#define PBS_PIXEL_INCLUDE 

FLOAT3 ApplyFog2(FLOAT3 color, FLOAT4 fogParam, FLOAT3 WorldPosition)
{
	FLOAT3 basecolor = color;
	FLOAT3 viewDir = normalize(WorldPosition - _WorldSpaceCameraPos.xyz);
	FLOAT3 lightDir = GetLightDir0();
	FLOAT VoL = saturate(dot(-viewDir, -lightDir));
	FLOAT VSDirFog=(1-saturate(viewDir.y *2.5+ _HeightFogParam.y-1) ); 
	FLOAT3 C = lerp(basecolor,lerp(basecolor,_HeightFogColor0.xyz+_HeightFogColor1.xyz*VSDirFog+_HeightFogColor2.xyz*VoL*VoL*2,VSDirFog),fogParam.x);
	return C;
}


FLOAT4 fragForwardBase(in FInterpolantsVSToPS Interpolants, in FLOAT4 SvPosition : SV_Position) : SV_Target
{
	DEBUG_PBS_CUSTOMDATA
	FFragData FragData = GetFragData(Interpolants, SvPosition);
	FMaterialData MaterialData = GetMaterialData(FragData);
	FLightingData LightingData = GetLighting(FragData, MaterialData DEBUG_PBS_PARAM);

	FLOAT3 DiffuseGI;
	FLOAT3 Color = 0;
	FLOAT Alpha = MaterialData.BaseColor.a;
#ifdef _ALPHA_MODIFY
	Alpha = GetAlpha(FragData);
#endif
#ifdef _FULL_SSS	
	#ifdef _SSSProfile
		FLOAT3 skin = SubsurfaceProfileBxDF(FragData, MaterialData, LightingData DEBUG_PBS_PARAM);
	#else
		FLOAT3 skin = GetSkinLighting(FragData, MaterialData, LightingData DEBUG_PBS_PARAM);
	#endif
	
	Color += skin;

#else//_FULL_SSS

	#ifdef _PART_SSS
		UNITY_BRANCH
		if (FragData.TexCoords[0].x > 1.0)
		{			
			FLOAT3 skin = GetSkinLighting(FragData, MaterialData, LightingData DEBUG_PBS_PARAM);
			Color += skin;
		#ifndef _ALPHA_FROM_COLOR
			Alpha = 1;
		#endif
		}
		else
	#endif//_PART_SSS > 1
		{			
			FLOAT3 DirectLighting = GetDirectLighting(FragData, MaterialData, LightingData DEBUG_PBS_PARAM);//direct diffuse
			Color += DirectLighting;
	#if !defined(_PBS_NO_IBL)
			FLOAT3 ImageBasedReflectionLighting = GetImageBasedReflectionLighting(FragData, MaterialData, LightingData DEBUG_PBS_PARAM);//gi(specular+sh)
			Color += ImageBasedReflectionLighting;
	#endif
		}

#endif//_FULL_SSS


#ifdef _CUSTOM_EFFECT
	GetCustomLighting(LightingData, Color, _Color);
#endif//_CUSTOM_EFFECT 

	//EMISSIVE
#ifdef _ETX_EFFECT
	Color += MaterialData.EmissiveAO.rgb;
#endif

#ifdef _NEED_BOX_PROJECT_REFLECT

	FLOAT3 ReflCube = DecodeHDR(SAMPLE_TEXCUBE_LOD(_EnvReflectTex, MaterialData.ReflectionVector, _BoxCenter.w), _EnvCubemapParam.xyz);
	DEBUG_PBS_CUSTOMDATA_PARAM(ReflCube, ReflCube)
	Color = lerp(Color,ReflCube,_BoxSize.w);
#endif//_NEED_BOX_PROJECT_REFLECT

#ifdef _PARALLAX_EFFECT
	Color *= FragData.Parallax.a;
#endif //_PARALLAX_EFFECT
	FLOAT4 OutColor = FLOAT4(Color, Alpha);


#ifdef _VERTEX_FOG	
	OutColor.rgb = ApplyFog2(OutColor.rgb,FragData.VertexFog, FragData.WorldPosition);
#endif//_VERTEX_FOG

	return OutColor;
}
#endif //PBS_PIXEL_INCLUDE