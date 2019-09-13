#ifndef PBS_PIXELADD_INCLUDE
#define PBS_PIXELADD_INCLUDE 


#include "MaterialTemplate.hlsl"
#include "Lighting.hlsl" 
#include "Debug.hlsl" 
//#if MAX_DYNAMIC_POINT_LIGHTS > 0
//#if VARIABLE_NUM_DYNAMIC_POINT_LIGHTS
//int NumDynamicPointLights;
//#endif
//float4 LightPositionAndInvRadius[MAX_DYNAMIC_POINT_LIGHTS];
//float4 LightColorAndFalloffExponent[MAX_DYNAMIC_POINT_LIGHTS];
//#endif

FLOAT4 fragAdd(FInterpolantsVSToPS Interpolants, in FLOAT4 SvPosition : SV_Position) : SV_Target
{
	#if defined(_VOXEL_LIGHT)||defined(LIGHTMAP_ON)||defined(_CUSTOM_LIGHTMAP_ON)||defined(DIRECTIONAL)
		return FLOAT4(0,0,0,0);
	#else
		DEBUG_PBS_CUSTOMDATA
		FFragData FragData = GetFragData(Interpolants, SvPosition);
		FMaterialData MaterialData = GetMaterialData(FragData);
		FLightingData LightingData = GetLighting(FragData, MaterialData DEBUG_PBS_PARAM);

		FLOAT3 DiffuseGI;
		FLOAT3 Color = 0;

		//AO
		//Color *= MaterialData.AmbientOcclusion;
		//LightingData.IndirectIrradiance *= MaterialData.AmbientOcclusion;

		FLOAT3 DirectLighting = GetDirectLighting(FragData, MaterialData, LightingData DEBUG_PBS_PARAM);//direct diffuse
		Color += DirectLighting;
	// #if !defined(LIGHTMAP_ON) && !defined(_CUSTOM_LIGHTMAP_ON) && !defined(_PBS_NO_IBL)
	// 	FLOAT3 ImageBasedReflectionLighting = GetImageBasedReflectionLighting(FragData,MaterialData, LightingData DEBUG_PBS_PARAM);//gi(specular+sh)
	// 	Color += ImageBasedReflectionLighting;
	// #endif

		#if ENABLE_SKY_LIGHT
			//@mw todo
			// TODO: Also need to do specular.
			Color += GetSkySHDiffuseSimple(MaterialParameters.WorldNormal) * ResolvedView.SkyLightColor.rgb * DiffuseColor;
		#endif

		//EMISSIVE
	// #ifdef _ETX_EFFECT
	// 	Color += MaterialData.EmissiveAOColor.rgb;
	// #endif
		FLOAT4 OutColor = FLOAT4(Color, MaterialData.BaseColor.a);

	// #ifdef _VERTEX_FOG
	// 	OutColor.rgb = lerp(OutColor.rgb, FragData.VertexFog.rgb, FragData.VertexFog.a);
	// #endif


		DEBUG_PBS_COLOR(OutColor, FragData, MaterialData, LightingData)
		return OutColor;
	#endif
}
#endif //PBS_PIXELADD_INCLUDE