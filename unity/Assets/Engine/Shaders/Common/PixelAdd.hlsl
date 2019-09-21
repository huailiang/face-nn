#ifndef PBS_PIXELADD_INCLUDE
#define PBS_PIXELADD_INCLUDE 

#include "MaterialTemplate.hlsl"
#include "Lighting.hlsl" 
#include "Debug.hlsl" 


FLOAT4 fragAdd(FInterpolantsVSToPS Interpolants, in FLOAT4 SvPosition : SV_Position) : SV_Target
{
	#if defined(LIGHTMAP_ON)||defined(_CUSTOM_LIGHTMAP_ON)||defined(DIRECTIONAL)
		return FLOAT4(0,0,0,0);
	#else
		DEBUG_PBS_CUSTOMDATA
		FFragData FragData = GetFragData(Interpolants, SvPosition);
		FMaterialData MaterialData = GetMaterialData(FragData);
		FLightingData LightingData = GetLighting(FragData, MaterialData DEBUG_PBS_PARAM);

		FLOAT3 DiffuseGI;
		FLOAT3 Color = 0;
		FLOAT3 DirectLighting = GetDirectLighting(FragData, MaterialData, LightingData DEBUG_PBS_PARAM);//direct diffuse
		Color += DirectLighting;


		#if ENABLE_SKY_LIGHT
			// TODO: Also need to do specular.
			Color += GetSkySHDiffuseSimple(MaterialParameters.WorldNormal) * ResolvedView.SkyLightColor.rgb * DiffuseColor;
		#endif


		FLOAT4 OutColor = FLOAT4(Color, MaterialData.BaseColor.a);

		DEBUG_PBS_COLOR(OutColor, FragData, MaterialData, LightingData)
		return OutColor;
	#endif
}
#endif //PBS_PIXELADD_INCLUDE