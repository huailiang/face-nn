// Copyright 2018- PWRD, Inc. All Rights Reserved.
#ifndef PBS_LIGHTING_FOR_TREE_INCLUDE
#define PBS_LIGHTING_FOR_TREE_INCLUDE
#ifdef _TREE_LIGHT_

void TreeShadingMode(FFragData FragData, FMaterialData MaterialData, FLightingData LightingData,inout FLOAT3 DirectDiffuse,inout FLOAT3 DirectSpecular DEBUG_PBS_ARGS)
{
	DirectDiffuse =  LightingData.DiffuseColor * max(LightingData.DirectLightColor*(1-LightingData.FixNdotL) + LightingData.NdotC,_AmbineScale)*_AmbineScale;
	DirectDiffuse += LightingData.gi.xyz;
	DirectSpecular = FLOAT3(0,0,0);

	#ifndef _NO_DEFAULT_SPEC
		DirectSpecular = LightingData.SpecularColor * CalcSpecular(MaterialData.Roughness, LightingData.NdotH, LightingData.H, MaterialData.WorldNormal)*LightingData.lighting0;
		DirectSpecular = min(2, DirectSpecular);
	#endif
}
#endif//_TREE_LIGHT_
#endif //PBS_LIGHTING_FOR_TREE_INCLUDE