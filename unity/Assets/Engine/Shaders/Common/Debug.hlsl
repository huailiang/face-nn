
#ifndef PBS_DEBUG_INCLUDE
#define PBS_DEBUG_INCLUDE

#if defined(DEBUG_APP)&&!defined(SHADER_API_MOBILE)
#include "MaterialTemplate.hlsl" 
#include "Lighting.hlsl" 

//DEBUG_START
//vertex data
#define Debug_None 1
#define Debug_uv (Debug_None+1)
#define Debug_uv2 (Debug_uv+1)
#define Debug_VertexNormal (Debug_uv2+1)
#define Debug_VertexTangent (Debug_VertexNormal+1)
#define Debug_VertexColor (Debug_VertexTangent+1)

//vs 2 ps
#define Debug_VertexFog (Debug_VertexColor+1)
#define Debug_LightmapUV (Debug_VertexFog+1)
#define Debug_SvPosition (Debug_LightmapUV+1)
#define Debug_ScreenPosition (Debug_SvPosition+1)
#define Debug_WorldPosition (Debug_ScreenPosition+1)
#define Debug_WorldPosition_CamRelative (Debug_WorldPosition+1)
#define Debug_CameraVector (Debug_WorldPosition_CamRelative+1)
//material
#define Debug_BaseColor (Debug_CameraVector+1)
#define Debug_BaseColorAlpha (Debug_BaseColor+1)
#define Debug_BlendTex (Debug_BaseColorAlpha+1)
#define Debug_Metallic (Debug_BlendTex+1)
#define Debug_Roughness (Debug_Metallic+1)
#define Debug_WorldNormal (Debug_Roughness+1)
#define Debug_TangentSpaceNormal (Debug_WorldNormal+1)
#define Debug_ReflectionVector (Debug_TangentSpaceNormal+1)
//lighting
#define Debug_DiffuseColor (Debug_ReflectionVector+1)
#define Debug_SpecularScale (Debug_DiffuseColor+1)
#define Debug_SpecularColor (Debug_SpecularScale+1)
#define Debug_AmbientDiffuse (Debug_SpecularColor+1)
#define Debug_NdotL (Debug_AmbientDiffuse+1)
#define Debug_FixNdotL (Debug_NdotL+1)
#define Debug_NdotC (Debug_FixNdotL+1)
#define Debug_VdotH (Debug_NdotC+1)
#define Debug_Shadow (Debug_VdotH+1)
#define Debug_DirectLighting (Debug_Shadow+1)
#define Debug_PointLighting (Debug_DirectLighting+1)
#define Debug_LightMapColor (Debug_PointLighting+1)
#define Debug_LightMapGI (Debug_LightMapColor+1)
#define Debug_LightMapShadowMask (Debug_LightMapGI+1)
#define Debug_Lighting0 (Debug_LightMapShadowMask+1)
#define Debug_Lighting1 (Debug_Lighting0+1)
#define Debug_DirectDiffuseNoGI (Debug_Lighting1+1)
#define Debug_DirectDiffuse (Debug_DirectDiffuseNoGI+1)
#define Debug_DirectSpecular (Debug_DirectDiffuse+1)
#define Debug_DirectLightingColor (Debug_DirectSpecular+1)
#define Debug_CubeMipmap (Debug_DirectLightingColor+1)
#define Debug_ImageBasedReflectionLighting (Debug_CubeMipmap+1)
#define Debug_RimLight (Debug_ImageBasedReflectionLighting+1)
//skin
#define Debug_LookupDiffuseSpec (Debug_RimLight+1)
#define Debug_SpecLevel (Debug_LookupDiffuseSpec+1)
#define Debug_Translucency (Debug_SpecLevel+1)
#define Debug_ReflCube (Debug_Translucency+1)
//water
#define Debug_WaterNormal (Debug_ReflCube+1)
#define Debug_WaterScatterColor (Debug_WaterNormal+1)
#define Debug_WaterEmissionColor (Debug_WaterScatterColor+1)
//DEBUG_END

FLOAT4 DebugOutputColor(FLOAT4 OutColor,FFragData FragData,FMaterialData MaterialData,FLightingData LightingData, FCustomData CustomData)
{
	FLOAT4 debugColor = OutColor;
	uint debugMode = _DebugMode;
	if(_GlobalDebugMode>0)
	{
		debugMode = _GlobalDebugMode;
	}
	if (debugMode<Debug_None)
	{
		return OutColor;
	}
	else if (debugMode < Debug_uv)
	{
		debugColor = FLOAT4(GET_FRAG_UV, 0, 1);
	}
	else if (debugMode < Debug_uv2)
	{
		debugColor = FLOAT4(GET_FRAG_UV2, 0, 1);
	}
	else if (debugMode < Debug_VertexNormal)
	{
		debugColor = FLOAT4(FragData.TangentToWorld[2], 1);
	}
	else if (debugMode < Debug_VertexTangent)
	{
		debugColor = FLOAT4(FragData.TangentToWorld[0], 1);
	}
	else if (debugMode < Debug_VertexColor)
	{
		debugColor = FragData.VertexColor;
	}
	else if (debugMode < Debug_VertexFog)
	{
		debugColor = FLOAT4(FragData.VertexFog.rrr,1);
	}
	else if (debugMode < Debug_LightmapUV)
	{
		debugColor = FLOAT4(GET_FRAG_LIGTHMAP_UV, 0, 1);	
	}
	else if (debugMode < Debug_SvPosition)
	{
		debugColor = FragData.SvPosition;
	}
	else if (debugMode < Debug_ScreenPosition)
	{
		debugColor = FragData.ScreenPosition;
	}
	else if (debugMode < Debug_WorldPosition)
	{
		debugColor = FLOAT4(FragData.WorldPosition, 1);
	}
	else if (debugMode < Debug_WorldPosition_CamRelative)
	{
		debugColor = FLOAT4(FragData.WorldPosition_CamRelative, 1);
	}
	else if (debugMode < Debug_CameraVector)
	{
		debugColor = FLOAT4(FragData.CameraVector, 1);
	}
	else if (debugMode < Debug_BaseColor)
	{
		debugColor = MaterialData.BaseColor;
	}
	else if (debugMode < Debug_BaseColorAlpha)
	{
		debugColor = FLOAT4(MaterialData.BaseColor.aaa,1);
	}
	else if (debugMode < Debug_BlendTex)
	{
		debugColor = MaterialData.BlendTex;
	}	
	else if (debugMode < Debug_Metallic)
	{
		debugColor = FLOAT4(MaterialData.Metallic.xxx, 1);
	}
	else if (debugMode < Debug_Roughness)
	{
		debugColor = FLOAT4(MaterialData.Roughness.xxx, 1);
	}
	else if (debugMode < Debug_WorldNormal)
	{
		debugColor = FLOAT4(MaterialData.WorldNormal, 1);
	}

	else if (debugMode < Debug_TangentSpaceNormal)
	{
		debugColor = FLOAT4(MaterialData.TangentSpaceNormal, 1);
	}
	else if (debugMode < Debug_ReflectionVector)
	{
		debugColor = FLOAT4(MaterialData.ReflectionVector, 1);
	}
	else if (debugMode < Debug_DiffuseColor)
	{
		debugColor = FLOAT4(LightingData.DiffuseColor, 1);
	}
	else if (debugMode < Debug_SpecularScale)
	{
#ifndef _FULL_SSS
		debugColor = FLOAT4(CustomData.SpecularScale.xxx, 1);
#else
		debugColor = FLOAT4(0, 0, 0, 1);
#endif
	}	
	else if (debugMode < Debug_SpecularColor)
	{
		debugColor = FLOAT4(LightingData.SpecularColor, 1);
	}
	else if (debugMode < Debug_AmbientDiffuse)
	{
		debugColor = FLOAT4(CustomData.AmbientDiffuse, 1);
	}
	else if (debugMode < Debug_NdotL)
	{
		debugColor = FLOAT4(LightingData.NdotL.xxx, 1);
	}
	else if (debugMode < Debug_FixNdotL)
	{
		debugColor = FLOAT4(LightingData.FixNdotL.xxx, 1);
	}	
	else if (debugMode < Debug_NdotC)
	{
		debugColor = FLOAT4(LightingData.NdotC.xxx, 1);
	}
	else if (debugMode < Debug_VdotH)
	{
		debugColor = FLOAT4(LightingData.VdotH.xxx, 1);
	}	
	else if (debugMode < Debug_Shadow)
	{
		debugColor = FLOAT4(CustomData.Shadow.xxx, 1);
	}	
	else if (debugMode < Debug_DirectLighting)
	{
		debugColor = FLOAT4(CustomData.DirectLighting, 1);
	}
	else if (debugMode < Debug_PointLighting)
	{
		debugColor = FLOAT4(LightingData.pointLighting, 1);
	}	
	else if (debugMode < Debug_LightMapColor)
	{
		debugColor = FLOAT4(CustomData.LightMapColor, 1);
	}
	else if (debugMode < Debug_LightMapGI)
	{
		debugColor = FLOAT4(CustomData.LightMapGI.xyz,1);
	}	
	else if (debugMode < Debug_LightMapShadowMask)
	{
		debugColor = FLOAT4(CustomData.LightMapGI.www,1);
	}
	else if (debugMode < Debug_Lighting0)
	{
		debugColor = FLOAT4(LightingData.lighting0, 1);
	}
	else if (debugMode < Debug_Lighting1)
	{
	#ifdef _DOUBLE_LIGHTS
		debugColor = FLOAT4(LightingData.lighting1, 1);
	#else
		debugColor = FLOAT4(0,0,0, 1);
	#endif
	}	
	else if (debugMode < Debug_DirectDiffuseNoGI)
	{
		debugColor = FLOAT4(CustomData.DirectDiffuseNoGI, 1);
	}	
	else if (debugMode < Debug_DirectDiffuse)
	{
		debugColor = FLOAT4(CustomData.DirectDiffuse, 1);
	}
	else if (debugMode < Debug_DirectSpecular)
	{
		debugColor = FLOAT4(CustomData.DirectSpecular, 1);
	}
	else if (debugMode < Debug_DirectLightingColor)
	{
		debugColor = FLOAT4(CustomData.DirectLightingColor, 1);
	}
	else if (debugMode < Debug_CubeMipmap)
	{
		if (CustomData.CubeMipmap < 1)
		{
			debugColor = FLOAT4(0, 0, 0, 1);//black
		}
		else if (CustomData.CubeMipmap < 2)
		{
			debugColor = FLOAT4(1, 0, 0, 1);//red
		}
		else if (CustomData.CubeMipmap < 3)
		{
			debugColor = FLOAT4(1, 0.5, 0, 1);//orange
		}
		else if (CustomData.CubeMipmap < 4)
		{
			debugColor = FLOAT4(1, 1, 0, 1);//yellow
		}
		else if (CustomData.CubeMipmap < 5)
		{
			debugColor = FLOAT4(0, 1, 0, 1);//green
		}
		else if (CustomData.CubeMipmap < 6)
		{
			debugColor = FLOAT4(0, 1, 1, 1);//cyan
		}
		else if (CustomData.CubeMipmap < 7)
		{
			debugColor = FLOAT4(0, 0, 1, 1);//blue
		}
		else if (CustomData.CubeMipmap < 8)
		{
			debugColor = FLOAT4(1, 0, 1, 1);//magenta
		}
		else if (CustomData.CubeMipmap < 9)
		{
			debugColor = FLOAT4(0.5, 0.5, 0.5, 1);//gray
		}
		else if (CustomData.CubeMipmap < 10)
		{
			debugColor = FLOAT4(1, 1, 1, 1);//white
		}
		debugColor = FLOAT4(CustomData.CubeMipmap.xxx, 1);
	}
	else if (debugMode < Debug_ImageBasedReflectionLighting)
	{
		debugColor = FLOAT4(CustomData.ImageBasedReflectionLighting, 1);
	}
	else if (debugMode < Debug_RimLight)
	{
		debugColor = FLOAT4(CustomData.RimLight, 1);
	}
	else if (debugMode < Debug_LookupDiffuseSpec)
	{
		debugColor = FLOAT4(CustomData.LookupDiffuseSpec, 1);
	}
	else if (debugMode < Debug_SpecLevel)
	{
		debugColor = FLOAT4(CustomData.SpecLevel, 1);
	}
	else if (debugMode < Debug_Translucency)
	{
		debugColor = FLOAT4(CustomData.Translucency, 1);
	}
	else if (debugMode < Debug_ReflCube)
	{
		debugColor = FLOAT4(CustomData.ReflCube, 1);
	}	
	else if (debugMode < Debug_WaterNormal)
	{
		debugColor = FLOAT4(CustomData.WaterNormal, 1);
	}	
	else if (debugMode < Debug_WaterScatterColor)
	{
		debugColor = FLOAT4(CustomData.WaterScatterColor, 1);
	}	
	else if (debugMode < Debug_WaterEmissionColor)
	{
		debugColor = FLOAT4(CustomData.WaterEmissionColor, 1);
	}	

	if(_DebugDisplayType==0)
	{
		FLOAT U = FragData.SvPosition.x/_ScreenParams.x * 2 - 1;
		FLOAT V = FragData.SvPosition.y/_ScreenParams.y * 2 - 1;
		FLOAT y = _SplitAngle.x*U + _SplitPos -V;
		y *= _SplitAngle.y;
		return  y<0?debugColor:OutColor;
	}
	else
	{
		return  debugColor;
	}
	
}

#endif//DEBUG_APP

#endif //PBS_DEBUG_INCLUDE