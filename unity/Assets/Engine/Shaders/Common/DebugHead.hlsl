
#ifndef PBS_DEBUGHEAD_INCLUDE
#define PBS_DEBUGHEAD_INCLUDE

#if defined(DEBUG_APP)&&!defined(SHADER_API_MOBILE)
struct FCustomData
{
	FLOAT Shadow;
	FLOAT3 DirectLighting;
	FLOAT3 LightMapColor;
	FLOAT4 LightMapGI;
	FLOAT3 DirectDiffuse;
	FLOAT3 DirectDiffuseNoGI;
	FLOAT3 DirectSpecular;
	FLOAT3 DirectLightingColor;
	FLOAT3 ImageBasedReflectionLighting;
	FLOAT CubeMipmap;
	FLOAT3 RimLight;
	FLOAT3 LookupDiffuseSpec;
	FLOAT3 SpecLevel;
	FLOAT3 Translucency;
	FLOAT3 AmbientDiffuse;
	FLOAT SpecularScale;
	FLOAT3 ReflCube;
	FLOAT3 WaterNormal;
	FLOAT3 WaterScatterColor;
	FLOAT3 WaterEmissionColor;
};

FLOAT _DebugMode;
FLOAT _GlobalDebugMode;
uint _DebugDisplayType;
FLOAT2 _SplitAngle;
FLOAT _SplitPos;
#define DEBUG_PBS_COLOR(OutColor, FragData, MaterialData, LightingData) OutColor = DebugOutputColor(OutColor,FragData, MaterialData, LightingData,CustomData);
#define DEBUG_PBS_CUSTOMDATA DECLARE_OUTPUT(FCustomData, CustomData);
#define DEBUG_PBS_CUSTOMDATA_PARAM(FieldName,FieldValue) CustomData.##FieldName = FieldValue;
#define DEBUG_PBS_ARGS ,inout FCustomData CustomData
#define DEBUG_PBS_PARAM ,CustomData
#else//!DEBUG_APP
#define DEBUG_PBS_COLOR(OutColor, FragData, MaterialData, LightingData)
#define DEBUG_PBS_CUSTOMDATA 
#define DEBUG_PBS_CUSTOMDATA_PARAM(FieldName,FieldValue)
#define DEBUG_PBS_ARGS
#define DEBUG_PBS_PARAM
#endif//DEBUG_APP

#endif //PBS_DEBUG_INCLUDE