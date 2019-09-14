#ifndef PBS_VERTEX_INCLUDE
#define PBS_VERTEX_INCLUDE 

#include "Head.hlsl"
#include "InterpolantsVSToPS.hlsl"
#include "Instance.hlsl"

struct FMobileShadingVSToPS
{
	FInterpolantsVSToPS Interpolants;
	FLOAT4 Position : SV_POSITION;
};
#ifdef _TERRAIN_LODCULL
FLOAT4 _TerrainLodCull;
#endif

void vertForwardBase(FVertexInput Input
	INSTANCE_INPUT,
out FMobileShadingVSToPS Output)
{  
	INITIALIZE_OUTPUT(FMobileShadingVSToPS, Output);

#if _ROT90_POS
	Input.Position = mul(obj2World, Input.Position);
#endif

#ifdef _TERRAIN_LODCULL
	FLOAT inRangeMask = InRange(Input.uv0,_TerrainLodCull);
	if(inRangeMask>0.99)
	{
		Output.Position.z = -100;
		return;
	}
#endif//_TERRAIN_LODCULL
	
	FLOAT4 WorldPosition = INSTANCE_WPOS(Input.Position)
	Output.Position = TransformWorldToClipPos(WorldPosition);

	Output.Interpolants = GetInterpolantsVSToPS(Input, WorldPosition);
#ifdef _SCREEN_POS
	FLOAT3 pos = ComputeScreenPos(Output.Position).xyw;
	Output.Interpolants.ScreenPosition.xy = pos.xy;
	Output.Interpolants.ScreenPositionW.x = pos.z;
	FLOAT3 posGrab = ComputeGrabScreenPos(Output.Position).xyw;
	Output.Interpolants.ScreenPosition.zw = posGrab.xy;
	Output.Interpolants.ScreenPositionW.y = posGrab.z;
#endif//_SCREEN_POS
	Output.Interpolants.WorldPosition = WorldPosition;
	Output.Interpolants.WorldPosition.w = Output.Position.w;
} 

#endif //PBS_VERTEX_INCLUDE