#ifndef PBS_VERTEXADD_INCLUDE
#define PBS_VERTEXADD_INCLUDE 

#include "Head.hlsl"
#include "InterpolantsVSToPS.hlsl"
#include "Effect.hlsl"

struct FMobileShadingVSToPS
{
	FInterpolantsVSToPS Interpolants;
	FLOAT4 Position : SV_POSITION;
};

void vertAdd(FVertexInput Input, out FMobileShadingVSToPS Output)
{  
	INITIALIZE_OUTPUT(FMobileShadingVSToPS, Output);

	FLOAT4 WorldPosition = mul(unity_ObjectToWorld, Input.Position);
	WorldPosition = VertexEffect(Input,WorldPosition);
	Output.Position = TransformWorldToClipPos(WorldPosition);
	
	Output.Interpolants = GetInterpolantsVSToPS(Input, WorldPosition);
	Output.Interpolants.WorldPosition = WorldPosition;
	Output.Interpolants.WorldPosition.w = Output.Position.w;
} 

#endif //PBS_VERTEXADD_INCLUDE