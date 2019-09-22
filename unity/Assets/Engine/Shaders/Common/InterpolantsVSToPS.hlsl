#ifndef PBS_INTERPOLANTSVSTOPS_INCLUDE
#define PBS_INTERPOLANTSVSTOPS_INCLUDE
#include "SH.hlsl"
#include "Fog.hlsl"
#include "PCH.hlsl"

#ifdef _UV_SCALE
FLOAT4 _uvST;
	#ifdef _UV_SCALE2
	FLOAT4 _uvST1;
	#endif//_UV_SCALE2
#endif//_UV_SCALE

#ifdef _SHADOW_MAP
FLOAT4x4 _ShadowMapViewProj;
FLOAT4x4 _ShadowMapViewProj1;
#endif//_SHADOW_MAP

#ifdef _WORLDSPACE_UV
FLOAT4 _wpOffset;
#endif//_WORLDSPACE_UV

// Used for vertex factory shaders which need to use the resolved view
FLOAT4 SvPositionToResolvedScreenPosition(FLOAT4 SvPosition)
{
	FLOAT2 PixelPos = SvPosition.xy;// -ResolvedView.ViewRectMin.xy;

	// NDC (NormalizedDeviceCoordinates, after the perspective divide)
	FLOAT2 ViewSizeAndInvSize = FLOAT2(_ScreenParams.z - 1.0f, _ScreenParams.w - 1.0f);
	FLOAT3 NDCPos = FLOAT3((PixelPos * ViewSizeAndInvSize - 0.5f) * FLOAT2(2, -2), SvPosition.z);

	// SvPosition.w: so .w has the SceneDepth, some mobile code and the DepthFade material expression wants that
	return FLOAT4(NDCPos.xyz, 1) * SvPosition.w;
}


// Transforms direction from object to world space
inline FLOAT3 ObjectToWorldDir(in FLOAT3 dir)
{
	return normalize(mul((FLOAT3x3)_objectToWorld, dir));
}

// Transforms normal from object to world space
inline FLOAT3 ObjectToWorldNormal(in FLOAT3 norm)
{
#ifdef UNITY_ASSUME_UNIFORM_SCALING
	return ObjectToWorldDir(norm);
#else
	return normalize(mul(norm, (FLOAT3x3)_worldToObject));
#endif
}

FInterpolantsVSToPS GetInterpolantsVSToPS(FVertexInput Input, FLOAT4 WorldPosition)
{
	DECLARE_OUTPUT(FInterpolantsVSToPS, Interpolants);

	SET_UV(Input.uv0);
	SET_UV2(Input.uv0);
	SET_BACKUP_UV(Input.uv0);

#if defined(LIGHTMAP_ON)||defined(_CUSTOM_LIGHTMAP_ON)
	#if defined(_WORLDSPACE_UV)
		FLOAT2 uv = (WorldPosition.xz - _wpOffset.xy)*_wpOffset.zw;
	#else
		FLOAT2 uv = Input.uv2;
	#endif
	SET_LIGTHMAP_UV(uv);
#endif//((LIGHTMAP_ON)||(_CUSTOM_LIGHTMAP_ON))
	
	FLOAT3 WorldNormal = ObjectToWorldNormal(Input.TangentX);
	FLOAT3x3 TangentToWorld;
	FLOAT TangentSign = 0;
#ifndef _PBS_FROM_PARAM
	FLOAT4 WorldTangent = FLOAT4(ObjectToWorldDir(Input.TangentZ.xyz), Input.TangentZ.w);	
	TangentSign = WorldTangent.w * unity_WorldTransformParams.w;
	FLOAT3 WorldBinormal = cross(WorldNormal, WorldTangent.xyz) * TangentSign;
	TangentToWorld = FLOAT3x3(WorldTangent.xyz, WorldBinormal, WorldNormal);	
#else//_PBS_FROM_PARAM
	TangentToWorld[0].xyz = 0;
	TangentToWorld[1].xyz = 0;
	TangentToWorld[2].xyz = WorldNormal;
#endif//_PBS_FROM_PARAM
	Interpolants.TangentToWorld0 = FLOAT4(TangentToWorld[0],0);
	Interpolants.TangentToWorld2 = FLOAT4(TangentToWorld[2], TangentSign);

#if defined(_OUTPUT_VERTEX_COLOR)
	#ifdef _VERTEX_COLOR
		Interpolants.Color = Input.Color;
	#endif//_VERTEX_COLOR
#endif//_OUTPUT_VERTEX_COLOR

#ifdef _VERTEX_GI
	Interpolants.VertexGI.xyz = VertexGI(WorldPosition.xyz,WorldNormal);
#endif//_VERTEX_GI

#ifdef _GRASS_LIGHT
	FLOAT occ = saturate(Input.Position.y * _Occ_Height);
    Interpolants.VertexGI.w = pow(occ, _Occ_Power);
#endif//_GRASS_LIGHT

#ifdef _SHADOW_MAP
	Interpolants.ShadowCoord.xyz = mul(_ShadowMapViewProj, FLOAT4(WorldPosition.xyz,1)).xyz;
	Interpolants.ShadowCoord.xy = Interpolants.ShadowCoord.xy*0.5f+FLOAT2(0.5,0.5);
	#ifdef _SHADOW_MAP_CSM
		FLOAT3 coord = mul(_ShadowMapViewProj1, FLOAT4(WorldPosition.xyz,1)).xyz;
		coord.xy = coord.xy*0.5f+FLOAT2(0.5,0.5);
		Interpolants.ShadowCoord.w = coord.z;
		Interpolants.ShadowCoordCSM.xy = coord.xy;
	#endif//_SHADOW_MAP_CSM
#endif//_SHADOW_MAP

	return Interpolants;
}

#endif //PBS_INTERPOLANTSVSTOPS_INCLUDE