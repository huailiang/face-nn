#ifndef PBS_HEAD_INCLUDE
#define PBS_HEAD_INCLUDE

#include "Common.hlsl"
// #include "ShadowLib.hlsl"
// #ifdef _CUSTOM_UV_LAYOUT
// 	#define _INPUT_UV0
// #else//!_CUSTOM_UV_LAYOUT
	#define _INPUT_UV0
	#if (defined(LIGHTMAP_ON)||defined(_CUSTOM_LIGHTMAP_ON))&&!defined(_TERRAIN)
		#define _INPUT_UV2		
	#endif//((LIGHTMAP_ON)||(_CUSTOM_LIGHTMAP_ON))&&!defined(_TERRAIN)
	
	#if defined(LIGHTMAP_ON)||defined(_CUSTOM_LIGHTMAP_ON)
		#define SET_LIGTHMAP_UV(uv) Interpolants.TexCoords[0].zw = (uv) * unity_LightmapST.xy + unity_LightmapST.zw
		#define GET_FRAG_LIGTHMAP_UV FragData.TexCoords[0].zw
		#ifdef _UV_SCALE2
			#define _OUTPUT_UV_COUNT 2
			#define SET_UV2(uv) Interpolants.TexCoords[1].xy = (uv) * _uvST1.xy + _uvST1.zw
			#define GET_FRAG_UV2 FragData.TexCoords[1].xy
			#ifdef _BACKUP_UV
				#define SET_BACKUP_UV(uv) Interpolants.TexCoords[1].zw = (uv)
				#define GET_FRAG_BACKUP_UV FragData.TexCoords[1].zw
			#else//!_BACKUP_UV
				#define SET_BACKUP_UV(uv)
				#define GET_FRAG_BACKUP_UV FLOAT2(0,0)
			#endif//_BACKUP_UV
		#else//!_UV_SCALE2
			#define _OUTPUT_UV_COUNT 1
			#define SET_UV2(uv)
			#define SET_BACKUP_UV(uv)
			#define GET_FRAG_UV2 FragData.TexCoords[0].xy
			#define GET_FRAG_BACKUP_UV FLOAT2(0,0)
		#endif//_UV_SCALE2
	#else//!((LIGHTMAP_ON)||(_CUSTOM_LIGHTMAP_ON))
		#define SET_LIGTHMAP_UV(uv)
		#define GET_FRAG_LIGTHMAP_UV FLOAT2(0,0)
		#ifdef _UV_SCALE2
			#define SET_UV2(uv) Interpolants.TexCoords[0].zw = (uv) * _uvST1.xy + _uvST1.zw
			#define GET_FRAG_UV2 FragData.TexCoords[0].zw
			#ifdef _BACKUP_UV
				#define _OUTPUT_UV_COUNT 2				
				#define SET_BACKUP_UV(uv) Interpolants.TexCoords[1].xy = (uv)
				#define GET_FRAG_BACKUP_UV FragData.TexCoords[1].xy
			#else//!_BACKUP_UV
				#define _OUTPUT_UV_COUNT 1
				#define SET_BACKUP_UV(uv)
				#define GET_FRAG_BACKUP_UV FLOAT2(0,0)
			#endif//_BACKUP_UV	
		#else//!_UV_SCALE2
			#define _OUTPUT_UV_COUNT 1
			#define SET_UV2(uv)
			#define SET_BACKUP_UV(uv)
			#define GET_FRAG_UV2 FragData.TexCoords[0].xy
			#define GET_FRAG_BACKUP_UV FLOAT2(0,0)
		#endif//_UV_SCALE2
	#endif//((LIGHTMAP_ON)||(_CUSTOM_LIGHTMAP_ON))

	#ifdef _UV_SCALE
		#define SET_UV(uv) Interpolants.TexCoords[0].xy = (uv) * _uvST.xy + _uvST.zw
	#else//!_UV_SCALE
		#define SET_UV(uv) Interpolants.TexCoords[0].xy = (uv)		
	#endif//_UV_SCALE
	#define GET_FRAG_UV FragData.TexCoords[0].xy

// #endif//_CUSTOM_UV_LAYOUT

struct FVertexInput
{  
	FLOAT4	Position	: POSITION;
	FLOAT3	TangentX	: NORMAL;
	FLOAT4	TangentZ 	: TANGENT;

#ifdef _INPUT_UV0 
	FLOAT2	uv0 : TEXCOORD0;
#endif//_INPUT_UV

#ifdef _INPUT_UV2 
	FLOAT2	uv2 : TEXCOORD1;
#endif//_INPUT_UV2

#ifdef _VERTEX_COLOR
	FLOAT4	Color : COLOR;
#endif //_VERTEX_COLOR

}; 

struct FInterpolantsVSToPS
{
	FLOAT4 TangentToWorld0 : TANGENTTOWORLD0;
	FLOAT4 TangentToWorld2 : TANGENTTOWORLD2;
	
#ifdef _VERTEX_COLOR
	FLOAT4	Color : COLOR0;
#endif//_VERTEX_COLOR

#if _OUTPUT_UV_COUNT
	FLOAT4	TexCoords[_OUTPUT_UV_COUNT]	: TEXCOORD0;
#endif//_OUTPUT_UV_COUNT
	
#ifdef _VERTEX_GI
	FLOAT4	VertexGI : TEXCOORD2;
#endif//_VERTEX_GI

// #ifdef _VERTEX_FOG
// 	FLOAT4 VertexFog : TEXCOORD3;
// #endif//_VERTEX_FOG

	FLOAT4 WorldPosition : TEXCOORD4; // xyz = world position, w = clip z
#ifdef _SCREEN_POS
	FLOAT4 ScreenPosition : TEXCOORD5;
	FLOAT2 ScreenPositionW : TEXCOORD6;
#endif
#ifdef _SHADOW_MAP
	FLOAT4 ShadowCoord : TEXCOORD7; 
	#ifdef _SHADOW_MAP_CSM
		FLOAT2 ShadowCoordCSM : TEXCOORD8;
	#endif
#endif
};

#define _HAS_SKIN (_FULL_SSS || _PART_SSS)

struct FFragData
{
#if _OUTPUT_UV_COUNT
	FLOAT4	TexCoords[_OUTPUT_UV_COUNT];
#endif//_OUTPUT_UV_COUNT

	FLOAT4 SvPosition;
	/** Post projection position reconstructed from SvPosition, before the divide by W. left..top -1..1, bottom..top -1..1  within the viewport, W is the SceneDepth */
	FLOAT4 ScreenPosition;
	FLOAT3 WorldPosition;
	FLOAT3 WorldPosition_CamRelative;
	FLOAT4 VertexColor;
	FLOAT3 Ambient;	
	// FLOAT3 VertexPointLight;
	FLOAT3 CameraVector;//viewDIR
	FLOAT3x3 TangentToWorld;
	FLOAT4 VertexFog;
	FLOAT GrassOcc;
	FLOAT3 ShadowCoord;
	FLOAT3 ShadowCoordCSM;
	FLOAT4 Parallax;

};

struct FMaterialData
{
	//common data
	FLOAT3 WorldNormal;
	FLOAT3 TangentSpaceNormal;
	FLOAT4 BaseColor;
	FLOAT4 BlendTex;
	FLOAT3 BlendColor;
	FLOAT MetallicScale;
	FLOAT4 SrcPbs;
	FLOAT Roughness;	

#if defined(_TERRAIN_WATER)
	float WaterHeight;
#endif

	FLOAT3 ReflectionVector;	
	FLOAT Metallic;
#ifdef _ETX_EFFECT
	FLOAT4 EmissiveAO;
#endif//_ETX_EFFECT
	FLOAT Shadow;
};

struct FLightingData
{	
    FLOAT3 DirectLightDir;
	FLOAT3 DirectLightColor;
	FLOAT3 lighting0;
	FLOAT NdotL;//normal dot lightdir	
	FLOAT FixNdotL;
	FLOAT NdotC;//normal dot camera	
	FLOAT3 H;//Camera + light dir
	FLOAT NdotH;//normal dot H	
	FLOAT VdotH;//normal dot H	
	FLOAT3 DiffuseColor;
	FLOAT3 SpecularColor;
	FLOAT Shadow;
	FLOAT3 IndirectDiffuseLighting;
	FLOAT3 IndirectPointLight;
#ifdef _DOUBLE_LIGHTS
	FLOAT3 DirectLightColor1;
	FLOAT3 lighting1;
	FLOAT NdotL1;
	FLOAT FixNdotL1;	
	FLOAT3 H1;
	FLOAT NdotH1;
	FLOAT VdotH1;//normal dot H	
#endif//_DOUBLE_LIGHTS
	FLOAT3 pointLighting;
	FLOAT4 gi;
};

#endif //PBS_HEAD_INCLUDE