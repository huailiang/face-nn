#include "DebugHead.hlsl"

#ifndef PBS_SH_INCLUDE
#define PBS_SH_INCLUDE

// SH lighting environment
FLOAT4 unity_SHAr;
FLOAT4 unity_SHAg;
FLOAT4 unity_SHAb;
FLOAT4 _AmbientParam;

inline FLOAT3 SHEvalLinearL0L1(FLOAT4 normal)
{
	FLOAT3 x;

	// Linear (L1) + constant (L0) polynomial terms
	x.r = dot(unity_SHAr, normal);
	x.g = dot(unity_SHAg, normal);
	x.b = dot(unity_SHAb, normal);

	return x;
}

FLOAT4 unity_SHBr;
FLOAT4 unity_SHBg;
FLOAT4 unity_SHBb;
FLOAT4 unity_SHC;
inline FLOAT3 SHEvalLinearL2(FLOAT4 normal)
{
	FLOAT3 x1, x2;
	// 4 of the quadratic (L2) polynomials
	FLOAT4 vB = normal.xyzz * normal.yzzx;
	x1.r = dot(unity_SHBr, vB);
	x1.g = dot(unity_SHBg, vB);
	x1.b = dot(unity_SHBb, vB);

	// Final (5th) quadratic (L2) polynomial
	FLOAT vC = normal.x*normal.x - normal.y*normal.y;
	x2 = unity_SHC.rgb * vC;

	return x1 + x2;
}

// #ifdef _VERTEX_POINT_LIGHT
FLOAT4 unity_4LightPosX0;
FLOAT4 unity_4LightPosY0;
FLOAT4 unity_4LightPosZ0;
FLOAT4 unity_4LightAtten0;
FLOAT4 unity_LightColor[4];
// #endif//_VERTEX_POINT_LIGHT

inline FLOAT3 Shade4PointLights(
	FLOAT4 lightPosX, FLOAT4 lightPosY, FLOAT4 lightPosZ,
	FLOAT3 lightColor0, FLOAT3 lightColor1, FLOAT3 lightColor2, FLOAT3 lightColor3,
	FLOAT4 lightAttenSq,
	FLOAT3 pos, FLOAT3 normal)
{
	// to light vectors
	FLOAT4 toLightX = lightPosX - pos.x;
	FLOAT4 toLightY = lightPosY - pos.y;
	FLOAT4 toLightZ = lightPosZ - pos.z;
	// squared lengths
	FLOAT4 lengthSq = 0;
	lengthSq += toLightX * toLightX;
	lengthSq += toLightY * toLightY;
	lengthSq += toLightZ * toLightZ;
	// don't produce NaNs if some vertex position overlaps with the light
	lengthSq = max(lengthSq, 0.000001);

	// NdotL
	FLOAT4 ndotl = 0;
	ndotl += toLightX * normal.x;
	ndotl += toLightY * normal.y;
	ndotl += toLightZ * normal.z;
	// correct NdotL
	FLOAT4 corr = rsqrt(lengthSq);
	ndotl = max(FLOAT4(0, 0, 0, 0), ndotl * corr);
	// attenuation
	FLOAT4 atten = 1.0 / (1.0 + lengthSq * lightAttenSq);
	FLOAT4 diff = ndotl * atten;
	// final color
	FLOAT3 col = 0;
	col += lightColor0 * diff.x;
	col += lightColor1 * diff.y;
	col += lightColor2 * diff.z;
	col += lightColor3 * diff.w;
	return col;
}

//SH Lighting
inline FLOAT3 ShadeSHPerVertex(FLOAT3 normal, FLOAT3 ambient)
{
	ambient += SHEvalLinearL2(FLOAT4(normal, 1.0));
	return ambient;
}

inline FLOAT3 VertexPointLight(FLOAT3 WorldPosition, FLOAT3 WorldNormal)
{
	FLOAT3 ambient = 0;

// #ifdef _VERTEX_POINT_LIGHT
	ambient.rgb = Shade4PointLights(
		unity_4LightPosX0, unity_4LightPosY0, unity_4LightPosZ0,
		unity_LightColor[0].rgb, unity_LightColor[1].rgb, unity_LightColor[2].rgb, unity_LightColor[3].rgb,
		unity_4LightAtten0, WorldPosition, WorldNormal);
// #endif	
	return ambient;
}

inline FLOAT3 VertexGI(FLOAT3 WorldPosition, FLOAT3 WorldNormal)
{
	FLOAT3 ambient = 0;
	
	ambient.rgb = ShadeSHPerVertex(WorldNormal, ambient.rgb);

	return ambient;
}

FLOAT3 ShadeSHPerPixel(FLOAT3 normal, FLOAT3 ambient, FLOAT3 worldPos)
{
	FLOAT3 ambient_contrib = 0.0;

#ifdef _VERTEX_GI
	// Completely per-pixel
	ambient_contrib = SHEvalLinearL0L1(FLOAT4(normal, 1.0));

	ambient = max(_AmbientParam.zzz, ambient + ambient_contrib);
	
#endif

	return ambient;
}
#endif //PBS_SH_INCLUDE