// Copyright 2018- PWRD, Inc. All Rights Reserved.
#ifndef PBS_WATER_LIGHT_INCLUDE
#define PBS_WATER_LIGHT_INCLUDE

#ifdef _WATER_LIGHT
#define DEPTH_BASELINE 10.

// Geometry data
// x: A square is formed by 2 triangles in the mesh. Here x is square size
// yz: normalScrollSpeed0, normalScrollSpeed1
FLOAT4 _GeomData;
FLOAT4 _OceanCenterPosWorld;

TEXTURE2D_SAMPLER2D(_WaterNormal);

//x:_NormalsStrength("Strength", Range(0.01, 2.0)) = 0.3
//y:_NormalsScale("Scale", Range(0.01, 50.0)) = 1.0
//z:_SubSurfaceBase("Base Mul", Range(0.0, 2.0)) = 0.6
//w:_SubSurfaceSun("Sun Mul", Range(0.0, 10.0)) = 0.8
// FLOAT4 _Param0;
#define _NormalsStrength _Param0.x
#define _NormalsScale _Param0.y
#define _SubSurfaceBase _Param0.z
#define _SubSurfaceSun _Param0.w

//x:_SubSurfaceSunFallOff("Sun Fall-Off", Range(1.0, 16.0)) = 4.0
//y:_SubSurfaceHeightMax("Height Max", Range(0.0, 50.0)) = 3.0
//z:_SubSurfaceHeightPower("Height Power", Range(0.01, 10.0)) = 1.0
//w:_SubSurfaceDepthMax("Depth Max", Range(0.01, 50.0)) = 3.0
// FLOAT4 _Param1;
#define _SubSurfaceSunFallOff _Param1.x
#define _SubSurfaceHeightMax _Param1.y
#define _SubSurfaceHeightPower _Param1.z
#define _TransparencyWidth _Param1.w

//x:_SubSurfaceDepthPower("Depth Power", Range(0.01, 10.0)) = 1.0
//y:_FresnelPower("Fresnel Power", Range(0.0, 20.0)) = 3.0
//z:_DirectionalLightFallOff("Fall-Off", Range(1.0, 4096.0)) = 128.0
//w:_DirectionalLightBoost("Boost", Range(0.0, 512.0)) = 5.0
// FLOAT4 _Param2;
#define _TransparencyWidthPower _Param2.x
#define _FresnelPower _Param2.y
#define _DirectionalLightFallOff _Param2.z
#define _DirectionalLightBoost _Param2.w

//x:_RefractionStrength("Refraction Strength", Range(0.0, 1.0)) = 0.1
//y:_SceneDefaultDepth("scene Depth", Range(0.0, 1.0)) = 0.1
//y:_AlphaScale("scene Alpha", Range(0.0, 1.0)) = 0.5
// FLOAT4 _Param3;
#define _RefractionStrength _Param3.x
#define _SceneDefaultDepth _Param3.y
#define _AlphaScale _Param3.z
#define _ReflectionScale _Param3.w



// FLOAT4 _Param4;
#define _FlowSpeed FLOAT2(_Param4.x-5,_Param4.y-5)



// FLOAT4 _DepthFogDensity;
FLOAT4 _ScatteringColor;
FLOAT4 _SubSurfaceColor;
FLOAT4 _SubSurfaceCrestColor;

// FLOAT4 _SubSurfaceShallowColor;
FLOAT4 _SunDir;

TEXTURE2D_SAMPLER2D(_SceneRT);

inline FLOAT3 UnpackNormalMap(FLOAT4 packednormal)
{
	return packednormal.xyz * 2 - 1;
}

FLOAT2 SampleNormalMaps(FLOAT2 worldXZUndisplaced, FLOAT lodAlpha)
{
	const FLOAT2 v0 = FLOAT2(0.24, -0.34)+_FlowSpeed, v1 = FLOAT2(-0.25, -0.53)+_FlowSpeed;
	const FLOAT geomSquareSize = 0.25;//_GeomData.x;
	FLOAT nstretch = _NormalsScale * geomSquareSize; // normals scaled with geometry
	const FLOAT spdmulL = 0.68;//_GeomData.y;
	FLOAT2 norm =
		UnpackNormalMap(SAMPLE_TEXTURE2D(_WaterNormal, (v0*_Time.y*spdmulL + worldXZUndisplaced*0.7*FLOAT2(0.29,0.5)) / nstretch)).xy +
		UnpackNormalMap(SAMPLE_TEXTURE2D(_WaterNormal, (v1*_Time.y*spdmulL + worldXZUndisplaced*FLOAT2(0.4,0.7)) / nstretch)).xy;

	// approximate combine of normals. would be better if normals applied in local frame.
	return _NormalsStrength *  norm;
}

// void ApplyNormalMapsWithFlow(FLOAT2 worldXZUndisplaced, FLOAT2 flow, FLOAT lodAlpha, inout FLOAT3 io_n)
// {
// 	const FLOAT FLOAT_period = 1;
// 	const FLOAT period = FLOAT_period * 2;
// 	FLOAT sample1_offset = fmod(_CrestTime, period);
// 	FLOAT sample1_weight = sample1_offset / FLOAT_period;
// 	if (sample1_weight > 1.0) sample1_weight = 2.0 - sample1_weight;
// 	FLOAT sample2_offset = fmod(_CrestTime + FLOAT_period, period);
// 	FLOAT sample2_weight = 1.0 - sample1_weight;

// 	// In order to prevent flow from distorting the UVs too much,
// 	// we fade between two samples of normal maps so that for each
// 	// sample the UVs can be reset
// 	FLOAT2 io_n_1 = SampleNormalMaps(worldXZUndisplaced - (flow * sample1_offset), lodAlpha);
// 	FLOAT2 io_n_2 = SampleNormalMaps(worldXZUndisplaced - (flow * sample2_offset), lodAlpha);
// 	io_n.xz += sample1_weight * io_n_1;
// 	io_n.xz += sample2_weight * io_n_2;
// 	io_n = normalize(io_n);
// }
FLOAT3 ScatterColour(
	in FFragData FragData,in FMaterialData MaterialData,in FLightingData LightingData,
	in const FLOAT i_surfaceOceanDepth)
{
	FLOAT depth = i_surfaceOceanDepth;
	FLOAT waveHeight = 0.1f;//FragData.WorldPosition.y - _OceanCenterPosWorld.y;

	// base colour
	FLOAT3 col = _ScatteringColor.xyz;

	//SubsurfaceScattering

	//SubsurfaceShallow
	// FLOAT shallowness = pow(1. - saturate(depth), _SubSurfaceDepthPower);
	// col = lerp(col, _SubSurfaceShallowColor.xyz, shallowness);
	
	//SubsurfaceHeight
	col += pow(saturate(0.5 + 2.0 * waveHeight / _SubSurfaceHeightMax), _SubSurfaceHeightPower) * _SubSurfaceCrestColor.rgb;

	// // // light
	// // // use the constant term (0th order) of SH stuff - this is the average. it seems to give the right kind of colour
	col *= FLOAT3(0.33, 0.46, 0.73);

	// // // Approximate subsurface scattering - add light when surface faces viewer. Use geometry normal - don't need high freqs.
	FLOAT towardsSun = pow(max(0., dot( _SunDir.xyz, reflect(-FragData.CameraVector, FLOAT3(0, 1, 0)))), _SubSurfaceSunFallOff);
	col += (_SubSurfaceBase + _SubSurfaceSun * towardsSun) * _SubSurfaceColor.rgb * LightingData.DirectLightColor;

	return col;
}

#if _CAUSTICS_ON
void ApplyCaustics(in const FLOAT3 i_view, in const FLOAT3 i_lightDir, in const FLOAT i_sceneZ, in sampler2D i_normals, inout FLOAT3 io_sceneColour)
{
	// could sample from the screen space shadow texture to attenuate this..
	// underwater caustics - dedicated to P
	FLOAT3 camForward = mul((FLOAT3x3)unity_CameraToWorld, FLOAT3(0., 0., 1.));
	FLOAT3 scenePos = _WorldSpaceCameraPos - i_view * i_sceneZ / dot(camForward, -i_view);
	const FLOAT2 scenePosUV = LD_1_WorldToUV(scenePos.xz);
	FLOAT3 disp = 0.;
	// this gives height at displaced position, not exactly at query position.. but it helps. i cant pass this from vert shader
	// because i dont know it at scene pos.
	SampleDisplacements(_LD_Sampler_AnimatedWaves_1, scenePosUV, 1.0, disp);
	FLOAT waterHeight = _OceanCenterPosWorld.y + disp.y;
	FLOAT sceneDepth = waterHeight - scenePos.y;
	FLOAT bias = abs(sceneDepth - _CausticsFocalDepth) / _CausticsDepthOfField;
	// project along light dir, but multiply by a fudge factor reduce the angle bit - compensates for fact that in real life
	// caustics come from many directions and don't exhibit such a strong directonality
	FLOAT2 surfacePosXZ = scenePos.xz + i_lightDir.xz * sceneDepth / (4.*i_lightDir.y);
	FLOAT2 causticN = _CausticsDistortionStrength * UnpackNormal(tex2D(i_normals, surfacePosXZ / _CausticsDistortionScale)).xy;
	FLOAT4 cuv1 = FLOAT4((surfacePosXZ / _CausticsTextureScale + 1.3 *causticN + FLOAT2(0.044*_CrestTime + 17.16, -0.169*_CrestTime)), 0., bias);
	FLOAT4 cuv2 = FLOAT4((1.37*surfacePosXZ / _CausticsTextureScale + 1.77*causticN + FLOAT2(0.248*_CrestTime, 1.117*_CrestTime)), 0., bias);

	FLOAT causticsStrength = _CausticsStrength;
#if _SHADOWS_ON
	{
		// only sample the bigger lod. if pops are noticeable this could lerp the 2 lods smoothly, but i didnt notice issues.
		fixed2 causticShadow = 0.;
		FLOAT2 uv_1 = LD_1_WorldToUV(surfacePosXZ);
		SampleShadow(_LD_Sampler_Shadow_1, uv_1, 1.0, causticShadow);
		causticsStrength *= 1. - causticShadow.y;
	}
#endif // _SHADOWS_ON

	io_sceneColour *= 1. + causticsStrength *
		(0.5*tex2Dbias(_CausticsTexture, cuv1).x + 0.5*tex2Dbias(_CausticsTexture, cuv2).x - _CausticsTextureAverage);
}
#endif // _CAUSTICS_ON


FLOAT4 OceanEmission(
	in FFragData FragData,in FMaterialData MaterialData,in FLightingData LightingData,	
	in const FLOAT3 i_n_pixel, in const FLOAT i_pixelZ, in const FLOAT2 i_uvDepth, in const FLOAT i_sceneZ, in const FLOAT i_sceneZ01,
	in const FLOAT3 i_bubbleCol, in const FLOAT3 i_scatterCol)
{
	FLOAT4 col = FLOAT4(i_scatterCol,1);

	// underwater bubbles reflect in light
	col.rgb += i_bubbleCol;

	
#ifdef _TRANSPARENCY_ON
	// have we hit a surface? this check ensures we're not sampling an unpopulated backbuffer.
	// if (i_sceneZ01 != 0.0)
	FLOAT2 grabPos = FragData.ScreenPosition.zw;
	// view ray intersects geometry surface either above or below ocean surface

	FLOAT2 uvBackgroundRefract = grabPos.xy + _RefractionStrength * i_n_pixel.xz;
	FLOAT2 uvDepthRefract = i_uvDepth + _RefractionStrength * i_n_pixel.xz;
	
	FLOAT3 sceneColour = SAMPLE_TEXTURE2D(_SceneRT, uvBackgroundRefract).xyz;
	FLOAT3 alpha = 0;


	// depth fog & caustics - only if view ray starts from above water
	// if (!i_underwater)
	
		// if we haven't refracted onto a surface in front of the water surface, compute an alpha based on Z delta
		// FLOAT refractSceneZ = LinearEyeDepth(sceneColour.a);
		if (i_sceneZ > i_pixelZ)
		{
			FLOAT sceneZRefract = LinearEyeDepth(SAMPLE_TEXTURE2D(_SceneRT, uvDepthRefract).a);
			// if(sceneColour.sceneColour> i_pixelZ)
			{
				//FLOAT maxZ = max(i_sceneZ, sceneZRefract);
				FLOAT deltaZ = sceneZRefract - i_pixelZ;
				alpha = pow(saturate(deltaZ/_TransparencyWidth),1)*_AlphaScale;
				//col.rgb = FLOAT3(1,0,0);//lerp(sceneColour, col.rgb, alpha);
			}
		}
	#if _CAUSTICS_ON
		ApplyCaustics(i_view, i_lightDir, i_sceneZ, i_normals, sceneColour);
	#endif

	// blend from water colour to the scene colour
	col.rgb = lerp(sceneColour, col.rgb, alpha);
	col.a = alpha;
#endif // _TRANSPARENCY_ON
	return col;
}

#if _PROCEDURALSKY_ON
uniform FLOAT3 _SkyBase, _SkyAwayFromSun, _SkyTowardsSun;
uniform FLOAT _SkyDirectionality;

FLOAT3 SkyProceduralDP(FLOAT3 refl, FLOAT3 lightDir)
{
	FLOAT dp = dot(refl, lightDir);

	if (dp > _SkyDirectionality)
	{
		dp = (dp - _SkyDirectionality) / (1. - _SkyDirectionality);
		return lerp(_SkyBase, _SkyTowardsSun, dp);
	}

	dp = (dp - -1.0) / (_SkyDirectionality - -1.0);
	return lerp(_SkyAwayFromSun, _SkyBase, dp);
}
#endif

#if _PLANARREFLECTIONS_ON
TEXTURE2D_SAMPLER2D(_ReflectionTex);

FLOAT3 PlanarReflection(FLOAT3 refl, FLOAT4 i_screenPos, FLOAT3 n_pixel)
{
	i_screenPos.xy += n_pixel.xz;
	return SAMPLE_TEXPROJ(_ReflectionTex, i_screenPos).xyz;
}
#endif // _PLANARREFLECTIONS_ON


#if !_PLANARREFLECTIONS_ON
TEXCUBE_SAMPLERCUBE(_Skybox);
#endif

void ApplyReflectionSky(
	in FFragData FragData,in FMaterialData MaterialData,in FLightingData LightingData,
	FLOAT3 n_pixel, inout FLOAT4 col)
{
	// Reflection
	FLOAT3 skyColor;
	FLOAT3 refl = reflect(-FragData.CameraVector, n_pixel);
// #if _PLANARREFLECTIONS_ON
// 	skyColor = PlanarReflection(refl, FragData.ScreenPosition, n_pixel);
// #elif _PROCEDURALSKY_ON
// 	skyColor = SkyProceduralDP(refl, LightingData.DirectLightDir.xyz);
// #else
	skyColor = SAMPLE_TEXCUBE(_Skybox, refl).rgb*(1/max(_AlphaScale,0.001))*_ReflectionScale;
// #endif
	// Add primary light to boost it
	skyColor += pow(max(0., dot(refl, _SunDir.xyz)), _DirectionalLightFallOff) * _DirectionalLightBoost * LightingData.DirectLightColor;
	// Fresnel
	const FLOAT IOR_AIR = 1.0;
	const FLOAT IOR_WATER = 1.33;
	// reflectance at facing angle
	FLOAT R_0 = (IOR_AIR - IOR_WATER) / (IOR_AIR + IOR_WATER); R_0 *= R_0;
	// schlick's approximation
	FLOAT R_theta = R_0 + (1.0 - R_0) * pow(1.0 - max(dot(n_pixel, FragData.CameraVector), 0.), _FresnelPower);
	col.rgb = lerp(col.rgb, skyColor, R_theta*col.a);
	//col.rgb = skyColor;
}

void WaterShadingMode(in FFragData FragData,in FMaterialData MaterialData,in FLightingData LightingData,inout FLOAT3 DirectDiffuse,inout FLOAT3 DirectSpecular DEBUG_PBS_ARGS)
{
	FLOAT pixelZ = LinearEyeDepth(FragData.SvPosition.z);
	FLOAT2 uvDepth = FragData.ScreenPosition.xy;
#ifdef _TRANSPARENCY_ON
	FLOAT sceneZ01 = SAMPLE_TEXTURE2D(_SceneRT, uvDepth).a;
#else//!_TRANSPARENCY_ON
	FLOAT sceneZ01 = _SceneDefaultDepth;
#endif//_TRANSPARENCY_ON
	FLOAT sceneZ = LinearEyeDepth(sceneZ01);
	FLOAT4 lodAlpha_worldXZUndisplaced_oceanDepth = FLOAT4(0,FragData.WorldPosition.xz,DEPTH_BASELINE);

	FLOAT3 n_geom = FLOAT3(0, 1, 0);
	FLOAT3 n_pixel = n_geom;
	n_pixel.xz +=  SampleNormalMaps(lodAlpha_worldXZUndisplaced_oceanDepth.yz, lodAlpha_worldXZUndisplaced_oceanDepth.x);
	n_pixel = normalize(n_pixel);
	DEBUG_PBS_CUSTOMDATA_PARAM(WaterNormal, n_pixel)

	FLOAT3 WaterScatterColor = ScatterColour(FragData,MaterialData,LightingData, (sceneZ - pixelZ)*DEPTH_BASELINE);
	DEBUG_PBS_CUSTOMDATA_PARAM(WaterScatterColor, WaterScatterColor)
	FLOAT3 bubbleCol = (FLOAT3)0.;

	FLOAT4 col = OceanEmission(FragData,MaterialData,LightingData, n_pixel, pixelZ, uvDepth, sceneZ, sceneZ01, bubbleCol, WaterScatterColor);
	DEBUG_PBS_CUSTOMDATA_PARAM(WaterEmissionColor, col.rgb)
	ApplyReflectionSky(FragData,MaterialData,LightingData, n_pixel, col);
	DirectDiffuse = col.rgb;
	// if((sceneZ - pixelZ)<1)
	// {
	// 	DirectDiffuse = FLOAT3(1,0,0);
	// }
}


#ifdef _ALPHA_MODIFY
FLOAT GetAlpha(FFragData FragData)
{
	return _AlphaScale;
}
#endif
#endif//_WATER_LIGHT
#endif //PBS_WATER_LIGHT_INCLUDE