/**
* MaterialTemplate.hlsl: Filled in by user defined function for each material being compiled.
*/
#include "PCH.hlsl" 
#include "../Noise.hlsl"

#ifndef PBS_MATERIALTEMPLATE_INCLUDE
#define PBS_MATERIALTEMPLATE_INCLUDE


#define _ParallaxParamX _Param0.x
#define _ParallaxParamY _Param0.y
#define _ParallaxCount ((int)_Param0.z)
#define _ParallaxCount2 ((int)_Param0.w)

FLOAT4 CalculateHeightFog2(FLOAT3 wsPosition)
{
#if !defined(SHADER_API_MOBILE)
	if (_FogDisable > 0)
		return FLOAT4(0, 0, 0, 0);
#endif

	FLOAT dist = length(wsPosition - _WorldSpaceCameraPos.xyz);
	FLOAT d = max(0.0, dist - _HeightFogParam.w);
	FLOAT c = _HeightFogParam.x ;
	FLOAT fogAmount = (1.0 - exp(-d * c)); 	
	return FLOAT4(fogAmount * fogAmount,0,0,0);
}

FLOAT4 ParallaxMapping(FLOAT3 V,FLOAT3 L,FLOAT2 uv)
{
	FLOAT3 uvh = FLOAT3(uv.x,uv.y,0);
	FLOAT  ss = 1.3;

	for(int pi=0; pi < _ParallaxCount; pi++ )
	{
		FLOAT4 v = SAMPLE_TEXTURE2D(_EffectTex,uvh.xy);
		FLOAT  h = v.r*_ParallaxParamX+_ParallaxParamY;
		uvh += (h-uvh.z)*v.z*V.xyz;
	}

	FLOAT4 result = FLOAT4(uvh.xy,0,1.0);

	if( _ParallaxCount2>0 )
	{
		FLOAT3 dT = L;

		FLOAT2 sh = dT.xy*_ParallaxParamX*ss;
		FLOAT ho  = tex2D(_EffectTex,uvh.xy).r;
		FLOAT hn  = ho;
		FLOAT d = 1.0f;
		for(int pi = 0;pi < _ParallaxCount2; pi ++)
		{
			hn  = max( hn,SAMPLE_TEXTURE2D(_EffectTex,uvh.xy+d*sh).r);
			d -= 0.3f;	
		}

		result.a = 1.0f-saturate(saturate(hn-ho)*ss);
	}

	return result;
}


FFragData GetFragData(FInterpolantsVSToPS Interpolants, FLOAT4 SvPosition)
{
	DECLARE_OUTPUT(FFragData, FragData);
	FragData.SvPosition = SvPosition;
#ifdef _SCREEN_POS
	FragData.ScreenPosition = FLOAT4(Interpolants.ScreenPosition.xy/Interpolants.ScreenPositionW.x,Interpolants.ScreenPosition.zw/Interpolants.ScreenPositionW.y);
#endif
	FragData.WorldPosition = Interpolants.WorldPosition.xyz;

#ifdef _GAMEVIEW_CAMERA_POS
	FragData.WorldPosition_CamRelative = Interpolants.WorldPosition.xyz - _GameViewWorldSpaceCameraPos.xyz;
#else
	FragData.WorldPosition_CamRelative = Interpolants.WorldPosition.xyz - _WorldSpaceCameraPos.xyz;
#endif
	

	FLOAT3 TangentToWorld1 = cross(Interpolants.TangentToWorld2.xyz, Interpolants.TangentToWorld0.xyz)* Interpolants.TangentToWorld2.w;
	FragData.TangentToWorld = FLOAT3x3(Interpolants.TangentToWorld0.xyz, TangentToWorld1, Interpolants.TangentToWorld2.xyz);


#if SHADER_API_GLES3
	FragData.CameraVector = normalize(-0.01 * FragData.WorldPosition_CamRelative.xyz);
#else
	FragData.CameraVector = normalize(-FragData.WorldPosition_CamRelative.xyz);
#endif//SHADER_API_GLES3

#if (_OUTPUT_UV_COUNT>0)
	FragData.TexCoords[0] = Interpolants.TexCoords[0];
	#if (_OUTPUT_UV_COUNT>1)
		FragData.TexCoords[1] = Interpolants.TexCoords[1];
	#endif//(_OUTPUT_UV_COUNT>1)
#endif//(_OUTPUT_UV_COUNT>0)

#ifdef _VERTEX_COLOR
	FragData.VertexColor = Interpolants.Color;
#endif//_VERTEX_COLOR

#ifdef _VERTEX_FOG
	FragData.VertexFog =  CalculateHeightFog2(FragData.WorldPosition.xyz);
#endif//_VERTEX_FOG

#ifdef _VERTEX_GI
	FragData.Ambient = Interpolants.VertexGI.xyz;
#endif//_VERTEX_GI

#ifdef _GRASS_LIGHT
	FragData.GrassOcc = Interpolants.VertexGI.w;
#endif//_GRASS_LIGHT

#ifdef _SHADOW_MAP
	FragData.ShadowCoord = Interpolants.ShadowCoord.xyz;
	#ifdef _SHADOW_MAP_CSM
		FragData.ShadowCoordCSM = FLOAT3(Interpolants.ShadowCoordCSM.xy,Interpolants.ShadowCoord.w);
	#endif//_SHADOW_MAP_CSM
#endif//_SHADOW_MAP

#ifdef _PARALLAX_EFFECT
	FLOAT3 tsView = mul(FragData.TangentToWorld,normalize(-FragData.WorldPosition_CamRelative));
	FLOAT3 tsLightDir = mul(FragData.TangentToWorld,normalize(GetLightDir0()));
	FragData.Parallax = ParallaxMapping(tsView,tsLightDir,GET_FRAG_UV);
	GET_FRAG_UV = FragData.Parallax.xy;
#endif//_PARALLAX_EFFECT
	return FragData;
}

void MaterialAlphaTest(FMaterialData MaterialData)
{
	clip(MaterialData.BaseColor.a - _RoleCutout);
}

FLOAT4 GetDefaultBaseColor(in FFragData FragData)
{			
#ifdef _UV_MODIFY
	FLOAT2 uv = ModifyUV(GET_FRAG_UV);
#else
	FLOAT2 uv = GET_FRAG_UV;
#endif

#ifdef USE_MAINTEX
	FLOAT4 color =  SAMPLE_TEXTURE2D(_MainTex, uv);
#else
	FLOAT4 color =  SAMPLE_TEXTURE2D(_BaseTex, uv);
#endif

#ifdef _OVERLAY
#endif//_OVERLAY	

#ifdef _MAIN_COLOR
	return color*_MainColor;
#else//_MAIN_COLOR
	return color;
#endif//_MAIN_COLOR
}

inline FLOAT4 GetBaseColor(in FFragData FragData,inout FMaterialData MaterialData)
{
#if !defined(_BASE_FROM_COLOR)
	#ifdef _COLOR_MODIFY		
		#ifdef _DEFAUTL_BASE_COLOR
			return ColorModify(FragData,GetDefaultBaseColor(FragData));
		#else//!_DEFAUTL_BASE_COLOR
			return ColorModify(FragData);
		#endif//_DEFAUTL_BASE_COLOR		
	#else//!_COLOR_MODIFY
		return GetDefaultBaseColor(FragData);
	#endif//_COLOR_MODIFY
#else
	return _Color;
#endif
}

inline FLOAT4 GetPBSColor(FFragData FragData, in FMaterialData MaterialData)
{
	FLOAT4 pbs = FLOAT4(0.75,0.75,0,1);
#ifndef _DEFAULT_PBS_PARAM
	#if defined(_PBS_FROM_PARAM)
		pbs.zw = _PbsParam.zw;
	#else//!_PBS_FROM_PARAM
		#ifdef _MESH_BLEND
			FLOAT4 splat = FLOAT4(0,0,0,0);
			FLOAT4 blend =MaterialData.BlendTex;
			splat += (SAMPLE_TEXTURE2D(_PBSTex, GET_FRAG_UV)*2-1)*blend.r;
			splat += (SAMPLE_TEXTURE2D(_PBSTex1, GET_FRAG_UV2)*2-1)*blend.g;
			return splat*0.5+0.5;
		#else
			pbs = SAMPLE_TEXTURE2D(_PBSTex, GET_FRAG_UV2);
		#endif//_MESH_BLEND
		#if defined(_PBS_HALF_FROM_PARAM)
			pbs.zw = _PbsParam.zw;
		#elif defined(_PBS_M_FROM_PARAM)
			pbs.w = pbs.z;
			pbs.z = _PbsParam.z;
		#endif//_PBS_HALF_FROM_PARAM
	#endif//_PBS_FROM_PARAM
#endif
	return pbs;
}

inline FLOAT4 GetEmissionAOColor(FLOAT2 uv,FLOAT4 color)
{
	FLOAT4 emi = FLOAT4(0,0,0,0);
	FLOAT4 effectTex = SAMPLE_TEXTURE2D(_EffectTex, uv);
	effectTex.xyz = effectTex.x*color.xyz;		

	FLOAT a = abs(frac((uv.y+_Time.x*3)*3)-0.5); 	
	a = a * a;
	FLOAT noise = abs(cnoise(uv*5*_Emi_Amplitude+FLOAT2(0,_Time.y*_Emi_FlowSpeed))*0.5+0.5);
	noise = noise*noise*noise*noise;
	emi.xyz = effectTex.xyz *_Emi_Color.xyz*(1+noise*20)*_Emi_Intensity;
	emi.a = effectTex.a;
	return emi;
}

inline FLOAT3 GetOverlayColor(FLOAT3 color,FLOAT2 uv, FLOAT worldNormalY)
{
	uv *= _OverLay_UVScale;
	FLOAT4 effectColor = SAMPLE_TEXTURE2D(_EffectTex,uv);
	FLOAT ratio = saturate(effectColor.a - 1 + (worldNormalY + _OverLay_Ratio)*2);
	FLOAT mask = saturate ( lerp(-_OverLay_Mask , _OverLay_Mask+1 , ratio ));
	effectColor *= _OverLay_Color*_OverLay_Color.a*FLOAT(10);
	color = lerp(color.rgb, effectColor.rgb, mask);
	return color;
}

// FLOAT _SelfShadowFade;
inline FLOAT GetShadow(in FFragData FragData,in FMaterialData MaterialData)
{
#ifdef _SHADOW_MAP
	FLOAT ndotl = 1;
	FLOAT camera2shadow = dot(FragData.WorldPosition_CamRelative,FragData.WorldPosition_CamRelative);
	UNITY_BRANCH
	if(ndotl>=0&&camera2shadow<400)
	{
		FLOAT3 coord = FragData.ShadowCoord.xyz;	

		#ifdef _SELF_SHADOW_MAP		
			FLOAT depthInLightSpace = PCF8Filter(coord);
			FLOAT depth = depthInLightSpace ;
		#else//!_SELF_SHADOW_MAP
			FLOAT4 shadowMap = SAMPLE_TEXTURE2D(_ShadowMapTex, coord.xy);
			FLOAT depthInLightSpace = saturate((shadowMap.r - FragData.ShadowCoord.z)*1000);//DecodeFloatRGBA(shadowMap);
			FLOAT depth = depthInLightSpace;
		#endif//_SELF_SHADOW_MAP

		#ifdef _SHADOW_MAP_CSM
		UNITY_BRANCH
		if((coord.x <= 0 || coord.y <= 0 || coord.x >= 1 || coord.y >= 1))
		{
			FLOAT4 shadowMap1 = SAMPLE_TEXTURE2D(_ShadowMapTexCSM, FragData.ShadowCoordCSM.xy);
			depthInLightSpace = shadowMap1.a;
			depth = depthInLightSpace;
		}
		#endif//_SHADOW_MAP_CSM
		return (1-depth);
	}
	return 1;
#else//!_SHADOW_MAP
	return 1;
#endif//_SHADOW_MAP
}

inline FLOAT3 BoxProjectedCubemapReflect (FLOAT3 worldRefl, FLOAT3 worldPos, FLOAT3 cubemapCenter, FLOAT3 boxMin, FLOAT3 boxMax)
{
	FLOAT3 nrdir = normalize(worldRefl);
	FLOAT3 rbmax = (nrdir == 0.0f ? FLT_MAX : (boxMax - worldPos) / nrdir);
	FLOAT3 rbmin = (nrdir == 0.0f ? FLT_MAX : (boxMin - worldPos) / nrdir);

	FLOAT3 rbminmax = (nrdir > 0.0f) ? rbmax : rbmin;

	FLOAT fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);

	worldPos -= cubemapCenter;
	worldRefl = worldPos + nrdir * fa;
	
	return worldRefl;
}

inline FMaterialData GetMaterialData(FFragData FragData)
{
	DECLARE_OUTPUT(FMaterialData, MaterialData)

#ifdef _CUSTOM_MATERIAL
	CustomMaterial(FragData,MaterialData);
#else//!_CUSTOM_MATERIAL
	MaterialData.BaseColor = GetBaseColor(FragData,MaterialData);
	
	#ifdef _COLORBLEND
		MaterialData.BlendColor = 
		(_ColorR.rgb*MaterialData.BaseColor.r*_ColorR.a + 
			_ColorG.rgb*MaterialData.BaseColor.g*_ColorG.a +
			_ColorB.rgb*MaterialData.BaseColor.b*_ColorB.a) * FLOAT(10);
			FLOAT BaseGray = MaterialData.BaseColor.r + MaterialData.BaseColor.g + MaterialData.BaseColor.b;
			FLOAT3 colors[3] = 
			{
				MaterialData.BlendColor,
				_ColorEX1.rgb*_ColorEX1.a*10*BaseGray,
				_ColorEX2.rgb*_ColorEX2.a*10*BaseGray
			};
			int id = (int)(FragData.TexCoords[0].y);
			MaterialData.BlendColor =saturate(colors[id]);
	#else
		MaterialData.BlendColor = MaterialData.BaseColor.rgb;
	#endif

	#ifdef _ALPHA_FROM_COLOR
		MaterialData.BaseColor.a = _Color.a;
	#endif

	#if _ALPHA_TEST
		MaterialAlphaTest(MaterialData);
	#endif

	#ifdef _PBS_MODIFY
		PbsModify(FragData,MaterialData);
	#else//!_PBS_MODIFY
		MaterialData.MetallicScale = _MetallicScale;		
		FLOAT4 pbs = GetPBSColor(FragData, MaterialData);
		MaterialData.SrcPbs = pbs;

		// 定义了 _TERRAIN_PBS 宏后, pbs.xyz 表示切线空间法线(已经Unpack过), pbs.w 表示roughness //
		#if defined(_TERRAIN_PBS)
			float3 normalTS = normalize(pbs.xyz);
			MaterialData.Roughness = pbs.w;

			MaterialData.Metallic = 0.01;
			MaterialData.TangentSpaceNormal = normalTS;
			MaterialData.WorldNormal = normalize(mul(normalTS, FragData.TangentToWorld));
		#else
			MaterialData.Roughness = clamp(pbs.a, FLOAT(0.01), FLOAT(1));
			#ifndef _FULL_SSS
				MaterialData.Metallic = clamp(pbs.b, FLOAT(0.01), FLOAT(1));// *MaterialData.BaseColor.a;
			#endif
			CalcWorldNormal(FragData.TangentToWorld,pbs.xy,MaterialData.WorldNormal,MaterialData.TangentSpaceNormal);
		#endif

		#if defined(_NEED_BOX_PROJECT_REFLECT)||!defined(_PBS_NO_IBL)
			MaterialData.ReflectionVector = -FragData.CameraVector + MaterialData.WorldNormal  * dot(MaterialData.WorldNormal, FragData.CameraVector) * 2.0;
			#ifdef _NEED_BOX_PROJECT_REFLECT
				MaterialData.ReflectionVector = BoxProjectedCubemapReflect(MaterialData.ReflectionVector,FragData.WorldPosition,_BoxCenter.xyz,_BoxCenter.xyz-_BoxSize.xyz*0.5,_BoxCenter.xyz+_BoxSize.xyz*0.5);
			#endif
		#endif//_NEED_BOX_PROJECT_REFLECT||!_PBS_NO_IBL

	#endif//_PBS_MODIFY

		MaterialData.Shadow = GetShadow(FragData,MaterialData);
	#ifdef _OVERLAY
		MaterialData.BlendColor = GetOverlayColor(MaterialData.BlendColor,GET_FRAG_UV,MaterialData.WorldNormal.y);
	#endif
#endif//_CUSTOM_MATERIAL

#ifdef _ETX_EFFECT
	MaterialData.EmissiveAO = GetEmissionAOColor(GET_FRAG_UV,MaterialData.BaseColor);
#endif //_ETX_EFFECT
	return MaterialData;
}

#endif //PBS_MATERIALTEMPLATE_INCLUDE