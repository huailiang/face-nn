#ifndef PBS_PCH_INCLUDE
#define PBS_PCH_INCLUDE


TEXTURE2D_SAMPLER2D(_MainTex);
TEXTURE2D_SAMPLER2D(_BaseTex);
TEXTURE2D_SAMPLER2D(_PBSTex);
TEXTURE2D_SAMPLER2D(_PBSTex1);

FLOAT4 _PbsParam;
FLOAT4 _Color;
FLOAT4 _MainColor;

FLOAT4 _ColorR;
FLOAT4 _ColorG;
FLOAT4 _ColorB;
FLOAT4 _ColorEX1;
FLOAT4 _ColorEX2;
FLOAT4 _Color0;
FLOAT4 _Color1;
FLOAT4 _Color2;

FLOAT4 _MagicParam;
#define _MetallicScale _MagicParam.x
#define _SpecularScale _MagicParam.y
#define _RoleCutout _MagicParam.z
#define _IBLScale _MagicParam.w

#define _SpecScale0 _MagicParam.x
#define _SpecScale1 _MagicParam.y

#define _AmbineScale _MagicParam.z
#define _SelfAmbient _MagicParam.w

#define _TerrainHasNormal _MagicParam.z>0.6
#define _TerrainNormalScale _MagicParam.w

FLOAT4 _Param0;
FLOAT4 _Param1;
FLOAT4 _Param2;
FLOAT4 _Param3;
FLOAT4 _Param4;

#define _LightMapEnable _Param3.x>0.5

TEXTURE2D_SAMPLER2D(_EffectTex);

FLOAT4 _SkinSpecularScatter;
#define _SpecularIntensity _SkinSpecularScatter.x
#define _SpecularRoughness _SkinSpecularScatter.y
#define _ScatteringOffset _SkinSpecularScatter.z
#define _ScatteringPower _SkinSpecularScatter.w

//box Reflect
TEXCUBE_SAMPLERCUBE(_EnvReflectTex);
TEXTURE2D_SAMPLER2D(_ShadowMapTexCSM);

FLOAT4 _EffectParameter;

#ifdef _PARAM_REMAP
    //Emission
    #define _Emi_Intensity _Param0.x
    #define _Emi_FlowSpeed _Param0.y
    #define _Emi_Amplitude _Param0.z
    #define _Emi_Color _Color1
    //OverLay
    #define _OverLay_Mask _Param0.x
    #define _OverLay_Ratio _Param0.y
    #define _OverLay_UVScale _Param0.zz
    #define _OverLay_Color _Color1
    //EnvReflect
    #define _BoxCenter _Param0
    #define _BoxSize _Param1
    //Parallax
    #define _ParallaxParamX _Param0.x
    #define _ParallaxParamY _Param0.y
    #define _ParallaxCount ((int)_Param0.z)
    #define _ParallaxCount2 ((int)_Param0.w)
    //Grass
    #define _Occ_Height _Param0.x
    #define _Occ_Power _Param0.y
    #define _Occ_Scale _Param0.z
#else//!_PARAM_REMAP
    FLOAT4 _OverLayParam;
    FLOAT4 _EmissionAOColor;
    FLOAT4 _OverlayColor;
    FLOAT4 _BoxCenter;
    FLOAT4 _BoxSize;
    FLOAT4 _OccParam;
    //Emission
    #define _Emi_Intensity _OverLayParam.x
    #define _Emi_FlowSpeed _OverLayParam.y
    #define _Emi_Amplitude _OverLayParam.z
    #define _Emi_Color _EmissionAOColor.xyz    
    //OverLay
    #define _OverLay_Mask _OverLayParam.x
    #define _OverLay_Ratio _OverLayParam.y
    #define _OverLay_UVScale _OverLayParam.zz
    #define _OverLay_Color _OverlayColor
    //Parallax
    #define _ParallaxParamX _Param0.x
    #define _ParallaxParamY _Param0.y
    #define _ParallaxCount ((int)_Param0.z)
    #define _ParallaxCount2 ((int)_Param0.w)
    //Grass
    #define _Occ_Height _OccParam.x
    #define _Occ_Power _OccParam.y
    #define _Occ_Scale _OccParam.z
#endif//_PARAM_REMAP

#endif //PBS_PCH_INCLUDE