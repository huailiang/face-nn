Shader "Custom/PBS/Role"
{
	Properties
	{
		_BaseTex ("Base Tex", 2D) = "white" {}
		_Color("Base Color", Color) = (1,1,1,1)
		_PBSTex("Norma:xy Metallic:b Roughness:a", 2D) = "" {}

		_ColorR("R Channel" , Color) = (1,1,1,0.1)
		_ColorG("G Channel", Color) = (1,1,1,0.1)
		_ColorB("B Channel", Color) = (1,1,1,0.1)
		_ColorEX1("EX1 Channel" , Color) = (1,1,1,0.1)
		_ColorEX2("EX2 Channel", Color) = (1,1,1,0.1)

		_MagicParam("MagicParam", Vector) = (0.1,1,0.02,1.0)

		_EffectTex("Effect:Emission AO", 2D) = "white" {}
		_Param0("Effect Param : Emission OverLay Parallax EnvReflect1",Vector) = (0,0,0,0)		
		[HDR]_Color1("Effect Color : Emission OverLay",Color) = (1,1,1,1)

		_SkinSpecularScatter("SpecularScatter", Vector) = (0.9,1,0.7,0.0)

		[HideInInspector] _SrcBlend("__src", Float) = 1.0
		[HideInInspector] _DstBlend("__dst", Float) = 0.0
		[HideInInspector] _ZWrite("__zw", Float) = 1.0
		[HideInInspector] _DebugMode("__debugMode", Float) = 0.0
	}

	HLSLINCLUDE
		//feature
		#define _CUSTOM_EFFECT
		#define _COLORBLEND
		#define _PART_SSS
		#define SKIN_2
		#define _PARAM_REMAP
		//lighting
		#define _VERTEX_GI
		#define _STANDARD_LIGHT
		#define _DOUBLE_LIGHTS
		#define _SELF_SHADOW_MAP
	ENDHLSL

	SubShader
	{
		Tags { "RenderType"="Opaque" "PerformanceChecks" = "False" "IgnoreProjector"="False" }
		LOD 100

		Pass
		{
			Name "FORWARD"
			Tags{"LightMode" = "ForwardBase" }
			Blend[_SrcBlend][_DstBlend]
			ZWrite[_ZWrite]
			HLSLPROGRAM

				#pragma target 3.0

				#include "../Common/Vertex.hlsl"
				#include "../Common/Pixel.hlsl"

				#pragma shader_feature _ _ALPHA_TEST
				#pragma shader_feature _ _ETX_EFFECT
				#pragma multi_compile _ _SHADOW_MAP
				
				#pragma vertex vertForwardBase
				#pragma fragment fragForwardBase
			
			ENDHLSL
		}
	}
	CustomEditor "XEngine.Editor.PBSShaderGUI"
}
