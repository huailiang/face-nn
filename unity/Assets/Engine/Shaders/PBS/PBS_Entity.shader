Shader "Custom/PBS/Entity"
{
	Properties
	{
		_BaseTex ("Base Tex", 2D) = "white" {}
		_Color("Base Color", Color) = (1,1,1,1)
		_PBSTex("Norma:xy Metallic:b Roughness:a", 2D) = "" {}
		_PbsParam("Pbs Vector", Vector) = (0.75,0.75,0,1)
		_MagicParam("MagicParam", Vector) = (0.1,0.5,0.02,1.0)

		_EffectTex("Effect:Emission AO", 2D) = "white" {}
		_Param0("Effect Param : Emission OverLay Parallax EnvReflect1",Vector) = (0,0,0,0)		
		[HDR]_Color1("Effect Color : Emission OverLay",Color) = (1,1,1,1)

		[HideInInspector] _SrcBlend("__src", Float) = 1.0
		[HideInInspector] _DstBlend("__dst", Float) = 0.0
		[HideInInspector] _ZWrite("__zw", Float) = 1.0
		[HideInInspector] _DebugMode("__debugMode", Float) = 0.0
	}

	HLSLINCLUDE
		//feature
		#define _CUSTOM_EFFECT
		#define _PARAM_REMAP
		//lighting
		#define _VERTEX_GI
		#define _STANDARD_LIGHT
		#define _DOUBLE_LIGHTS
	ENDHLSL

	SubShader
	{
		Tags { "RenderType"="Opaque" "PerformanceChecks" = "False" }
		LOD 100

		Pass
		{
			Name "FORWARD"
			Tags{ "LightMode" = "ForwardBase" }
			Blend[_SrcBlend][_DstBlend]
			ZWrite[_ZWrite]

			HLSLPROGRAM

				#pragma target 3.0

				#include "../Common/Vertex.hlsl"
				#include "../Common/Pixel.hlsl"

				#pragma shader_feature _ _BASE_FROM_COLOR
				#pragma shader_feature _ _PBS_FROM_PARAM
				#pragma shader_feature _ _PBS_HALF_FROM_PARAM
				#pragma shader_feature _ _PBS_NO_IBL
				#pragma shader_feature _ _ALPHA_TEST
				#pragma shader_feature _ _ETX_EFFECT
				
				#pragma vertex vertForwardBase
				#pragma fragment fragForwardBase
			
			ENDHLSL
		}
	}
	CustomEditor "XEngine.Editor.PBSShaderGUI"
}
