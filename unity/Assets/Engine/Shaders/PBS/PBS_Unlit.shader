Shader "Custom/PBS/Unlit"
{
	Properties
	{
		_BaseTex ("Base Tex", 2D) = "white" {}
		_MainColor("Main Color", Color) = (1,1,1,1)
		_uvST("Scale Offset 0", Vector) = (1,1,1,1)
		_Color("Base Color", Color) = (1,1,1,1)

		
		[HideInInspector] _SrcBlend("__src", Float) = 1.0
		[HideInInspector] _DstBlend("__dst", Float) = 0.0
		[HideInInspector] _ZWrite("__zw", Float) = 1.0
	}

	HLSLINCLUDE
		//feature
		#define _VERTEX_FOG
		#define _UV_SCALE
		#define _MAIN_COLOR
		//lighting
		// #define _VERTEX_GI
		#define _PBS_FROM_PARAM
		#define _UN_LIGHT
		#define _NO_DEFAULT_SPEC
		#define _PBS_NO_IBL		
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
		
				#pragma shader_feature _ _ALPHA_FROM_COLOR
				#pragma shader_feature _ _SHADOW_MAP

				#pragma vertex vertForwardBase
				#pragma fragment fragForwardBase
			
			ENDHLSL
		}
	}
	CustomEditor "CFEngine.Editor.PBSShaderGUI"
}
