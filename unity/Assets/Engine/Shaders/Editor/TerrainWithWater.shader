Shader "Custom/Editor/TerrainWithWater"
{
	Properties
	{
		_BlendTex("RGBA BlendWeight", 2D) = "" {}
		_BlendWeight("Blend Weight", Range(0.001, 1)) = 0
		
		_BaseTex0("Splat Tex 0", 2D) = "" {}
		_TerrainPBSTex0("PBS Tex 0", 2D) = "" {}

		_BaseTex1("Splat Tex 1", 2D) = "" {}
		_TerrainPBSTex1("PBS Tex 1", 2D) = "" {}

		_BaseTex2("Splat Tex 2", 2D) = "" {}
		_TerrainPBSTex2("PBS Tex 2", 2D) = "" {}

		_BaseTex3("Splat Tex 3", 2D) = "" {}
		_TerrainPBSTex3("PBS Tex 3", 2D) = "" {}

		_WaterHeight("Water Height", Range(0.001,0.995)) = 0
		_WetToGroundTrans("Wet To Dry Transition", Range(0,0.99)) = 0
		_WaterToWetTrans("Water To Wet Trasition", Range(0, 1)) = 0
		_WaterColor("Water Color(RGBA)", Color) = (1,1,1,1)
		_WaterRoughness("Water Roughness", Range(0,1)) = 1
		_SkyboxIntensity("Skybox Intensity", Range(0.01, 5)) = 0
		_FresnelPow("Fresnel Power", Range(0, 10)) = 0
		_WetGroundDarkPercent("Wet Ground Dark Percent", Range(0, 1)) = 0
		// _WaterNormalDisturbance("Water Normal Disturbance", Range(0, 0.005)) = 0
		_Skybox("Sky Box", Cube) = "" {}

		_TerrainScale("RGBA scale", Vector) = (0.125,0.125,0.125,0.125)
		_SplatParam("x:texCount y:tex scale", Vector) = (4,0,0,0)
		[PBSVector(SpecInstensity,0.01,1,MetalicSepcInstensity,0.1,5,E,0,2,IBLScale,0,2)]
		_MagicParam("MagicParam", Vector) = (0.1,0.5,1,1.0)

		[HideInInspector] _DebugMode("__debugMode", Float) = 0.0
	}

	HLSLINCLUDE
		#define DEBUG_APP
		#define _TERRAIN

		#define _TERRAIN_PBS
		#define _TERRAIN_WATER

		#define _VERTEX_FOG
		 
		//lighting		
		#define _VERTEX_GI
		#define _PBS_NO_IBL
		#define _STANDARD_LIGHT
		#define _DOUBLE_LIGHTS

		// LightingData.DirectLightDir 和 LightingData.lighting0 赋值不对导致 //
		// 没使用该宏导致设置WorldNormal后地形物体变黑 //
		#define _SCENE_EFFECT
		#define _SHADOW_MAP
	ENDHLSL

	SubShader
	{
		Tags { "RenderType"="Opaque" "PerformanceChecks" = "False" }
		LOD 100

		Pass
		{
			Name "FORWARD"
			Tags{ "LightMode" = "ForwardBase" }

			HLSLPROGRAM

				#pragma target 3.0

				#include "../Common/Vertex.hlsl"
				#include "../Common/Pixel.hlsl"
 
				// #pragma shader_feature _ _PBS_FROM_PARAM
				// #pragma shader_feature _ _PBS_HALF_FROM_PARAM
				// #pragma shader_feature _ LIGHTMAP_ON
				#pragma shader_feature _SPLAT4 _SPLAT1 _SPLAT2 _SPLAT3 _SPLAT4
				#pragma multi_compile _ CUSTOM_LIGHTMAP_ON
				#pragma multi_compile _ _VOXEL_LIGHT
				#pragma multi_compile_fwdbase

				#pragma vertex vertForwardBase
				#pragma fragment fragForwardBase
			
			ENDHLSL
		}

		Pass
		{
			Name "FORWARD_DELTA"
			Tags{ "LightMode" = "ForwardAdd" }
			Blend One One
			Fog{ Color(0,0,0,0) } // in additive pass fog should be black
			ZWrite Off
			ZTest LEqual

			HLSLPROGRAM
			#define _ADDLIGHTING
			#pragma target 3.0

			// -------------------------------------
			#include "../Common/VertexAdd.hlsl"
			#include "../Common/PixelAdd.hlsl"

			#pragma shader_feature _SPLAT1 _SPLAT2 _SPLAT3 _SPLAT4

			#pragma multi_compile _ CUSTOM_LIGHTMAP_ON
			#pragma multi_compile _ _VOXEL_LIGHT
			#pragma multi_compile_fwdadd_fullshadows

			#pragma vertex vertAdd
			#pragma fragment fragAdd
			//#include "UnityStandardCoreForward.cginc"

			ENDHLSL
		}

		// ------------------------------------------------------------------
		//  Shadow rendering pass
		Pass
		{
			Name "ShadowCaster"
			Tags{ "LightMode" = "ShadowCaster" }

			ZWrite On ZTest LEqual

			CGPROGRAM
				#pragma target 3.0


				#pragma vertex vertShadowCaster
				#pragma fragment fragShadowCaster

				#include "CustomStandardShadow.cginc"

			ENDCG
		}

		Pass
		{
			Name "META"
			Tags{ "LightMode" = "Meta" }

			Cull Off
			
			CGPROGRAM
			#define _TERRAIN
			#pragma vertex vert_meta
			#pragma fragment frag_meta

			#pragma shader_feature _EMISSION
			#pragma shader_feature _METALLICGLOSSMAP
			#pragma shader_feature _ _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A
			#pragma shader_feature ___ _DETAIL_MULX2
			#pragma shader_feature EDITOR_VISUALIZATION

			#include "CustomStandardMeta.cginc"
			ENDCG
		}
	}
	// CustomEditor "PBSShaderGUI"
}