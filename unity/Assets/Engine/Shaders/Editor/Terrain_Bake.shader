Shader "Custom/Editor/Terrain_Bake"
{
	Properties
	{
		_BlendTex("Blend Tex ", 2D) = "black" {}
		_MainTex0("Main Tex 0", 2D) = "white" {}
		_MainTex1("Main Tex 0", 2D) = "white" {}
		_MainTex2("Main Tex 0", 2D) = "white" {}
		_MainTex3("Main Tex 0", 2D) = "white" {}
	}

	HLSLINCLUDE
	#include "../StdLib.hlsl"

	TEXTURE2D_SAMPLER2D(_BlendTex);
	TEXTURE2D_SAMPLER2D(_MainTex0);
	TEXTURE2D_SAMPLER2D(_MainTex1);
	TEXTURE2D_SAMPLER2D(_MainTex2);
	TEXTURE2D_SAMPLER2D(_MainTex3);
	FLOAT4 _Scale0;
	FLOAT4 _Scale1;

	FLOAT4 Frag(VaryingsDefault i) : SV_Target
	{
		FLOAT4 color = FLOAT4(0,0,0,0);
		FLOAT4 blend = SAMPLE_TEXTURE2D(_BlendTex, i.texcoord);
		color += SAMPLE_TEXTURE2D(_MainTex0, i.texcoord*_Scale0.xy)*blend.x;
		color += SAMPLE_TEXTURE2D(_MainTex1, i.texcoord*_Scale0.zw)*blend.y;
		color += SAMPLE_TEXTURE2D(_MainTex2, i.texcoord*_Scale1.xy)*blend.z;
		color += SAMPLE_TEXTURE2D(_MainTex3, i.texcoord*_Scale1.zw)*blend.w;
		return color;
	}
	ENDHLSL

	SubShader
	{
		Cull Off ZWrite Off ZTest Always
		Pass
		{
			Blend One Zero
			HLSLPROGRAM

			#pragma vertex VertDefault
			#pragma fragment Frag

			ENDHLSL
		}
		Pass
		{
			Blend One One
			HLSLPROGRAM
			

			#pragma vertex VertDefault
			#pragma fragment Frag

			ENDHLSL
		}

	}
}
