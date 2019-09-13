Shader "Custom/Editor/Terrain_Merge"
{
	Properties
	{
		_MainTex("Main Tex", 2D) = "white" {}
		_MergeParam("Merge Param", Vector) = (0,0,0,0)		
		_Color("Merge Color",Color) = (0,0,0,0)
	}

	HLSLINCLUDE
	#include "../StdLib.hlsl"

	struct Attributes
	{
		FLOAT3 vertex : POSITION;
		FLOAT2 texcoord : TEXCOORD0;
	};
	FLOAT4 _MergeParam;
	FLOAT4 _Color;
	TEXTURE2D_SAMPLER2D(_MainTex);
	VaryingsDefault Vert(Attributes v)
	{
		VaryingsDefault o;
		o.vertex = FLOAT4(v.vertex.xy, 0.0, 1.0);
		o.texcoord = v.texcoord;

	#if UNITY_UV_STARTS_AT_TOP
		o.texcoord = o.texcoord * FLOAT2(1.0, -1.0) + FLOAT2(0.0, 1.0);
	#endif

		return o;
	}
	//VaryingsDefault Vert2(Attributes v)
	//{
	//	VaryingsDefault o;
	//	o.vertex = FLOAT4(v.vertex.xy, 0.0, 1.0);

	//	o.texcoord = v.texcoord;
	//	return o;
	//}

	FLOAT4 Frag(VaryingsDefault i) : SV_Target
	{
		FLOAT circle = length(_MergeParam.xz - i.texcoord.xy);		
		FLOAT4 color = SAMPLE_TEXTURE2D(_MainTex, i.texcoord);
		FLOAT mask = color.r+color.g+color.b;
		if (mask >0)
		{
			color = circle < _MergeParam.y ? _Color : color;
		}
		
		return color;
	}
	//half4 Frag2(VaryingsDefault i) : SV_Target
	//{
	//	half4 color = SAMPLE_TEXTURE2D(_MainTex, i.texcoord) + half4(1, 1, 1, 0);
	//	return color;
	//}

	ENDHLSL

	SubShader
	{
		Cull Off ZWrite Off ZTest Always

			Pass
			{
				HLSLPROGRAM

				#pragma vertex VertDefault
				#pragma fragment Frag

				ENDHLSL
			}
			Pass
			{
				HLSLPROGRAM


				#pragma vertex Vert
				#pragma fragment Frag

				ENDHLSL
			}

	}
}
