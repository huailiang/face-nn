Shader "Custom/Editor/EditorPreviewTex"
{
	Properties
	{
		_MainTex ("Main Tex", 2D) = "white" {} 
	}

		HLSLINCLUDE

			#include "../StdLib.hlsl"

			struct Attributes
			{
				FLOAT4 vertex : POSITION;
				FLOAT2 texcoord : TEXCOORD0;
			};
			struct Varyings
			{
				FLOAT4 vertex : SV_POSITION;
				FLOAT2 texcoord : TEXCOORD0;
				FLOAT4 color : COLOR;
			};
			TEXTURE2D_SAMPLER2D(_MainTex);
			Varyings Vert(Attributes v)
			{
				Varyings o;
				FLOAT4 WorldPosition = mul(unity_ObjectToWorld, v.vertex);
				o.vertex = mul(unity_MatrixVP, WorldPosition);
				o.texcoord = v.texcoord;
				o.color = FLOAT4(1,1,1,0.3);
				return o;
			}

			half4 Frag(Varyings i) : SV_Target
			{
				half4 color = SAMPLE_TEXTURE2D(_MainTex, i.texcoord)*i.color;
				return color;
			}

		ENDHLSL

		SubShader
		{
			Cull Off ZWrite Off ZTest Always
			Pass
			{
				Tags{ "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" }
				LOD 100

				Blend SrcAlpha OneMinusSrcAlpha
				HLSLPROGRAM


				#pragma vertex Vert
				#pragma fragment Frag

				ENDHLSL
			}
		}
}
