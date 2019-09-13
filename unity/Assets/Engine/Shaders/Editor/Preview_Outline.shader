Shader "Custom/Editor/Preview_Outline"
{
	Properties
	{
		_Color("Outline Color",Color) = (1,1,1,1)
	}

		HLSLINCLUDE

			#include "../StdLib.hlsl"

			struct Attributes
			{
				FLOAT4 vertex : POSITION;
				FLOAT3 normal : NORMAL;
				// FLOAT2 texcoord : TEXCOORD0;
			};
			
			struct Varyings
			{
				FLOAT4 vertex : SV_POSITION;
				// FLOAT2 texcoord : TEXCOORD0;
				// FLOAT4 color : COLOR;
			};
			float4 _Color;
			Varyings Vert(Attributes v)
			{
				Varyings o;
				v.vertex.xyz += v.normal.xyz * 0.1f;
				FLOAT4 WorldPosition = mul(unity_ObjectToWorld, v.vertex);
				o.vertex = mul(unity_MatrixVP, WorldPosition);
				return o;
			}

			half4 Frag(Varyings i) : SV_Target
			{
				half4 color = _Color;
				color.a = 0.5f;
				return color;
			}

		ENDHLSL

		SubShader
		{
			ZWrite Off
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
