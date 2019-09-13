Shader "Custom/Editor/TextureBake"
{
	Properties
	{
		_MainTex ("Main Tex", 2D) = "white" {}
		
	}

		HLSLINCLUDE
			#include "../StdLib.hlsl"

			struct Attributes
			{
				FLOAT3 vertex : POSITION;
				FLOAT2 texcoord : TEXCOORD0;
			};
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
			VaryingsDefault Vert2(Attributes v)
			{
				VaryingsDefault o;
				o.vertex = FLOAT4(v.vertex.xy, 0.0, 1.0);

				o.texcoord = v.texcoord;
				return o;
			}

			half4 Frag(VaryingsDefault i) : SV_Target
			{
				half4 color = SAMPLE_TEXTURE2D(_MainTex, i.texcoord);
				return color;
			}

			half4 Frag2(VaryingsDefault i) : SV_Target
			{
				half4 color = SAMPLE_TEXTURE2D(_MainTex, i.texcoord) + half4(1, 1, 1, 0);
				return color;
			}

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

			Pass
			{
				Tags{ "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" }
				LOD 100

				Blend SrcAlpha OneMinusSrcAlpha

				HLSLPROGRAM

				#pragma vertex Vert2
				#pragma fragment Frag2

				ENDHLSL
			}
		}
}
