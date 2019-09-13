Shader "Custom/Editor/TextureCombineNormal"
{
	Properties
	{
		_NormalMap01 ("Normal Map 01", 2D) = "bump" {}
		_NormalMap02 ("Normal Map 02", 2D) = "bump" {}
		
	}

		HLSLINCLUDE
			#include "../StdLib.hlsl"

			struct Attributes
			{
				FLOAT3 vertex : POSITION;
				FLOAT2 texcoord : TEXCOORD0;
			};
			TEXTURE2D_SAMPLER2D(_NormalMap01);
			TEXTURE2D_SAMPLER2D(_NormalMap02);
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

			float4 Frag(VaryingsDefault i) : SV_Target
			{
				float4 normal0 = tex2D(_NormalMap01, i.texcoord);
				float4 normal1 = tex2D(_NormalMap02, i.texcoord);
				return float4(normal0.xy,normal1.xy);
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
		}
}
