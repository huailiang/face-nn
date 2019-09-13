Shader "Custom/Editor/Terrain_BlendBake"
{
	Properties
	{
		_MainTex ("Main Tex", 2D) = "black" {}
		_BlendParam("xy:pos z:Radius w:strength", Vector) = (0,0,1,1)
		_BlendChannel("x:mask y:index z:lerp weight", Vector) = (0,0,1,0)
		
	}

		HLSLINCLUDE

			#include "../StdLib.hlsl"

			TEXTURE2D_SAMPLER2D(_MainTex);
			FLOAT4 _BlendParam;
			FLOAT4 _BlendChannel;
			FLOAT4 Frag(VaryingsDefault i) : SV_Target
			{
				FLOAT4 blend = SAMPLE_TEXTURE2D(_MainTex, i.texcoord);
				FLOAT dis = length(i.texcoord.xy - _BlendParam.xy) - _BlendParam.z;
				if (dis < 0)
				{
					if (_BlendChannel.x < 0.5f)
					{
						//channel 0 r
						blend.x = _BlendChannel.y;
						//blend.w = 1 - lerp(blend.w, _BlendParam.w, -dis / _BlendParam.z);
					}
					else
					{
						//channel 1 g
						blend.y = _BlendChannel.y;
						blend.z = lerp(blend.z, _BlendParam.w, clamp(-dis / _BlendParam.z, 0, 1));
						
					}
				}
				return blend;
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
