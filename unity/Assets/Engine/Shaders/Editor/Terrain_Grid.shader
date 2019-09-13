Shader "Custom/Editor/Terrain_Grid"
{
	Properties
	{
		_MainTex ("Main Tex", 2D) = "white" {} 
		_HitPos("xyz:Hit Pos w:scale",Vector) = (-1,-1,-1,1)	
		_HitColor("Hit Color",Color) = (1,1,1,1)
	}

		HLSLINCLUDE

			#include "../StdLib.hlsl"

			struct Attributes
			{
				FLOAT4 vertex : POSITION;
				FLOAT2 texcoord : TEXCOORD0;
				FLOAT4 color : COLOR;
			};
			struct Varyings
			{
				FLOAT4 vertex : SV_POSITION;
				FLOAT2 texcoord : TEXCOORD0;
				FLOAT4 color : COLOR;
			};
			TEXTURE2D_SAMPLER2D(_MainTex);
			float4 _HitPos;
			float4 _HitColor;
			Varyings Vert(Attributes v)
			{
				Varyings o;
				FLOAT4 WorldPosition = mul(unity_ObjectToWorld, v.vertex);
				o.vertex = mul(unity_MatrixVP, WorldPosition);
				o.texcoord = v.texcoord;
				o.color = v.color;
				if (abs(v.vertex.x - _HitPos.x) < _HitPos.w)
				{
					if (abs(v.vertex.z - _HitPos.z) < _HitPos.w)
					{
						o.color = _HitColor;
					}
				}
				return o;
			}

			half4 Frag(Varyings i) : SV_Target
			{
				half4 color = SAMPLE_TEXTURE2D(_MainTex, i.texcoord)*i.color;
				color.a += 0.5f;
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
