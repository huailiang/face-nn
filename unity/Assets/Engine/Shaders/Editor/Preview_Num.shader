Shader "Custom/Editor/Preview_Num"
{
	Properties
	{
		_MainTex ("Main Tex", 2D) = "white" {} 
		_uvST("uvst",Vector) = (1,1,0,0)	
		_NumOffset("xy offset",Float) = 0	
		_Color("Hit Color",Color) = (1,1,1,1)
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
			float4 _uvST;
			float _NumOffset;
			float4 _Color;
			Varyings Vert(Attributes v)
			{
				Varyings o;
				FLOAT4 WorldPosition = mul(unity_ObjectToWorld, v.vertex);
				o.vertex = mul(unity_MatrixVP, WorldPosition);
				
				// o.texcoord = v.texcoord*_uvST.xy+_uvST.zw;
				
				// o.color = _Color;
				return o;
			}

			half4 Frag(Varyings i) : SV_Target
			{
				// float x = fmod(_NumOffset,10)*0.1f;
				// float y = floor(_NumOffset/10)*0.1f;
				// float2 uv = frac(i.texcoord)*float2(0.1,0.1)+float2(x,y);
				half4 color = _Color;
				// color.a += 0.5f;
				return color;
			}

		ENDHLSL

		SubShader
		{
			Cull off ZWrite Off ZTest Always
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
