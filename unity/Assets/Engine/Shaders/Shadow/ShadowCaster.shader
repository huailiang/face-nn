Shader "Custom/Common/ShadowCaster" 
{
	Properties 
	{
	}

	Subshader 
	{
		Tags{ "RenderType" = "Opaque" "IgnoreProjector" = "True"}
		Pass
		{
			Name "ShadowCaster"
			ZTest LEqual
			HLSLPROGRAM
			#pragma target 3.0
			#include "../StdLib.hlsl"

			struct AttributesShadow
			{
				FLOAT3 vertex : POSITION;
				FLOAT3 TangentX	: NORMAL;
			};

			struct VaryingsShadow
			{
				FLOAT4 vertex : SV_POSITION;
			};

			#pragma vertex vertCustomCast
			#pragma fragment vertCustomFrag
			#pragma shader_feature _ _SELF_SHADOW_MAP
			FLOAT4 _ShadowMapSize;
			FLOAT4 _ShadowBias; // x: depth bias, y: normal bias
			FLOAT3 _DirectionalLightDir0;
			
			FLOAT4 ApplyShadowBias(FLOAT3 positionWS, FLOAT3 normalWS, FLOAT3 lightDirection)
			{
				// FLOAT shadowCos = dot(lightDirection, normalWS);
				// FLOAT shadowSine = sqrt(1-shadowCos*shadowCos);
				// FLOAT normalBias = _ShadowBias.y * shadowSine;

				// positionWS.xyz -= normalWS * normalBias;
				// #if UNITY_REVERSED_Z
				// 	FLOAT clamped = min(clipPos.z, clipPos.w*UNITY_NEAR_CLIP_VALUE);
				// 		positionWS.z = min(positionCS.z, positionCS.w * 1);
				// 	#else
				// 		positionWS.z = max(positionCS.z, positionCS.w * -1);
				// 	#endif
				// #else
				// FLOAT invNdotL = 1.0 - saturate(dot(lightDirection, normalWS));
				// FLOAT scale = invNdotL * _ShadowBias.y;

				// // normal bias is negative since we want to apply an inset normal offset
				positionWS = lightDirection * _ShadowBias.xxx + positionWS;
				// positionWS = normalWS * scale.xxx + positionWS;
				return FLOAT4(positionWS,1);
			}

			VaryingsShadow vertCustomCast(AttributesShadow v)
			{
				VaryingsShadow o;
				o.vertex = mul(unity_ObjectToWorld, float4(v.vertex.xyz,1.0f));				
				#ifdef _SELF_SHADOW_MAP
					FLOAT3 normalWS = mul(unity_ObjectToWorld, float4(v.TangentX.xyz,1.0f));				
					FLOAT4 positionCS = mul(unity_MatrixVP, ApplyShadowBias(o.vertex.xyz, normalWS, _DirectionalLightDir0));
					#if UNITY_REVERSED_Z
						positionCS.z = min(positionCS.z, positionCS.w * 1);
					#else
						positionCS.z = max(positionCS.z, positionCS.w * -1);
					#endif
				#else
					FLOAT4 positionCS = mul(unity_MatrixVP, o.vertex);	
				#endif
				o.vertex = positionCS;
				return o;
			}

			FLOAT4 vertCustomFrag(VaryingsShadow i) : SV_Target
			{				
				FLOAT maskX = saturate(FLOAT(0.99f) - abs((i.vertex.x-_ShadowMapSize.x)*_ShadowMapSize.y));
				FLOAT maskY = saturate(FLOAT(0.99f) - abs((i.vertex.y-_ShadowMapSize.x)*_ShadowMapSize.y));
				FLOAT depth = i.vertex.z/i.vertex.w*sign(maskX*maskY);
				return FLOAT4(depth,depth*depth,0,1);				
			}
			ENDHLSL
		}
	}
}
