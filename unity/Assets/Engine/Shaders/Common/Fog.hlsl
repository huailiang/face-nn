
#ifndef PBS_FOG_INCLUDE
#define PBS_FOG_INCLUDE

#include "../Noise.hlsl"
#include "../Common/LightingHead.hlsl"

#if !defined(SHADER_API_MOBILE)
FLOAT _FogDisable;
#endif


FLOAT4 _HeightFogParam;	//x:Density y: Fog Speed z: EndHeight w: StartDistance
FLOAT4 _HeightFogColor0;
FLOAT4 _HeightFogColor1;
FLOAT4 _HeightFogColor2;

FLOAT4 CalculateHeightFog(FLOAT3 wsPosition)
{
#if !defined(SHADER_API_MOBILE)
	if (_FogDisable > 0)
		return FLOAT4(0, 0, 0, 0);
#endif

	FLOAT dist = length(wsPosition - _WorldSpaceCameraPos.xyz);
	FLOAT d = max(0.0, dist - _HeightFogParam.w);
	FLOAT c = _HeightFogParam.x ;
	FLOAT fogAmount = (1.0 - exp(-d * c)); 	

	return FLOAT4(fogAmount * fogAmount,0,0,0);
}

FLOAT3 ApplyFog(FLOAT3 color, FLOAT4 fogParam, FLOAT3 WorldPosition)
{
	FLOAT3 basecolor=color;
	FLOAT3 viewDir = normalize(WorldPosition - _WorldSpaceCameraPos.xyz);
	FLOAT3 lightDir = GetLightDir0();
	FLOAT VoL = saturate(dot(-viewDir, -lightDir));
	FLOAT VSDirFog=(1-saturate(viewDir.y *2.5+ _HeightFogParam.y-1) ); 
	FLOAT3 C = lerp(basecolor,lerp(basecolor,_HeightFogColor0.xyz+_HeightFogColor1.xyz*VSDirFog+_HeightFogColor2.xyz*VoL*VoL*2,VSDirFog),fogParam.x);
	return C;
}

#endif //PBS_FOG_INCLUDE