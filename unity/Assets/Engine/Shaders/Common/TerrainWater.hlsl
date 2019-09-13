#ifndef TERRAIN_WATER_LIGHT_INCLUDE
#define TERRAIN_WATER_LIGHT_INCLUDE

// 混合basecolor时判断权重的阈值，权重差超过该值的不参与颜色混合 //
float _BlendWeight;

// 混合切线空间法线 //
float3 BlendNormalTS(float2 normalXY0, float2 normalXY1, float2 normalXY2, float2 normalXY3, float4 blend)
{
	float3 normalTS0 = UnpackNormal(normalXY0);
	float3 normalTS1 = UnpackNormal(normalXY1);
	float3 normalTS2 = UnpackNormal(normalXY2);
	float3 normalTS3 = UnpackNormal(normalXY3);
	float3 normalTS = normalTS0 * blend.r + normalTS1 * blend.g + normalTS2 * blend.b + normalTS3 * blend.a;
	return normalTS;
}

// 混合粗糙度 //
float BlendRoughness(float4 roughness, float4 blend)
{
	return roughness.r * blend.r + roughness.g * blend.g + roughness.b * blend.b +roughness.a * blend.a;
}

// 获得混合权重值 //
float4 BlendFactor(float4 height, float4 blend)
{
	float4 blendFactor;
	blendFactor.r = height.r * blend.r;
	blendFactor.g = height.g * blend.g;
	blendFactor.b = height.b * blend.b;
	blendFactor.a = height.a * blend.a;
	float maxValue = max(blendFactor.r, max(blendFactor.g, max(blendFactor.b, blendFactor.a)));
	blendFactor = max(blendFactor - maxValue + _BlendWeight, 0) * blend;
	return blendFactor/(blendFactor.r + blendFactor.g + blendFactor.b + blendFactor.a);
}

// 混合地形颜色 //
float4 BlendTerrainColor(float4 terrainColor0, float4 terrainColor1, float4 terrainColor2, float4 terrainColor3, float4 blend)
{
	float4 height = float4(terrainColor0.a, terrainColor1.a, terrainColor2.a, terrainColor3.a);
	float4 blendFactor = BlendFactor(height, blend);
	float4 terrainColor = terrainColor0 * blendFactor.r + terrainColor1 * blendFactor.g + 
	 terrainColor2 * blendFactor.b + terrainColor3 * blendFactor.a;

	return terrainColor;
}

// 获取水面高度，水面部分值为 [0,1] 区间，非水面部分值为 -1 //
// channelIndex 范围是 [0,3], 表示使用第1到第4张图 //
float GetWaterHeight(float4 height, float4 blend, int channelIndex)
{
	// 方式1 //
	if(channelIndex < 0 || channelIndex > 3)
	{
		return -1;
	}

	blend = BlendFactor(height, blend);

	float compareValue = -1;
	compareValue = channelIndex == 0 ? blend.r : compareValue;
	compareValue = channelIndex == 1 ? blend.g : compareValue;
	compareValue = channelIndex == 2 ? blend.b : compareValue;
	compareValue = channelIndex == 3 ? blend.a : compareValue;
	
	float h = -1;
	h = channelIndex == 0 ? height.r : h;
	h = channelIndex == 1 ? height.g : h;
	h = channelIndex == 2 ? height.b : h;
	h = channelIndex == 3 ? height.a : h;

	float maxValue = max(blend.r, max(blend.g, max(blend.b, blend.a)));
	float waterHeight = maxValue == compareValue ? h : -1;
	return waterHeight;
}

// 获取水面高度, 1: 不是水面, [0, 1) 区间: 是水面 //
void TerrainWaterHeight(float4 blend, inout FMaterialData MaterialData)
{
#if defined(_TERRAIN_WATER)
	MaterialData.WaterHeight = 1-blend.a;
#endif
}


// ==================== 以下是只有定义了 _TERRAIN_PBS 宏才能使用的方法 ==================== //
#if defined(_TERRAIN_PBS)
// 地形的pbs图对象声明 //
TEXTURE2D_SAMPLER2D(_TerrainPBSTex0);
TEXTURE2D_SAMPLER2D(_TerrainPBSTex1);
TEXTURE2D_SAMPLER2D(_TerrainPBSTex2);
TEXTURE2D_SAMPLER2D(_TerrainPBSTex3);
#endif

// ==================== 以下是只有定义了 _TERRAIN_WATER 宏才能使用的方法 ==================== //
#if defined(_TERRAIN_WATER)

// 为了 _Skybox 对象的声明 //
#define _WATER_LIGHT

#include "WaterLighting.hlsl"

// 地形上的水面相关参数 //
float _WaterHeight;
float _WetToGroundTrans;			// 潮湿地面到正常地面的过渡, [0,1]区间 //
float _WaterToWetTrans;				// 水面到潮湿地面的过渡, [0,1]区间 //
float4 _WaterColor;					// 水的颜色, RGBA //
float _FresnelPow;					// 菲涅尔强度 //
float _SkyboxIntensity;				// 天空盒反射强度 //
float _WaterRoughness;				// 水的粗糙度 //
// float _WaterNormalDisturbance;   	// 水表面法线的扰动 //
float _WetGroundDarkPercent;		// 潮湿地面的颜色变暗程度, [0,1]区间 //

// 处理 MaterialData //
void TerrainMaterialData(inout FMaterialData MaterialData)
{
	// float rnd1 = (rand(_Time.xz) - 0.5) * 2 * _WaterNormalDisturbance;
	// float rnd2 = (rand(_Time.yw) - 0.5) * 2 * _WaterNormalDisturbance;
	// float3 waterNormal = normalize(float3(rnd1, 1, rnd2));

	float3 waterNormal = float3(0,1,0);

	// 法线和粗糙度插值 //
	// 实际的水面区域 lerpValue = 0，过渡部分 lerpValue 在(0,1)之间， 非实际水面部分 lerpValue = 1 //
	float lerpValue = smoothstep(_WaterHeight - _WetToGroundTrans, _WaterHeight, MaterialData.WaterHeight);

	if(lerpValue == 0)		// 水区域 //
	{
		// 水面和过渡区域的界面分界明显是因为法线没有过渡，直接设置成了地面的法线 //
		// 法线过渡: 水面法线 到 地面法线 //
		float waterToWetLerp = smoothstep(_WaterHeight-_WetToGroundTrans*(1+_WaterToWetTrans), _WaterHeight-_WetToGroundTrans, MaterialData.WaterHeight);
		waterNormal = lerp(waterNormal, MaterialData.WorldNormal, waterToWetLerp);

		MaterialData.WorldNormal = waterNormal;
		MaterialData.Roughness = _WaterRoughness;
	}
	else if(lerpValue < 1)	// 水到地面过渡区域 //
	{
		// MaterialData.WorldNormal = MaterialData.WorldNormal;
		MaterialData.Roughness = lerp(_WaterRoughness, MaterialData.Roughness, lerpValue);
	}
	else {}				// 地面区域 //
}

// 处理地形水的光照 //
void TerrainWaterLighting(FFragData FragData, FMaterialData MaterialData, inout float3 DirectDiffuse, inout float3 DirectSpecular)
{
	// 实际的水面区域 lerpValue = 0，过渡区域 lerpValue 在(0,1)之间， 地面区域 lerpValue = 1 //
	float lerpValue = smoothstep(_WaterHeight - _WetToGroundTrans, _WaterHeight, MaterialData.WaterHeight);

	if(lerpValue == 0)			// 水区域 //
	{
		// 离水面距离越大alpha值越大，即越不透明 //
		float alpha = _WaterColor.a;
		alpha = alpha + 0.5*alpha*(max(0, _WaterHeight-MaterialData.WaterHeight)/_WaterHeight);
		alpha = saturate(alpha);
		float3 waterCol = alpha * _WaterColor + (1 - alpha) * DirectDiffuse;

		// reflect //
		float3 worldNormal = MaterialData.WorldNormal;
		worldNormal = normalize(worldNormal);

		float3 reflectDir = reflect(-normalize(FragData.CameraVector), worldNormal);
		float3 reflectCol = SAMPLE_TEXCUBE(_Skybox, reflectDir).rgb;
		
		// fresnel //
		float3 worldView = normalize(FragData.CameraVector);
		float fresnel = pow(1 - max(0, dot(worldNormal, worldView)), _FresnelPow);
		float3 waterDiffCol = (1-fresnel)*waterCol + fresnel*(reflectCol*_SkyboxIntensity);

		float waterToWetLerp = smoothstep(_WaterHeight-_WetToGroundTrans*(1+_WaterToWetTrans), _WaterHeight-_WetToGroundTrans, MaterialData.WaterHeight);
		waterDiffCol = lerp(waterDiffCol, DirectDiffuse*_WetGroundDarkPercent, waterToWetLerp);

		DirectDiffuse = waterDiffCol;

		// DirectSpecular = DirectSpecular * _SpecIntensity;
	}
	else if(lerpValue < 1)		// 水到地面过渡区域 //
	{
		DirectDiffuse = lerp(DirectDiffuse*_WetGroundDarkPercent, DirectDiffuse, lerpValue);
		// DirectSpecular = lerp(DirectSpecular * _SpecIntensity, 0, lerpValue);
	}
	else						// 地面区域 //
	{
		// DirectDiffuse 和 DirectSpecular  不做修改 //
	}
}

#endif				// _TERRAIN_WATER //
#endif				// TERRAIN_WATER_LIGHT_INCLUDE //