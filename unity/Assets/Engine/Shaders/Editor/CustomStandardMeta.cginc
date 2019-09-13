// Unity built-in shader source. Copyright (c) 2016 Unity Technologies. MIT license (see license.txt)

#ifndef CUSTOM_STANDARD_META_INCLUDED
#define CUSTOM_STANDARD_META_INCLUDED

// Functionality for Standard shader "meta" pass
// (extracts albedo/emission for lightmapper etc.)

#include "UnityCG.cginc"
#include "UnityStandardInput.cginc"
#include "UnityMetaPass.cginc"
#include "UnityStandardCore.cginc"

struct v2f_meta
{
    float4 uv       : TEXCOORD0;
    float4 pos      : SV_POSITION;
    float3 worldPos : TEXCOORD1;
};

v2f_meta vert_meta (VertexInput v)
{
    v2f_meta o;
    o.pos = UnityMetaVertexPosition(v.vertex, v.uv1.xy, v.uv2.xy, unity_LightmapST, unity_DynamicLightmapST);
    o.uv = TexCoords(v);
    o.worldPos  == mul(unity_ObjectToWorld, v.vertex).xyz;
    return o;
}

// Albedo for lightmapping should basically be diffuse color.
// But rough metals (black diffuse) still scatter quite a lot of light around, so
// we want to take some of that into account too.
half3 UnityLightmappingAlbedo (half3 diffuse, half3 specular, half smoothness)
{
    half roughness = SmoothnessToRoughness(smoothness);
    half3 res = diffuse;
    res += specular * roughness * 0.5;
    return res;
}
half4 _PbsParam;
half2 CustomMetallicRough(float2 uv)
{
    return half2(_PbsParam.z,1-_PbsParam.w);
//     half2 mg;
// #ifdef _METALLICGLOSSMAP
//     mg.r = tex2D(_MetallicGlossMap, uv).r;
// #else
//     mg.r = _Metallic;
// #endif

// #ifdef _SPECGLOSSMAP
//     mg.g = 1.0f - tex2D(_SpecGlossMap, uv).r;
// #else
//     mg.g = 1.0f - _Glossiness;
// #endif
//     return mg;
}
half4 _MainColor;
sampler2D _BaseTex;
sampler2D _BlendTex;
sampler2D _BaseTex0;
sampler2D _BaseTex1;
sampler2D _BaseTex2;
sampler2D _BaseTex3;
half4 _TerrainScale;
half4 _Color0;
half4 _uvST;
half4 _uvST1;
half4 CustomAlbedo(float4 texcoords,float3 worldPos)
{
    #ifdef _TERRAIN
        half4 splat = half4(0,0,0,0);
        #ifdef _SPLAT1
            half2 uv0 = worldPos.xz*_TerrainScale[0];			
            splat += tex2D(_BaseTex0, uv0);
        #endif
        #ifdef _SPLAT2
            half4 blend = tex2D(_BlendTex, texcoords.xy);
            half2 uv0 = worldPos.xz*_TerrainScale[0];
            half2 uv1 = worldPos.xz*_TerrainScale[1];

            splat += tex2D(_BaseTex0, uv0)*blend.r;
            splat += tex2D(_BaseTex1, uv1)*blend.g;
        #endif
        #ifdef _SPLAT3
            half4 blend = tex2D(_BlendTex, texcoords.xy);
            half2 uv0 = worldPos.xz*_TerrainScale[0];
            half2 uv1 = worldPos.xz*_TerrainScale[1];
            half2 uv2 = worldPos.xz*_TerrainScale[2];
            splat += tex2D(_BaseTex0, uv0)*blend.r;
            splat += tex2D(_BaseTex1, uv1)*blend.g;
            splat += tex2D(_BaseTex2, uv2)*blend.b;
            // splat = blend.rrrr;
        #endif
        #ifdef _SPLAT4
            half4 blend = tex2D(_BlendTex, texcoords.xy);
            half2 uv0 = worldPos.xz*_TerrainScale[0];
            half2 uv1 = worldPos.xz*_TerrainScale[1];
            half2 uv2 = worldPos.xz*_TerrainScale[2];
            half2 uv3 = worldPos.xz*_TerrainScale[3];
            splat += tex2D(_BaseTex0, uv0)*blend.r;
            splat += tex2D(_BaseTex1, uv1)*blend.g;
            splat += tex2D(_BaseTex2, uv2)*blend.b;
            splat += tex2D(_BaseTex3, uv3)*blend.a;
            // splat = blend.bbbb;
        #endif
        half4 albedo = splat;// half4(0, blend.g, 0, 1);	
    #else
		#ifdef _MESH_BLEND
			half4 splat = half4(0,0,0,0);
			half2 uv0 = texcoords.xy*_uvST.xy+_uvST.wz;
			half2 uv1 = texcoords.xy*_uvST1.xy+_uvST1.wz;
			half4 blend = tex2D(_BlendTex,texcoords.xy);
			half4 base0 = tex2D(_BaseTex,uv0);
			half4 base1 = tex2D(_BaseTex1,uv1);				
			splat += base0*blend.r*_MainColor*_MainColor.a*10;
			splat += base1*blend.g*_Color0*_Color0.a*10;
		//	MaterialData.BlendTex = blend;
			//MaterialData.MetallicScale = blend.r*_SpecScale0+blend.g*_SpecScale1;	
				half4 albedo = splat;
		#else//!_MESH_BLEND
				half4 albedo = _MainColor * tex2D (_BaseTex, texcoords.xy);
		#endif
    #endif
    
// #if _DETAIL
//     #if (SHADER_TARGET < 30)
//         // SM20: instruction count limitation
//         // SM20: no detail mask
//         half mask = 1;
//     #else
//         half mask = DetailMask(texcoords.xy);
//     #endif
//     half3 detailAlbedo = tex2D (_DetailAlbedoMap, texcoords.zw).rgb;
//     #if _DETAIL_MULX2
//         albedo *= LerpWhiteTo (detailAlbedo * unity_ColorSpaceDouble.rgb, mask);
//     #elif _DETAIL_MUL
//         albedo *= LerpWhiteTo (detailAlbedo, mask);
//     #elif _DETAIL_ADD
//         albedo += detailAlbedo * mask;
//     #elif _DETAIL_LERP
//         albedo = lerp (albedo, detailAlbedo, mask);
//     #endif
// #endif
    return albedo;
}
inline FragmentCommonData CustomRoughnessSetup(float4 i_tex,float3 i_worldPos)
{
    half2 metallicGloss = CustomMetallicRough(i_tex.xy);
    half metallic = metallicGloss.x;
    half smoothness = metallicGloss.y; // this is 1 minus the square root of real roughness m.

    half oneMinusReflectivity;
    half3 specColor;
    half4 color = CustomAlbedo(i_tex,i_worldPos);
    half3 diffColor = DiffuseAndSpecularFromMetallic(color, metallic, /*out*/ specColor, /*out*/ oneMinusReflectivity);
    half alpha = color.a;
    #if defined(_ALPHATEST_ON)
        clip (alpha - _Cutoff);
    #endif
    FragmentCommonData o = (FragmentCommonData)0;
    o.diffColor = diffColor;
    o.specColor = specColor;
    o.oneMinusReflectivity = oneMinusReflectivity;
    o.smoothness = smoothness;
    o.alpha = alpha;
    return o;
}

float4 frag_meta (v2f_meta i) : SV_Target
{
    // we're interested in diffuse & specular colors,
    // and surface roughness to produce final albedo.
    FragmentCommonData data = CustomRoughnessSetup (i.uv,i.worldPos);

    UnityMetaInput o;
    UNITY_INITIALIZE_OUTPUT(UnityMetaInput, o);

#if defined(EDITOR_VISUALIZATION)
    o.Albedo = data.diffColor;
#else
    o.Albedo = UnityLightmappingAlbedo (data.diffColor, data.specColor, data.smoothness);
#endif
    o.SpecularColor = data.specColor;
    o.Emission = Emission(i.uv.xy);

    return UnityMetaFragment(o);
}

#endif // CUSTOM_STANDARD_META_INCLUDED
