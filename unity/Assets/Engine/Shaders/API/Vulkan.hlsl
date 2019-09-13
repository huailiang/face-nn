#define UNITY_UV_STARTS_AT_TOP 1
#define UNITY_REVERSED_Z 1
#define UNITY_GATHER_SUPPORTED (SHADER_TARGET >= 50)

SamplerState global_point_clamp_sampler;

#define SharedPointClampState global_point_clamp_sampler

#define TEXTURE2D_SAMPLER2D(textureName) sampler2D textureName
#define TEXTURE2D_SAMPLER2D_ST(textureName) FLOAT4 textureName##_ST
#define TRANSFORM_TEX(tex,name) (tex.xy * name##_ST.xy + name##_ST.zw)
#define TEXTURE2D_SAMPLER2D_2(textureName) Texture2D textureName; SamplerState sampler##textureName
#define TEXTURE2D_2(textureName) Texture2D textureName
#define TEXCUBE_SAMPLERCUBE(textureName) samplerCUBE textureName
#define TEXTURE2DARRAY_SAMPLER2D(textureName) Texture2DArray textureName; SamplerState sampler##textureName
// #define TEX2DARRAY_SAMPLER2D(textureName) Texture2DArray textureName; SamplerState sampler_##texture

// #define SAMPLER2D(textureName) sampler2D textureName
// #define SHADOWMAP2D(textureName) Texture2D textureName; SamplerComparisonState sampler_##textureName

#define TEXTURE2D_ARGS(textureName) sampler2D textureName
#define TEXTURE2D_PARAM(textureName) textureName

#define SAMPLE_TEXTURE2D(textureName, coord2) tex2D(textureName, coord2)
#define SAMPLE_TEXTURE2D_STATE(textureName,samplerName, coord2) textureName.Sample(samplerName, coord2)
#define SAMPLE_TEXTURE2D_STATE_LOD(textureName,samplerName, coord2,lod) textureName.SampleLevel(samplerName, coord2,lod)

#define SAMPLE_TEXTURE2DDDXY(textureName, coord2,dx,dy) tex2D(textureName, coord2,dx,dy)
#define SAMPLE_TEXTURE2D_LOD(textureName, coord2, lod) tex2Dlod(textureName, FLOAT4(coord2, 0.0, lod))
// #define TEX2D(textureName, coord2) tex2D(textureName, coord2)
#define SAMPLE_TEXPROJ(textureName,coord4)  tex2Dproj(textureName, coord4)

#define SAMPLE_TEXCUBE(textureName,coord2)  texCUBE(textureName, coord2)
#define SAMPLE_TEXCUBE_LOD(textureName,coord,lod) texCUBElod (textureName,FLOAT4(coord, lod))
#define SAMPLE_TEX2DARRAY(textureName,coord) textureName.Sample (sampler##textureName,coord)

// #define LOAD_TEXTURE2D(textureName, texelSize, icoord2) textureName.Load(int3(icoord2, 0))
// #define LOAD_TEXTURE2D_LOD(textureName, texelSize, icoord2) textureName.Load(int3(icoord2, lod))

#define GATHER_TEXTURE2D(textureName, coord2) textureName.Gather(sampler##textureName, coord2)
#define GATHER_TEXTURE2D_STATE(textureName,samplerName, coord2) textureName.Gather(samplerName, coord2)
// #define GATHER_RED_TEXTURE2D(textureName, samplerName, coord2) textureName.GatherRed(samplerName, coord2)
// #define GATHER_GREEN_TEXTURE2D(textureName, samplerName, coord2) textureName.GatherGreen(samplerName, coord2)
// #define GATHER_BLUE_TEXTURE2D(textureName, samplerName, coord2) textureName.GatherBlue(samplerName, coord2)

#define SAMPLE_DEPTH_TEXTURE(textureName, coord2) SAMPLE_TEXTURE2D(textureName, coord2).r
#define SAMPLE_DEPTH_TEXTURE_LOD(textureName, coord2, lod) SAMPLE_TEXTURE2D_LOD(textureName, coord2, lod).r

#define UNITY_BRANCH    [branch]
#define UNITY_FLATTEN   [flatten]
#define UNITY_UNROLL    [unroll]
#define UNITY_LOOP      [loop]
#define UNITY_FASTOPT   [fastopt]

#define CBUFFER_START(name) cbuffer name {
#define CBUFFER_END };

#if UNITY_GATHER_SUPPORTED
    #define FXAA_HLSL_5 1
    #define SMAA_HLSL_4_1 1
#else
    #define FXAA_HLSL_4 1
    #define SMAA_HLSL_4 1
#endif
