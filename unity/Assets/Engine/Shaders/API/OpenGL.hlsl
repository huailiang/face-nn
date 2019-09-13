// For now OpenGL is considered at GLES2 level
#define UNITY_UV_STARTS_AT_TOP 0
#define UNITY_REVERSED_Z 0
#define UNITY_GATHER_SUPPORTED 0

#define SharedPointClampState

#define TEXTURE2D_SAMPLER2D(textureName) sampler2D textureName
#define TEXTURE2D_SAMPLER2D_ST(textureName) FLOAT4 textureName##_ST
#define TRANSFORM_TEX(tex,name) (tex.xy * name##_ST.xy + name##_ST.zw)
#define TEXTURE2D_2(textureName)
#define TEXTURE2D_SAMPLER2D_2(textureName) 

#define TEXCUBE_SAMPLERCUBE(textureName) samplerCUBE textureName
#define TEXTURE2DARRAY_SAMPLER2D(textureName) Texture2DArray textureName; SamplerState sampler##textureName
// #define TEX2DARRAY_SAMPLER2D(textureName) Texture2DArray textureName; SamplerState sampler_##textureName

// #define SAMPLER2D(textureName) sampler2D textureName
// #define SHADOWMAP2D(textureName) Texture2D textureName; SamplerComparisonState sampler_##textureName

#define TEXTURE2D_ARGS(textureName) sampler2D textureName
#define TEXTURE2D_PARAM(textureName) textureName

#define SAMPLE_TEXTURE2D(textureName, coord2) tex2D(textureName, coord2)
#define SAMPLE_TEXTURE2D_STATE(textureName,samplerName, coord2)
#define SAMPLE_TEXTURE2D_STATE_LOD(textureName,samplerName, coord2,lod)

#define SAMPLE_TEXTURE2DDDXY(textureName, coord2,dx,dy) tex2D(textureName, coord2,dx,dy)
#define SAMPLE_TEXTURE2D_LOD(textureName, coord2, lod) tex2Dlod(textureName, FLOAT4(coord2, 0.0, lod))
// #define TEX2D(textureName, coord2) tex2D(textureName, coord2)
#define SAMPLE_TEXPROJ(textureName,coord4)  tex2Dproj(textureName, coord4)

#define SAMPLE_TEXCUBE(textureName,coord2)  texCUBE(textureName, coord2)
#define SAMPLE_TEXCUBE_LOD(textureName,coord,lod) texCUBElod (textureName,FLOAT4(coord, lod))
#define SAMPLE_TEX2DARRAY(textureName,coord) textureName.Sample (sampler##textureName,coord)
// #define SAMPLE_TEX2DARRAY(textureName,coord) textureName.Sample (sampler_##textureName,coord)

// #define LOAD_TEXTURE2D(textureName, texelSize, icoord2) tex2D(textureName, icoord2 / texelSize)
// #define LOAD_TEXTURE2D_LOD(textureName, texelSize, icoord2) tex2Dlod(textureName, float4(icoord2 / texelSize, 0.0, lod))
#define GATHER_TEXTURE2D(textureName, coord2)
#define GATHER_TEXTURE2D_STATE(textureName,samplerName, coord2)
#define SAMPLE_DEPTH_TEXTURE(textureName, coord2) SAMPLE_TEXTURE2D(textureName, coord2).r
#define SAMPLE_DEPTH_TEXTURE_LOD(textureName, coord2, lod) SAMPLE_TEXTURE2D_LOD(textureName, coord2, lod).r

#if SHADER_API_GLES
#    define UNITY_BRANCH
#    define UNITY_FLATTEN
#    define UNITY_UNROLL
#    define UNITY_LOOP
#    define UNITY_FASTOPT
#else
#    define UNITY_BRANCH    [branch]
#    define UNITY_FLATTEN   [flatten]
#    define UNITY_UNROLL    [unroll]
#    define UNITY_LOOP      [loop]
#    define UNITY_FASTOPT   [fastopt]
#endif

#define CBUFFER_START(name)
#define CBUFFER_END

#define FXAA_HLSL_3 1
#define SMAA_HLSL_3 1

// pragma exclude_renderers is only supported since Unity 2018.1 for compute shaders
#if UNITY_VERSION < 201810 && !defined(SHADER_API_GLCORE)
#    define DISABLE_COMPUTE_SHADERS 1
#    define TRIVIAL_COMPUTE_KERNEL(name) [numthreads(1, 1, 1)] void name() {}
#endif
