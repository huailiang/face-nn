#ifndef PBS_LIGHTINGHEAD_INCLUDE
#define PBS_LIGHTINGHEAD_INCLUDE

FLOAT3 _DirectionalLightDir0;
FLOAT4 _DirectionalLightColor0;

FLOAT4 _PointLightPos0;
FLOAT4 _PointLightColor0;

#ifdef _DOUBLE_LIGHTS
    FLOAT3 _DirectionalLightDir1;
    FLOAT4 _DirectionalLightColor1;
#endif


FLOAT3 GetLightColor0()
{
#ifdef _ADDLIGHTING
    #ifdef LIGHTMAP_ON
        return FLOAT3(0,0,0);
    #else//!LIGHTMAP_ON
        return _LightColor0.xyz;
    #endif//LIGHTMAP_ON    
#else//!_ADDLIGHTING
    // return _DirectionalLightColor0.xyz;
    return FLOAT3(1.0,0.9,0.7);
#endif//_ADDLIGHTING
}

FLOAT3 GetLightColor1()
{
#ifdef _ADDLIGHTING
    return FLOAT3(0,0,0);
#else//!_ADDLIGHTING
     #ifdef _DOUBLE_LIGHTS
        return _DirectionalLightColor1.xyz;
     #else//!_DOUBLE_LIGHTS
        return FLOAT3(0,0,0);
    #endif//_DOUBLE_LIGHTS
#endif//_ADDLIGHTING
}

FLOAT3 GetLightDir0()
{
#ifdef _ADDLIGHTING
    return _WorldSpaceLightPos0.xyz;
#else//!_ADDLIGHTING
    return _DirectionalLightDir0.xyz;
#endif//_ADDLIGHTING
}

FLOAT3 GetLightDir1()
{
#ifdef _ADDLIGHTING
    return FLOAT3(0,0,0);
#else//!_ADDLIGHTING
     #ifdef _DOUBLE_LIGHTS
            return _DirectionalLightDir1.xyz;
     #else//!_DOUBLE_LIGHTS
        return FLOAT3(0,0,0);
    #endif//_DOUBLE_LIGHTS
#endif//_ADDLIGHTING
}
#endif //PBS_LIGHTINGHEAD_INCLUDE