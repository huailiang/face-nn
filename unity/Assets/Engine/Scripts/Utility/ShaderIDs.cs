using UnityEngine;

namespace XEngine
{
    public static class ShaderIDs
    {
        //common
        public static readonly int MainTex = Shader.PropertyToID("_MainTex");
        public static readonly int BaseTex = Shader.PropertyToID("_BaseTex");
        public static readonly int PBSTex = Shader.PropertyToID("_PBSTex");
        public static readonly int UVTransform = Shader.PropertyToID("_UVTransform");
        public static readonly int DebugLayer = Shader.PropertyToID("_DebugLayer");
        //Env
        public static readonly int Env_Cubemap = Shader.PropertyToID("_EnvCubemap");
        public static readonly int Env_CubemapParam = Shader.PropertyToID("_EnvCubemapParam");
        public static readonly int Env_LightmapScale = Shader.PropertyToID("_LightmapScale");
        //Lighting
        public static readonly int Env_DirectionalLightDir = Shader.PropertyToID("_DirectionalLightDir0");
        public static readonly int Env_DirectionalLightDir1 = Shader.PropertyToID("_DirectionalLightDir1");
        public static readonly int Env_DirectionalLightColor = Shader.PropertyToID("_DirectionalLightColor0");
        public static readonly int Env_DirectionalLightColor1 = Shader.PropertyToID("_DirectionalLightColor1");

        public static readonly int Env_AmbientParam = Shader.PropertyToID("_AmbientParam");
        public static readonly int Env_SkyCube = Shader.PropertyToID("_Skybox");
        public static readonly int Env_SkyCubeTex = Shader.PropertyToID("_Tex");
        public static readonly int Env_GameViewCameraPos = Shader.PropertyToID("_GameViewWorldSpaceCameraPos");
        //Shadow
        public static readonly int Env_ShadowBias = Shader.PropertyToID("_ShadowBias");
        public static readonly int Env_ShadowSmooth = Shader.PropertyToID("_SmoothClamp");
        public static readonly int Env_ShadowMapView = Shader.PropertyToID("_ShadowMapViewProj");
        public static readonly int Env_ShadowMapView1 = Shader.PropertyToID("_ShadowMapViewProj1");
        public static readonly int Env_ShadowMapTex = Shader.PropertyToID("_ShadowMapTex");
        public static readonly int Env_ShadowMapSize = Shader.PropertyToID("_ShadowMapSize");
        //Fog
        public static readonly int Env_FogDisable = Shader.PropertyToID("_FogDisable");
        public static readonly int Env_HeightFogParameters = Shader.PropertyToID("_HeightFogParam");
        public static readonly int Env_HeighFogColorParameter0 = Shader.PropertyToID("_HeightFogColor0");
        public static readonly int Env_HeighFogColorParameter1 = Shader.PropertyToID("_HeightFogColor1");
        public static readonly int Env_HeighFogColorParameter2 = Shader.PropertyToID("_HeightFogColor2");

    }
}
