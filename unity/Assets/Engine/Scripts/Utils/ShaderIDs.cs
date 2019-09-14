using UnityEngine;

namespace CFEngine
{
    public static class ShaderIDs
    {
        //common
        public static readonly int MainTex = Shader.PropertyToID("_MainTex");
        public static readonly int SceneDepthTex = Shader.PropertyToID("_SceneDepthRT");
        public static readonly int CameraDepthTex = Shader.PropertyToID("_CameraDepthRT");
        public static readonly int PostCameraDepthTex = Shader.PropertyToID("_PostCameraDepthRT");
        public static readonly int AutoExposureScale = Shader.PropertyToID("_AutoExposureScale");
        public static readonly int RenderViewportScaleFactor = Shader.PropertyToID("_RenderViewportScaleFactor");
        public static readonly int UVTransform = Shader.PropertyToID("_UVTransform");
        public static readonly int DebugMode = Shader.PropertyToID("_PPDebugMode");
        public static readonly int DebugLayer = Shader.PropertyToID("_DebugLayer");
        //postprocess
        //SMAA
        public static readonly int SMAA_Flip = Shader.PropertyToID("_SMAA_Flip");
        public static readonly int SMAA_Flop = Shader.PropertyToID("_SMAA_Flop");
        //Env
        public static readonly int Env_Cubemap = Shader.PropertyToID("_EnvCubemap");
        public static readonly int Env_CubemapParam = Shader.PropertyToID("_EnvCubemapParam");
        public static readonly int Env_LightmapScale = Shader.PropertyToID("_LightmapScale");

        //Lighting
        public static readonly int Env_CameraPointLight = Shader.PropertyToID("_CameraPointLight");
        public static readonly int Env_DirectionalLightDir = Shader.PropertyToID("_DirectionalLightDir0");
        public static readonly int Env_DirectionalLightDir1 = Shader.PropertyToID("_DirectionalLightDir1");
        public static readonly int Env_DirectionalLightColor = Shader.PropertyToID("_DirectionalLightColor0");
        public static readonly int Env_DirectionalLightColor1 = Shader.PropertyToID("_DirectionalLightColor1");

        public static readonly int Env_DirectionalSceneLightDir = Shader.PropertyToID("_DirectionalSceneLightDir0");
        public static readonly int Env_DirectionalSceneLightDir1 = Shader.PropertyToID("_DirectionalSceneLightDir1");
        public static readonly int Env_DirectionalSceneLightColor = Shader.PropertyToID("_DirectionalSceneLightColor0");
        public static readonly int Env_DirectionalSceneLightColor1 = Shader.PropertyToID("_DirectionalSceneLightColor1");
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

        public static readonly int Env_EffectParameter = Shader.PropertyToID("_EffectParameter");
        public static readonly int Env_Interactive = Shader.PropertyToID("_Interactive");

        public static readonly string Weather_WeatherKeyWord = "WEATHER_EFFECT_ON";
        public static readonly string Weather_ThunderKeyWord = "THUNDER_ON";
        public static readonly string Weather_RainbowKeyWord = "RAINBOW_ON";
        public static readonly string Weather_StarKeyWord = "STAR_ON";
        public static readonly string Weather_RainEffectKeyWord = "RAIN_EFFECT_ON";

    }
}
