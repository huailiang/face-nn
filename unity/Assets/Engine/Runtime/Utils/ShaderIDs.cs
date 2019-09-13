using UnityEngine;

namespace CFEngine
{
    // Pre-hashed shader ids - naming conventions are a bit off in this file as we use the same
    // fields names as in the shaders for ease of use... Would be nice to clean this up at some
    // point.
    public static class ShaderIDs
    {
        //common
        public static readonly int MainTex = Shader.PropertyToID ("_MainTex");
        public static readonly int SceneDepthTex = Shader.PropertyToID ("_SceneDepthRT");
        public static readonly int CameraDepthTex = Shader.PropertyToID ("_CameraDepthRT");
        public static readonly int PostCameraDepthTex = Shader.PropertyToID ("_PostCameraDepthRT");
        public static readonly int SceneRT = Shader.PropertyToID ("_SceneRT");
        public static readonly int SceneRT0 = Shader.PropertyToID ("_SceneRT0");
        public static readonly int SceneRT1 = Shader.PropertyToID ("_SceneRT1");
        public static readonly int AutoExposureScale = Shader.PropertyToID ("_AutoExposureScale");
        public static readonly int RenderViewportScaleFactor = Shader.PropertyToID ("_RenderViewportScaleFactor");
        public static readonly int UVTransform = Shader.PropertyToID ("_UVTransform");
        public static readonly int DebugMode = Shader.PropertyToID ("_PPDebugMode");
        public static readonly int DebugLayer = Shader.PropertyToID ("_DebugLayer");
        //postprocess
        //SMAA
        public static readonly int SMAA_Flip = Shader.PropertyToID ("_SMAA_Flip");
        public static readonly int SMAA_Flop = Shader.PropertyToID ("_SMAA_Flop");
        //Fog
        public static readonly int Fog_Color = Shader.PropertyToID ("_CustomFogColor");
        public static readonly int Fog_Params = Shader.PropertyToID ("_CustomFogParams");
        //Blur
        public static readonly int GhostBlur_Param = Shader.PropertyToID ("_BlurParam");
        public static readonly int RadialBlur_Param = Shader.PropertyToID ("_RadialBlurParam");
        //Dof
        public static readonly int Dof_Temp = Shader.PropertyToID ("_DepthOfFieldTemp");
        public static readonly int Dof_Tex = Shader.PropertyToID ("_DepthOfFieldTex");
        public static readonly int Dof_Distance = Shader.PropertyToID ("_Distance");
        public static readonly int Dof_LensCoeff = Shader.PropertyToID ("_LensCoeff");
        public static readonly int Dof_MaxCoC = Shader.PropertyToID ("_MaxCoC");
        public static readonly int Dof_RcpMaxCoC = Shader.PropertyToID ("_RcpMaxCoC");
        public static readonly int Dof_RcpAspect = Shader.PropertyToID ("_RcpAspect");
        public static readonly int Dof_CoCTex = Shader.PropertyToID ("_CoCTex");

        //preeffect
        public static readonly int PreEffect_Param = Shader.PropertyToID ("_PreffectParam");
        //distortion
        public static readonly int Distortion_Tex = Shader.PropertyToID ("_DistortionTex");
        public static readonly int Distortion_DepthTex = Shader.PropertyToID ("_DistortionDepth");

        // public static readonly int AutoExposureTex = Shader.PropertyToID ("_AutoExposureTex");
        //DOF
        public static readonly int Dof_Output0 = Shader.PropertyToID ("_Output0");
        public static readonly int Dof_Output1 = Shader.PropertyToID ("_Output1");
        public static readonly int Dof_SetupRT = Shader.PropertyToID ("_SetupRT");
        public static readonly int Dof_CocTileRT0 = Shader.PropertyToID ("_CocTileRT0");
        public static readonly int Dof_CocTileRT1 = Shader.PropertyToID ("_CocTileRT1");
        public static readonly int Dof_ViewportRect = Shader.PropertyToID ("_ViewportRect");
        public static readonly int Dof_CocModelParameters = Shader.PropertyToID ("_CocModelParameters");
        public static readonly int Dof_DepthBlurParameters = Shader.PropertyToID ("_DepthBlurParameters");
        public static readonly int Dof_DilateParam0 = Shader.PropertyToID ("_DilateParam0");
        public static readonly int Dof_Param = Shader.PropertyToID ("_DofParam");
        public static readonly int Dof_CoC = Shader.PropertyToID ("_CoCTex");
        public static readonly int Dof_Boken = Shader.PropertyToID ("_DoFTex");
        //Bloom
        public static readonly int Bloom_Tex = Shader.PropertyToID ("_BloomTex");

        public static readonly int Bloom_SampleScale = Shader.PropertyToID ("_SampleScale");
        public static readonly int Bloom_Threshold = Shader.PropertyToID ("_Threshold");
        // public static readonly int Bloom_DirtTex = Shader.PropertyToID ("_Bloom_DirtTex");
        public static readonly int Bloom_Settings = Shader.PropertyToID ("_Bloom_Settings");
        public static readonly int Bloom_Color = Shader.PropertyToID ("_Bloom_Color");
        // public static readonly int Bloom_DirtTileOffset = Shader.PropertyToID ("_Bloom_DirtTileOffset");
        //BgBlur
        public static readonly int BgBlur_DownSampleNum = Shader.PropertyToID ("_BgBlur_DownSampleNum");

        //ColorGrading
        public static readonly int ColorGrading_Lut2D = Shader.PropertyToID ("_Lut2D");
        public static readonly int ColorGrading_Tonemapping_ToneMode = Shader.PropertyToID ("_ToneMode");
        public static readonly int ColorGrading_Lut2D_Params = Shader.PropertyToID ("_Lut2D_Params");
        public static readonly int ColorGrading_PostExposure = Shader.PropertyToID ("_PostExposure");
        public static readonly int ColorGrading_ColorBalance = Shader.PropertyToID ("_ColorBalance");
        public static readonly int ColorGrading_ColorFilter = Shader.PropertyToID ("_ColorFilter");
        public static readonly int ColorGrading_HueSatCon = Shader.PropertyToID ("_HueSatCon");
        public static readonly int ColorGrading_ChannelMixerRed = Shader.PropertyToID ("_ChannelMixerRed");
        public static readonly int ColorGrading_ChannelMixerGreen = Shader.PropertyToID ("_ChannelMixerGreen");
        public static readonly int ColorGrading_ChannelMixerBlue = Shader.PropertyToID ("_ChannelMixerBlue");
        public static readonly int ColorGrading_Lift = Shader.PropertyToID ("_Lift");
        public static readonly int ColorGrading_InvGamma = Shader.PropertyToID ("_InvGamma");
        public static readonly int ColorGrading_Gain = Shader.PropertyToID ("_Gain");
        public static readonly int ColorGrading_Curves = Shader.PropertyToID ("_Curves");
        public static readonly int ColorGrading_CustomToneCurve = Shader.PropertyToID ("_CustomToneCurve");
        public static readonly int ColorGrading_ToeSegmentA = Shader.PropertyToID ("_ToeSegmentA");
        public static readonly int ColorGrading_ToeSegmentB = Shader.PropertyToID ("_ToeSegmentB");
        public static readonly int ColorGrading_MidSegmentA = Shader.PropertyToID ("_MidSegmentA");
        public static readonly int ColorGrading_MidSegmentB = Shader.PropertyToID ("_MidSegmentB");
        public static readonly int ColorGrading_ShoSegmentA = Shader.PropertyToID ("_ShoSegmentA");
        public static readonly int ColorGrading_ShoSegmentB = Shader.PropertyToID ("_ShoSegmentB");
        //Vignette
        public static readonly int Vignette_Color = Shader.PropertyToID ("_Vignette_Color");
        public static readonly int Vignette_Center = Shader.PropertyToID ("_Vignette_Center");
        public static readonly int Vignette_Settings = Shader.PropertyToID ("_Vignette_Settings");
        public static readonly int Vignette_Mode = Shader.PropertyToID ("_Vignette_Mode");

        // public static readonly int LumaInAlpha = Shader.PropertyToID ("_LumaInAlpha");

        //Dithering
        public static readonly int Dithering_Tex = Shader.PropertyToID ("_DitheringTex");
        public static readonly int Dithering_Coords = Shader.PropertyToID ("_Dithering_Coords");
        //Weather
        public static readonly int Weather_RainFact = Shader.PropertyToID ("_WeatherFact");
        public static readonly int Weather_ViewDirection = Shader.PropertyToID ("_ViewDirection");
        //GodRay
        public static readonly int GodRay_ViewPortLightPos = Shader.PropertyToID ("_ViewPortLightPos");
        public static readonly int GodRay_Param = Shader.PropertyToID ("_Param");
        public static readonly int GodRay_Offset = Shader.PropertyToID ("_Offset");
        public static readonly int GodRay_Tex = Shader.PropertyToID ("_GodRayTex");
        public static readonly int GodRay_Setting = Shader.PropertyToID ("_GodRay_Settings");
        public static readonly int GodRay_SunColor = Shader.PropertyToID ("_GodRay_SunColor");
        public static readonly int GodRay_BlendDepth = Shader.PropertyToID ("_GodRay_BlendDepth");
        //Env
        public static readonly int Env_Cubemap = Shader.PropertyToID ("_EnvCubemap");
        public static readonly int Env_CubemapParam = Shader.PropertyToID ("_EnvCubemapParam");
        public static readonly int Env_LightmapScale = Shader.PropertyToID ("_LightmapScale");

        //Lighting
        public static readonly int Env_CameraPointLight = Shader.PropertyToID ("_CameraPointLight");
        public static readonly int Env_DirectionalLightDir = Shader.PropertyToID ("_DirectionalLightDir0");
        public static readonly int Env_DirectionalLightDir1 = Shader.PropertyToID ("_DirectionalLightDir1");
        public static readonly int Env_DirectionalLightColor = Shader.PropertyToID ("_DirectionalLightColor0");
        public static readonly int Env_DirectionalLightColor1 = Shader.PropertyToID ("_DirectionalLightColor1");

        public static readonly int Env_DirectionalSceneLightDir = Shader.PropertyToID ("_DirectionalSceneLightDir0");
        public static readonly int Env_DirectionalSceneLightDir1 = Shader.PropertyToID ("_DirectionalSceneLightDir1");
        public static readonly int Env_DirectionalSceneLightColor = Shader.PropertyToID ("_DirectionalSceneLightColor0");
        public static readonly int Env_DirectionalSceneLightColor1 = Shader.PropertyToID ("_DirectionalSceneLightColor1");
        public static readonly int Env_AmbientParam = Shader.PropertyToID ("_AmbientParam");
        public static readonly int Env_SkyCube = Shader.PropertyToID ("_Skybox");
        public static readonly int Env_SkyCubeTex = Shader.PropertyToID ("_Tex");

        // public static readonly int Env_PointLightDir = Shader.PropertyToID("_PointLightPos0");

        // public static readonly int Env_PointLightColor = Shader.PropertyToID("_PointLightColor0");

        public static readonly int Env_GameViewCameraPos = Shader.PropertyToID ("_GameViewWorldSpaceCameraPos");
        //Shadow
        public static readonly int Env_ShadowBias = Shader.PropertyToID ("_ShadowBias");
        public static readonly int Env_ShadowSmooth = Shader.PropertyToID ("_SmoothClamp");
        public static readonly int Env_ShadowMapView = Shader.PropertyToID ("_ShadowMapViewProj");
        public static readonly int Env_ShadowMapView1 = Shader.PropertyToID ("_ShadowMapViewProj1");
        public static readonly int Env_ShadowMapTex = Shader.PropertyToID ("_ShadowMapTex");
        public static readonly int Env_ShadowMapSize = Shader.PropertyToID ("_ShadowMapSize");
        //Fog
        public static readonly int Env_FogDisable = Shader.PropertyToID ("_FogDisable");
        public static readonly int Env_HeightFogParameters = Shader.PropertyToID ("_HeightFogParam");
        public static readonly int Env_HeighFogColorParameter0 = Shader.PropertyToID ("_HeightFogColor0");
        public static readonly int Env_HeighFogColorParameter1 = Shader.PropertyToID ("_HeightFogColor1");
        public static readonly int Env_HeighFogColorParameter2 = Shader.PropertyToID ("_HeightFogColor2");
        //Wind
        public static readonly int Env_WindDir = Shader.PropertyToID ("_WindDir");
        public static readonly int Env_WindPos = Shader.PropertyToID ("_WindPos");
        public static readonly int Env_WindPlane = Shader.PropertyToID ("_WindPlane");
        public static readonly int Env_WindParam0 = Shader.PropertyToID ("_WindParam0");
        public static readonly int Env_WindParam1 = Shader.PropertyToID ("_WindParam1");
        public static readonly int Env_EffectParameter = Shader.PropertyToID ("_EffectParameter");
        public static readonly int Env_Interactive = Shader.PropertyToID ("_Interactive");
        //water
        public static readonly int Env_WaterGeomData = Shader.PropertyToID ("_GeomData");
        public static readonly int Env_WaterSunDir = Shader.PropertyToID ("_SunDir");

        //keyword
        //Fog
        // public static readonly string Env_FogAnimtionOn = "FOG_ANIMATION_ON";
        public static readonly string Env_WindOn = "_GLOBAL_WIND_EFFECT";
        //Weather
        public static readonly string Weather_RainDay = "RAINDAY_ON";

        public static readonly string Weather_WeatherKeyWord = "WEATHER_EFFECT_ON";
        public static readonly string Weather_ThunderKeyWord = "THUNDER_ON";
        public static readonly string Weather_RainbowKeyWord = "RAINBOW_ON";
        public static readonly string Weather_StarKeyWord = "STAR_ON";
        public static readonly string Weather_RainEffectKeyWord = "RAIN_EFFECT_ON";

        public static readonly string GhostBlur_KeyWord = "MOTIONBLUR";
        public static readonly string RadialBlur_KeyWord = "RADIALBLUR";
        public static readonly string Fog_KeyWord = "FOG";

    }
}
