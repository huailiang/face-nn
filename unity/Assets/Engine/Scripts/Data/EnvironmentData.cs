using CFUtilPoolLib;
using UnityEngine;
using System;
using UnityEngine.Rendering;
#if UNITY_EDITOR
using System.IO;
using UnityEditor;
#endif

namespace XEngine
{
    public struct EnverinmentContext
    {
        public RenderingEnvironment env;
        public RenderingManager envMgr;

#if UNITY_EDITOR
        public static void SaveEffect(BinaryWriter bw, IEnverimnentModify modify)
        {
            EffectModify em = modify as EffectModify;
            bw.Write(em.areaId);
            bw.Write(em.effectType);
            bw.Write(em.triggerCount);
        }
#endif
    }

    public interface IEnverimnentLerp
    {
        void Lerp(ref EnverinmentContext context, float percent, bool into, int state);//0 runing 1 begin 2 finish
#if UNITY_EDITOR
        bool needUpdate { get; set; }
#endif
    }

    [System.Serializable]
    public struct LightInfo
    {
        public Vector4 lightDir;
        public Color lightColor;

        public static LightInfo DefaultAsset = new LightInfo()
        {
            lightDir = Quaternion.Euler(90, 0, 0) * Vector3.forward,
            lightColor = Color.black,
        };
    }

    delegate IEnverimnentModify LoadEnv(XBinaryReader reader);
#if UNITY_EDITOR
    public delegate void SaveEnv(BinaryWriter bw, IEnverimnentModify modify);
#endif
    [Serializable]
    public class LightingModify : IEnverimnentModify, IEnverimnentLerp, ISceneObject
    {
        public LightInfo roleLightInfo0 = LightInfo.DefaultAsset;
        public LightInfo roleLightInfo1 = LightInfo.DefaultAsset;
        public LightInfo sceneLightInfo0 = LightInfo.DefaultAsset;
        public LightInfo sceneLightInfo1 = LightInfo.DefaultAsset;

        public EnverimentModifyType GetEnvType()
        {
            return EnverimentModifyType.Lighting;
        }
        public void Reset()
        {
        }
        public void Release()
        {
        }
        public static void OnTrigger(bool into)
        {

        }
#if UNITY_EDITOR
        public LightingModify()
        {
            needUpdate = true;
        }
        public bool needUpdate { get; set; }
#endif
        private void LerpLightInfo(ref LightInfo src, ref LightInfo des, float percent, int dirKey, int colorKey, bool setLightColor = true, bool notUseLightIntensity = false, bool main = true)
        {
            Vector3 dir = Vector3.Lerp(src.lightDir, des.lightDir, percent);
            Shader.SetGlobalVector(dirKey, dir);
            if (setLightColor)
            {
                Vector4 lightColorIntensity0;
                if (!notUseLightIntensity)
                {
                    lightColorIntensity0 = new Vector4(
                        Mathf.Pow(src.lightColor.r * src.lightDir.w, 2.2f),
                        Mathf.Pow(src.lightColor.g * src.lightDir.w, 2.2f),
                        Mathf.Pow(src.lightColor.b * src.lightDir.w, 2.2f), 1);
                    Vector4 lightColorIntensity1 = new Vector4(
                        Mathf.Pow(des.lightColor.r * des.lightDir.w, 2.2f),
                        Mathf.Pow(des.lightColor.g * des.lightDir.w, 2.2f),
                        Mathf.Pow(des.lightColor.b * des.lightDir.w, 2.2f), 1);

                    lightColorIntensity0 = Vector4.Lerp(lightColorIntensity0, lightColorIntensity1, percent);
                }
                else
                {
                    lightColorIntensity0 = main ? Vector4.one : new Vector4(0.5f, 0.5f, 0.5f, 0.5f);
                }
                Shader.SetGlobalVector(colorKey, lightColorIntensity0);
            }

        }

        public void Lerp(ref EnverinmentContext context, float percent, bool into, int state)
        {
#if UNITY_EDITOR
            if (into)
                context.env.lighting.needUpdate = false;
            else if (state == 2)
                context.env.lighting.needUpdate = true;
#endif
            if (into)
            {
                if (roleLightInfo0.lightDir.w > 0)
                    LerpLightInfo(ref context.env.lighting.roleLightInfo0, ref roleLightInfo0, percent,
                        ShaderIDs.Env_DirectionalLightDir, ShaderIDs.Env_DirectionalLightColor);
                if (roleLightInfo1.lightDir.w > 0)
                    LerpLightInfo(ref context.env.lighting.roleLightInfo1, ref roleLightInfo1, percent,
                        ShaderIDs.Env_DirectionalLightDir1, ShaderIDs.Env_DirectionalLightColor1);
                bool notUseLightIntensity = false;
#if UNITY_EDITOR
                notUseLightIntensity =  RenderingEnvironment.isPreview;
#endif
                if (sceneLightInfo0.lightDir.w > 0)
                    LerpLightInfo(ref context.env.lighting.sceneLightInfo0, ref sceneLightInfo0, percent,
                        ShaderIDs.Env_DirectionalSceneLightDir, ShaderIDs.Env_DirectionalSceneLightColor, true, notUseLightIntensity);
                if (sceneLightInfo1.lightDir.w > 0)
                    LerpLightInfo(ref context.env.lighting.sceneLightInfo1, ref sceneLightInfo1, percent,
                        ShaderIDs.Env_DirectionalSceneLightDir1, ShaderIDs.Env_DirectionalSceneLightColor1, true, notUseLightIntensity, false);
            }
            else
            {
                if (roleLightInfo0.lightDir.w > 0)
                    LerpLightInfo(ref roleLightInfo0, ref context.env.lighting.roleLightInfo0, percent,
                        ShaderIDs.Env_DirectionalLightDir, ShaderIDs.Env_DirectionalLightColor);

                if (roleLightInfo1.lightDir.w > 0)
                    LerpLightInfo(ref roleLightInfo1, ref context.env.lighting.roleLightInfo1, percent,
                        ShaderIDs.Env_DirectionalLightDir1, ShaderIDs.Env_DirectionalLightColor1);
                bool notUseLightIntensity = false;
#if UNITY_EDITOR
                notUseLightIntensity = RenderingEnvironment.isPreview;
#endif
                if (sceneLightInfo0.lightDir.w > 0)
                    LerpLightInfo(ref sceneLightInfo0, ref context.env.lighting.sceneLightInfo0, percent,
                        ShaderIDs.Env_DirectionalSceneLightDir, ShaderIDs.Env_DirectionalSceneLightColor, true, notUseLightIntensity);
                if (sceneLightInfo1.lightDir.w > 0)
                    LerpLightInfo(ref sceneLightInfo1, ref context.env.lighting.sceneLightInfo1, percent,
                        ShaderIDs.Env_DirectionalSceneLightDir1, ShaderIDs.Env_DirectionalSceneLightColor1, true, notUseLightIntensity, false);
            }
        }
        public static IEnverimnentModify Load(XBinaryReader reader)
        {
            LightingModify lm = CFAllocator.Allocate<LightingModify>();
            lm.roleLightInfo0.lightDir = reader.ReadVector4();
            lm.roleLightInfo0.lightColor = reader.ReadVector4();
            lm.roleLightInfo1.lightDir = reader.ReadVector4();
            lm.roleLightInfo1.lightColor = reader.ReadVector4();

            lm.sceneLightInfo0.lightDir = reader.ReadVector4();
            lm.sceneLightInfo0.lightColor = reader.ReadVector4();
            lm.sceneLightInfo1.lightDir = reader.ReadVector4();
            lm.sceneLightInfo1.lightColor = reader.ReadVector4();
            return lm;
        }

#if UNITY_EDITOR
        public static void Save(BinaryWriter bw, IEnverimnentModify modify)
        {
            LightingModify lm = modify as LightingModify;
            EditorCommon.WriteVector(bw, lm.roleLightInfo0.lightDir);
            EditorCommon.WriteVector(bw, lm.roleLightInfo0.lightColor);
            EditorCommon.WriteVector(bw, lm.roleLightInfo1.lightDir);
            EditorCommon.WriteVector(bw, lm.roleLightInfo1.lightColor);
            EditorCommon.WriteVector(bw, lm.sceneLightInfo0.lightDir);
            EditorCommon.WriteVector(bw, lm.sceneLightInfo0.lightColor);
            EditorCommon.WriteVector(bw, lm.sceneLightInfo1.lightDir);
            EditorCommon.WriteVector(bw, lm.sceneLightInfo1.lightColor);
        }
#endif
    }

    [Serializable]
    public class AmbientModify : IEnverimnentModify, IEnverimnentLerp, ISceneObject
    {
        public AmbientMode ambientMode = AmbientMode.Flat;
        public Color ambientSkyColor;
        public Color ambientEquatorColor;
        public Color ambientGroundColor;
        public float AmbientMax = 1.0f;
        public string SkyboxMatPath = "";
        private ResHandle SkyBoxMat;
        private Cubemap SkyBox;
        private ProcessLoadCb processResCb;

        private bool valid = false;

        public EnverimentModifyType GetEnvType()
        {
            return EnverimentModifyType.Ambient;
        }
        public void Reset()
        {
            valid = false;
            SkyBox = null;
            LoadMgr.singleton.Destroy(ref SkyBoxMat, false);
        }
        public void Release()
        {
        }
        public static void OnTrigger(bool into)
        {
        }
        private void UpdateSkyBox(Material mat)
        {
            RenderSettings.skybox = mat;
            if (mat != null)
            {
                SkyBox = mat.GetTexture(ShaderIDs.Env_SkyCubeTex) as Cubemap;
                if (SkyBox != null)
                {
                    Shader.SetGlobalTexture(ShaderIDs.Env_SkyCube, SkyBox);
                }
            }
        }
        private void ProcessResCb(ref ResHandle resHandle, ref Vector4Int param)
        {
            if (resHandle.obj != null)
            {
                if (valid && resHandle.obj is Material)
                {
                    SkyBoxMat.Set(ref resHandle);
                    Material mat = resHandle.obj as Material;
                    UpdateSkyBox(mat);

                    return;
                }
                LoadMgr.singleton.Destroy(ref resHandle);
            }
        }

        public AmbientModify()
        {
#if UNITY_EDITOR
            needUpdate = true;
#endif
            processResCb = ProcessResCb;
        }
#if UNITY_EDITOR
        public bool needUpdate { get; set; }
#endif
        public void Lerp(ref EnverinmentContext context, float percent, bool into, int state)
        {
#if UNITY_EDITOR
            if (into)
                context.env.ambient.needUpdate = false;
            else if (state == 2)
                context.env.ambient.needUpdate = true;
#endif
            if (into)
            {
                if (RenderSettings.ambientMode == AmbientMode.Flat)
                {
                    RenderSettings.ambientLight = Color.Lerp(context.env.ambient.ambientSkyColor, ambientSkyColor, percent);
                }
                else
                {
                    RenderSettings.ambientSkyColor = Color.Lerp(context.env.ambient.ambientSkyColor, ambientSkyColor, percent);
                    RenderSettings.ambientEquatorColor = Color.Lerp(context.env.ambient.ambientEquatorColor, ambientEquatorColor, percent);
                    RenderSettings.ambientGroundColor = Color.Lerp(context.env.ambient.ambientGroundColor, ambientGroundColor, percent);
                }
                float ambientMax = Mathf.Lerp(context.env.ambient.AmbientMax, AmbientMax, percent);
                Shader.SetGlobalColor(ShaderIDs.Env_AmbientParam, new Vector4(ambientMax, 0, 0, 0));
                valid = true;
                if (!string.IsNullOrEmpty(SkyboxMatPath))
                {

                    string path = string.Format("{0}/{1}.mat", AssetsConfig.GlobalAssetsConfig.ResourcePath, SkyboxMatPath);
                    Material mat = AssetDatabase.LoadAssetAtPath<Material>(path);
                    UpdateSkyBox(mat);

                }

            }
            else
            {
                if (RenderSettings.ambientMode == AmbientMode.Flat)
                {
                    RenderSettings.ambientLight = Color.Lerp(ambientSkyColor, context.env.ambient.ambientSkyColor, percent);
                }
                else
                {
                    RenderSettings.ambientSkyColor = Color.Lerp(ambientSkyColor, context.env.ambient.ambientSkyColor, percent);
                    RenderSettings.ambientEquatorColor = Color.Lerp(ambientEquatorColor, context.env.ambient.ambientEquatorColor, percent);
                    RenderSettings.ambientGroundColor = Color.Lerp(ambientGroundColor, context.env.ambient.ambientGroundColor, percent);
                }

                float ambientMax = Mathf.Lerp(AmbientMax, context.env.ambient.AmbientMax, percent);
                Shader.SetGlobalColor(ShaderIDs.Env_AmbientParam, new Vector4(ambientMax, 0, 0, 0));
                context.env.UpdateSkyBox();
                Reset();
            }
        }

        public static IEnverimnentModify Load(XBinaryReader reader)
        {
            AmbientModify am = CFAllocator.Allocate<AmbientModify>();
            am.ambientSkyColor = reader.ReadVector4();
            am.ambientEquatorColor = reader.ReadVector4();
            am.ambientGroundColor = reader.ReadVector4();
            am.AmbientMax = reader.ReadSingle();
            am.SkyboxMatPath = reader.ReadString();
            return am;
        }

#if UNITY_EDITOR
        public static void Save(BinaryWriter bw, IEnverimnentModify modify)
        {
            AmbientModify am = modify as AmbientModify;
            EditorCommon.WriteVector(bw, am.ambientSkyColor);
            EditorCommon.WriteVector(bw, am.ambientEquatorColor);
            EditorCommon.WriteVector(bw, am.ambientGroundColor);
            bw.Write(am.AmbientMax);
            bw.Write(am.SkyboxMatPath);
        }
#endif
    }

    [Serializable]
    public class FogModify : IEnverimnentModify, IEnverimnentLerp, ISceneObject
    {
        public float Density = 0.015f;
        public float EndHeight = 0.0f;
        public float StartDistance = 6.0f;
        public float SkyboxHeight = 0.0f;
        public Color Color0 = new Color(0.447f, 0.638f, 1.0f);
        public Color Color1 = new Color(0.6f, 0.67f, 0.78f);
        public Color Color2 = new Color(0.447f, 0.638f, 1.0f);

        public EnverimentModifyType GetEnvType()
        {
            return EnverimentModifyType.Fog;
        }
        public void Reset()
        {

        }
        public void Release()
        {

        }
        public static void OnTrigger(bool into)
        {

        }
#if UNITY_EDITOR
        public FogModify()
        {
            needUpdate = true;
        }
        public bool needUpdate { get; set; }
#endif
        public void Lerp(ref EnverinmentContext context, float percent, bool into, int state)
        {
#if UNITY_EDITOR
            if (into)
                context.env.fog.needUpdate = false;
            else if (state == 2)
                context.env.fog.needUpdate = true;
#endif
            if (into)
            {
                Vector4 HeightFogParameters = new Vector4();
                HeightFogParameters.x = Mathf.Lerp(context.env.fog.Density, Density, percent);
                HeightFogParameters.y = Mathf.Lerp(context.env.fog.SkyboxHeight, SkyboxHeight, percent);
                HeightFogParameters.z = Mathf.Lerp(context.env.fog.EndHeight, EndHeight, percent);
                HeightFogParameters.w = Mathf.Lerp(context.env.fog.StartDistance, StartDistance, percent);

                Shader.SetGlobalVector(ShaderIDs.Env_HeightFogParameters, HeightFogParameters);

                Color color0 = Color.Lerp(context.env.fog.Color0.linear, Color0.linear, percent);
                Color color1 = Color.Lerp(context.env.fog.Color1.linear, Color1.linear, percent);
                Color color2 = Color.Lerp(context.env.fog.Color2.linear, Color2.linear, percent);

                Shader.SetGlobalVector(ShaderIDs.Env_HeighFogColorParameter0, color0);
                Shader.SetGlobalVector(ShaderIDs.Env_HeighFogColorParameter1, color1);
                Shader.SetGlobalVector(ShaderIDs.Env_HeighFogColorParameter2, color2);
            }
            else
            {
                Vector4 HeightFogParameters = new Vector4();
                HeightFogParameters.x = Mathf.Lerp(Density, context.env.fog.Density, percent);
                HeightFogParameters.y = Mathf.Lerp(SkyboxHeight, context.env.fog.SkyboxHeight, percent);
                HeightFogParameters.z = Mathf.Lerp(EndHeight, context.env.fog.EndHeight, percent);
                HeightFogParameters.w = Mathf.Lerp(StartDistance, context.env.fog.StartDistance, percent);

                Shader.SetGlobalVector(ShaderIDs.Env_HeightFogParameters, HeightFogParameters);

                Color color0 = Color.Lerp(Color0.linear, context.env.fog.Color0.linear, percent);
                Color color1 = Color.Lerp(Color1.linear, context.env.fog.Color1.linear, percent);
                Color color2 = Color.Lerp(Color2.linear, context.env.fog.Color2.linear, percent);

                Shader.SetGlobalVector(ShaderIDs.Env_HeighFogColorParameter0, color0);
                Shader.SetGlobalVector(ShaderIDs.Env_HeighFogColorParameter1, color1);
                Shader.SetGlobalVector(ShaderIDs.Env_HeighFogColorParameter2, color2);
            }
        }

        public static IEnverimnentModify Load(XBinaryReader reader)
        {
            FogModify fm = CFAllocator.Allocate<FogModify>();
            fm.Density = reader.ReadSingle();
            fm.EndHeight = reader.ReadSingle();
            fm.StartDistance = reader.ReadSingle();
            fm.SkyboxHeight = reader.ReadSingle();
            fm.Color0 = reader.ReadVector4();
            fm.Color1 = reader.ReadVector4();
            fm.Color2 = reader.ReadVector4();
            return fm;
        }

#if UNITY_EDITOR
        public static void Save(BinaryWriter bw, IEnverimnentModify modify)
        {
            FogModify fm = modify as FogModify;
            bw.Write(fm.Density);
            bw.Write(fm.EndHeight);
            bw.Write(fm.StartDistance);
            bw.Write(fm.SkyboxHeight);
            EditorCommon.WriteVector(bw, fm.Color0);
            EditorCommon.WriteVector(bw, fm.Color1);
            EditorCommon.WriteVector(bw, fm.Color2);

        }
#endif
    }

    [Serializable]
    public class BloomModify : IEnverimnentModify, IEnverimnentLerp, ISceneObject
    {
        public bool enabled = true;
        public float intensity = 0f;
        public float threshold = 1f;
        public float softKnee = 0.5f;
        public float diffusion = 4f;
        public Color color = Color.white;

        private EffectLerpContext enableParam;
        private EffectLerpContext intensityParam;
        private EffectLerpContext thresholdParam;
        private EffectLerpContext softKneeParam;
        private EffectLerpContext diffusionParam;
        private EffectLerpContext colorParam;

        public EnverimentModifyType GetEnvType()
        {
            return EnverimentModifyType.PPBloom;
        }
        public void Reset()
        {

        }
        public void Release()
        {

        }
        public static void OnTrigger(bool into)
        {

        }
#if UNITY_EDITOR
        public bool needUpdate { get; set; }
#endif
        public void Lerp(ref EnverinmentContext context, float percent, bool into, int state)
        {
            if (into)
            {
                if (state == 1)
                {
                    context.envMgr.GetEffectParamLerp(ref enableParam, (int)EPostffects.EBloom, 0);
                    context.envMgr.GetEffectParamLerp(ref intensityParam, (int)EPostffects.EBloom, 1);
                    context.envMgr.GetEffectParamLerp(ref thresholdParam, (int)EPostffects.EBloom, 2);
                    context.envMgr.GetEffectParamLerp(ref softKneeParam, (int)EPostffects.EBloom, 3);
                    context.envMgr.GetEffectParamLerp(ref diffusionParam, (int)EPostffects.EBloom, 4);
                    context.envMgr.GetEffectParamLerp(ref colorParam, (int)EPostffects.EBloom, 5);
                }
                enableParam.LerpTo(enabled ? 1 : -1, percent);
                intensityParam.LerpTo(intensity, percent);
                thresholdParam.LerpTo(threshold, percent);
                softKneeParam.LerpTo(softKnee, percent);
                diffusionParam.LerpTo(diffusion, percent);
                colorParam.LerpTo(color, percent);
            }
            else
            {
                if (state == 2)
                {
                    enableParam.Reset();
                    intensityParam.Reset();
                    thresholdParam.Reset();
                    softKneeParam.Reset();
                    diffusionParam.Reset();
                    colorParam.Reset();
                }
                else
                {
                    enableParam.LerpRecover(enabled ? 1 : -1, percent);
                    intensityParam.LerpRecover(intensity, percent);
                    thresholdParam.LerpRecover(threshold, percent);
                    softKneeParam.LerpRecover(softKnee, percent);
                    diffusionParam.LerpRecover(diffusion, percent);
                    colorParam.LerpRecover(color, percent);
                }
            }
        }

        public static IEnverimnentModify Load(XBinaryReader reader)
        {
            BloomModify bm = CFAllocator.Allocate<BloomModify>();
            bm.enabled = reader.ReadBoolean();
            bm.intensity = reader.ReadSingle();
            bm.threshold = reader.ReadSingle();
            bm.softKnee = reader.ReadSingle();
            bm.diffusion = reader.ReadSingle();
            bm.color = reader.ReadVector4();
            return bm;
        }

#if UNITY_EDITOR
        public static void Save(BinaryWriter bw, IEnverimnentModify modify)
        {
            BloomModify bm = modify as BloomModify;
            bw.Write(bm.enabled);
            bw.Write(bm.intensity);
            bw.Write(bm.threshold);
            bw.Write(bm.softKnee);
            bw.Write(bm.diffusion);
            EditorCommon.WriteVector(bw, bm.color);

        }
#endif
    }

    [Serializable]
    public class LutModify : IEnverimnentModify, IEnverimnentLerp, ISceneObject
    {
        public bool enabled = false;
        public float toneCurveToeStrength = 0;
        public float toneCurveToeLength = 0.5f;
        public float toneCurveShoulderStrength = 0;
        public float toneCurveShoulderLength = 0.5f;
        public float toneCurveShoulderAngle = 0;
        public float toneCurveGamma = 1;
        public float temperature = 1;
        public float tint = 1;
        public Color colorFilter = Color.white;
        public float hueShift = 1;
        public float saturation = 0;
        public float postExposure = 0; // HDR only
        public float contrast = 0;
        public float mixerRedOutRedIn = 100;
        public float mixerRedOutGreenIn = 0;
        public float mixerRedOutBlueIn = 0;
        public float mixerGreenOutRedIn = 0;
        public float mixerGreenOutGreenIn = 100f;
        public float mixerGreenOutBlueIn = 0;
        public float mixerBlueOutRedIn = 0;
        public float mixerBlueOutGreenIn = 0;
        public float mixerBlueOutBlueIn = 100f;
        public Vector4 lift = new Vector4(1f, 1f, 1f, 0f);
        public Vector4 gamma = new Vector4(1f, 1f, 1f, 0f);
        public Vector4 gain = new Vector4(1f, 1f, 1f, 0f);
        public string lutPath = "";

        private EffectLerpContext enableParam;
        private EffectLerpContext tonemapperParam;
        private EffectLerpContext toneCurveToeStrengthParam;
        private EffectLerpContext toneCurveToeLengthParam;
        private EffectLerpContext toneCurveShoulderStrengthParam;
        private EffectLerpContext toneCurveShoulderLengthParam;
        private EffectLerpContext toneCurveShoulderAngleParam;
        private EffectLerpContext toneCurveGammaParam;
        private EffectLerpContext temperatureParam;
        private EffectLerpContext tintParam;
        private EffectLerpContext colorFilterParam;
        private EffectLerpContext hueShiftParam;
        private EffectLerpContext saturationParam;
        private EffectLerpContext postExposureParam;
        private EffectLerpContext contrastParam;
        private EffectLerpContext mixerRedOutRedInParam;
        private EffectLerpContext mixerRedOutGreenInParam;
        private EffectLerpContext mixerRedOutBlueInParam;
        private EffectLerpContext mixerGreenOutRedInParam;
        private EffectLerpContext mixerGreenOutGreenInParam;
        private EffectLerpContext mixerGreenOutBlueInParam;
        private EffectLerpContext mixerBlueOutRedInParam;
        private EffectLerpContext mixerBlueOutGreenInParam;
        private EffectLerpContext mixerBlueOutBlueInParam;
        private EffectLerpContext liftParam;
        private EffectLerpContext gammaParam;
        private EffectLerpContext gainParam;

        private ResHandle lutTexHandle;
        private bool valid = false;
        private ProcessLoadCb processResCb;

        public LutModify()
        {
#if UNITY_EDITOR
            needUpdate = true;
#endif
            processResCb = ProcessResCb;
        }
        public EnverimentModifyType GetEnvType()
        {
            return EnverimentModifyType.PPLut;
        }
        public void Reset()
        {
            valid = false;
        }
        public void Release()
        {

        }
        public static void OnTrigger(bool into)
        {

        }
        private void ProcessResCb(ref ResHandle resHandle, ref Vector4Int param)
        {
            if (resHandle.obj != null)
            {
                if (valid && resHandle.obj is Material)
                {
                    lutTexHandle.Set(ref resHandle);
                    Material mat = resHandle.obj as Material;

                    return;
                }
                LoadMgr.singleton.Destroy(ref resHandle);
            }
        }
#if UNITY_EDITOR
        public bool needUpdate { get; set; }
#endif
        public void Lerp(ref EnverinmentContext context, float percent, bool into, int state)
        {
            int effectType = (int)EPostffects.EColorGrading;
            if (into)
            {
                if (state == 1)
                {
                    context.envMgr.GetEffectParamLerp(ref enableParam, effectType, 0);
                    context.envMgr.GetEffectParamLerp(ref tonemapperParam, effectType, 1);
                    context.envMgr.GetEffectParamLerp(ref toneCurveToeStrengthParam, effectType, 2);
                    context.envMgr.GetEffectParamLerp(ref toneCurveToeLengthParam, effectType, 3);
                    context.envMgr.GetEffectParamLerp(ref toneCurveShoulderStrengthParam, effectType, 4);
                    context.envMgr.GetEffectParamLerp(ref toneCurveShoulderLengthParam, effectType, 5);
                    context.envMgr.GetEffectParamLerp(ref toneCurveShoulderAngleParam, effectType, 6);
                    context.envMgr.GetEffectParamLerp(ref toneCurveGammaParam, effectType, 7);
                    context.envMgr.GetEffectParamLerp(ref temperatureParam, effectType, 8);
                    context.envMgr.GetEffectParamLerp(ref tintParam, effectType, 9);
                    context.envMgr.GetEffectParamLerp(ref colorFilterParam, effectType, 10);
                    context.envMgr.GetEffectParamLerp(ref hueShiftParam, effectType, 11);
                    context.envMgr.GetEffectParamLerp(ref saturationParam, effectType, 12);
                    context.envMgr.GetEffectParamLerp(ref postExposureParam, effectType, 13);
                    context.envMgr.GetEffectParamLerp(ref contrastParam, effectType, 14);
                    context.envMgr.GetEffectParamLerp(ref mixerRedOutRedInParam, effectType, 15);
                    context.envMgr.GetEffectParamLerp(ref mixerRedOutGreenInParam, effectType, 16);
                    context.envMgr.GetEffectParamLerp(ref mixerRedOutBlueInParam, effectType, 17);
                    context.envMgr.GetEffectParamLerp(ref mixerGreenOutRedInParam, effectType, 18);
                    context.envMgr.GetEffectParamLerp(ref mixerGreenOutGreenInParam, effectType, 19);
                    context.envMgr.GetEffectParamLerp(ref mixerGreenOutBlueInParam, effectType, 20);
                    context.envMgr.GetEffectParamLerp(ref mixerBlueOutRedInParam, effectType, 21);
                    context.envMgr.GetEffectParamLerp(ref mixerBlueOutGreenInParam, effectType, 22);
                    context.envMgr.GetEffectParamLerp(ref mixerBlueOutBlueInParam, effectType, 23);
                    context.envMgr.GetEffectParamLerp(ref liftParam, effectType, 24);
                    context.envMgr.GetEffectParamLerp(ref gammaParam, effectType, 25);
                    context.envMgr.GetEffectParamLerp(ref gainParam, effectType, 26);
                    valid = true;
                }
                enableParam.LerpTo(enabled ? 1 : -1, percent);
                toneCurveToeStrengthParam.LerpTo(toneCurveToeStrength, percent);
                toneCurveToeLengthParam.LerpTo(toneCurveToeLength, percent);
                toneCurveShoulderStrengthParam.LerpTo(toneCurveShoulderStrength, percent);
                toneCurveShoulderLengthParam.LerpTo(toneCurveShoulderLength, percent);
                toneCurveShoulderAngleParam.LerpTo(toneCurveShoulderAngle, percent);
                toneCurveGammaParam.LerpTo(toneCurveGamma, percent);
                temperatureParam.LerpTo(temperature, percent);
                tintParam.LerpTo(tint, percent);
                colorFilterParam.LerpTo(colorFilter, percent);
                hueShiftParam.LerpTo(hueShift, percent);
                saturationParam.LerpTo(saturation, percent);
                postExposureParam.LerpTo(postExposure, percent);
                contrastParam.LerpTo(contrast, percent);
                mixerRedOutRedInParam.LerpTo(mixerRedOutRedIn, percent);
                mixerRedOutGreenInParam.LerpTo(mixerRedOutGreenIn, percent);
                mixerRedOutBlueInParam.LerpTo(mixerRedOutBlueIn, percent);
                mixerGreenOutRedInParam.LerpTo(mixerGreenOutRedIn, percent);
                mixerGreenOutGreenInParam.LerpTo(mixerGreenOutGreenIn, percent);
                mixerGreenOutBlueInParam.LerpTo(mixerGreenOutBlueIn, percent);
                mixerBlueOutRedInParam.LerpTo(mixerBlueOutRedIn, percent);
                mixerBlueOutGreenInParam.LerpTo(mixerBlueOutGreenIn, percent);
                mixerBlueOutBlueInParam.LerpTo(mixerBlueOutBlueIn, percent);
                liftParam.LerpTo(lift, percent);
                gammaParam.LerpTo(gamma, percent);
                gainParam.LerpTo(gain, percent);

                context.envMgr.DirtyEffect(effectType);
            }
            else
            {
                if (state == 2)
                {
                    enableParam.Reset();
                    tonemapperParam.Reset();
                    toneCurveToeStrengthParam.Reset();
                    toneCurveToeLengthParam.Reset();
                    toneCurveShoulderStrengthParam.Reset();
                    toneCurveShoulderLengthParam.Reset();
                    toneCurveShoulderAngleParam.Reset();
                    toneCurveGammaParam.Reset();
                    temperatureParam.Reset();
                    tintParam.Reset();
                    colorFilterParam.Reset();
                    hueShiftParam.Reset();
                    saturationParam.Reset();
                    postExposureParam.Reset();
                    contrastParam.Reset();
                    mixerRedOutRedInParam.Reset();
                    mixerRedOutGreenInParam.Reset();
                    mixerRedOutBlueInParam.Reset();
                    mixerGreenOutRedInParam.Reset();
                    mixerGreenOutGreenInParam.Reset();
                    mixerGreenOutBlueInParam.Reset();
                    mixerBlueOutRedInParam.Reset();
                    mixerBlueOutGreenInParam.Reset();
                    mixerBlueOutBlueInParam.Reset();
                    liftParam.Reset();
                    gammaParam.Reset();
                    gainParam.Reset();
                }
                else
                {
                    enableParam.LerpRecover(enabled ? 1 : -1, percent);
                    toneCurveToeStrengthParam.LerpRecover(toneCurveToeStrength, percent);
                    toneCurveToeLengthParam.LerpRecover(toneCurveToeLength, percent);
                    toneCurveShoulderStrengthParam.LerpRecover(toneCurveShoulderStrength, percent);
                    toneCurveShoulderLengthParam.LerpRecover(toneCurveShoulderLength, percent);
                    toneCurveShoulderAngleParam.LerpRecover(toneCurveShoulderAngle, percent);
                    toneCurveGammaParam.LerpRecover(toneCurveGamma, percent);
                    temperatureParam.LerpRecover(temperature, percent);
                    tintParam.LerpRecover(tint, percent);
                    colorFilterParam.LerpRecover(colorFilter, percent);
                    hueShiftParam.LerpRecover(hueShift, percent);
                    saturationParam.LerpRecover(saturation, percent);
                    postExposureParam.LerpRecover(postExposure, percent);
                    contrastParam.LerpRecover(contrast, percent);
                    mixerRedOutRedInParam.LerpRecover(mixerRedOutRedIn, percent);
                    mixerRedOutGreenInParam.LerpRecover(mixerRedOutGreenIn, percent);
                    mixerRedOutBlueInParam.LerpRecover(mixerRedOutBlueIn, percent);
                    mixerGreenOutRedInParam.LerpRecover(mixerGreenOutRedIn, percent);
                    mixerGreenOutGreenInParam.LerpRecover(mixerGreenOutGreenIn, percent);
                    mixerGreenOutBlueInParam.LerpRecover(mixerGreenOutBlueIn, percent);
                    mixerBlueOutRedInParam.LerpRecover(mixerBlueOutRedIn, percent);
                    mixerBlueOutGreenInParam.LerpRecover(mixerBlueOutGreenIn, percent);
                    mixerBlueOutBlueInParam.LerpRecover(mixerBlueOutBlueIn, percent);
                    liftParam.LerpRecover(lift, percent);
                    gammaParam.LerpRecover(gamma, percent);
                    gainParam.LerpRecover(gain, percent);
                }
                context.envMgr.DirtyEffect(effectType);
            }
        }

        public static IEnverimnentModify Load(XBinaryReader reader)
        {
            LutModify lm = CFAllocator.Allocate<LutModify>();
            lm.enabled = reader.ReadBoolean();
            lm.toneCurveToeLength = reader.ReadSingle();
            lm.toneCurveShoulderStrength = reader.ReadSingle();
            lm.toneCurveShoulderLength = reader.ReadSingle();
            lm.toneCurveShoulderAngle = reader.ReadSingle();
            lm.toneCurveGamma = reader.ReadSingle();
            lm.temperature = reader.ReadSingle();
            lm.tint = reader.ReadSingle();
            lm.colorFilter = reader.ReadVector4();
            lm.hueShift = reader.ReadSingle();
            lm.saturation = reader.ReadSingle();
            lm.postExposure = reader.ReadSingle();
            lm.contrast = reader.ReadSingle();
            lm.mixerRedOutRedIn = reader.ReadSingle();
            lm.mixerRedOutGreenIn = reader.ReadSingle();
            lm.mixerRedOutBlueIn = reader.ReadSingle();
            lm.mixerGreenOutRedIn = reader.ReadSingle();
            lm.mixerGreenOutGreenIn = reader.ReadSingle();
            lm.mixerGreenOutBlueIn = reader.ReadSingle();
            lm.mixerBlueOutRedIn = reader.ReadSingle();
            lm.mixerBlueOutGreenIn = reader.ReadSingle();
            lm.mixerBlueOutBlueIn = reader.ReadSingle();
            lm.lift = reader.ReadVector4();
            lm.gamma = reader.ReadVector4();
            lm.gain = reader.ReadVector4();

            return lm;
        }

    }
}