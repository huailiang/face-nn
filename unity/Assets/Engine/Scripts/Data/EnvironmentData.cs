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
        public Environment env;
    }

    public interface IEnverimnentLerp
    {
        void Lerp(ref EnverinmentContext context, float percent, bool into, int state);//0 runing 1 begin 2 finish
        bool needUpdate { get; set; }
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
    public class LightingModify : IEnverimnentModify, IEnverimnentLerp
    {
        public LightInfo roleLightInfo0 = LightInfo.DefaultAsset;
        public LightInfo roleLightInfo1 = LightInfo.DefaultAsset;

        public EnverimentModifyType GetEnvType()
        {
            return EnverimentModifyType.Lighting;
        }
        public static void OnTrigger(bool into)
        {
        }
        public LightingModify()
        {
            needUpdate = true;
        }
        public bool needUpdate { get; set; }

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
            }
            else
            {
                if (roleLightInfo0.lightDir.w > 0)
                    LerpLightInfo(ref roleLightInfo0, ref context.env.lighting.roleLightInfo0, percent,
                        ShaderIDs.Env_DirectionalLightDir, ShaderIDs.Env_DirectionalLightColor);

                if (roleLightInfo1.lightDir.w > 0)
                    LerpLightInfo(ref roleLightInfo1, ref context.env.lighting.roleLightInfo1, percent,
                        ShaderIDs.Env_DirectionalLightDir1, ShaderIDs.Env_DirectionalLightColor1);
            }
        }
    }

    [Serializable]
    public class AmbientModify : IEnverimnentModify, IEnverimnentLerp
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
            needUpdate = true;
            processResCb = ProcessResCb;
        }
        public bool needUpdate { get; set; }
        public void Lerp(ref EnverinmentContext context, float percent, bool into, int state)
        {
            if (into)
                context.env.ambient.needUpdate = false;
            else if (state == 2)
                context.env.ambient.needUpdate = true;
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
    }

    [Serializable]
    public class FogModify : IEnverimnentModify, IEnverimnentLerp
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
        public static void OnTrigger(bool into)
        {
        }
        public FogModify()
        {
            needUpdate = true;
        }
        public bool needUpdate { get; set; }
        public void Lerp(ref EnverinmentContext context, float percent, bool into, int state)
        {
            if (into)
                context.env.fog.needUpdate = false;
            else if (state == 2)
                context.env.fog.needUpdate = true;
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

    }
}