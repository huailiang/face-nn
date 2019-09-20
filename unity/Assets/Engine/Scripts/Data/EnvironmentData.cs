using CFUtilPoolLib;
using System;
using UnityEngine;


namespace XEngine
{

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

    [Serializable]
    public class LightingModify : IEnverimnentModify
    {
        public LightInfo roleLightInfo0 = LightInfo.DefaultAsset;
        public LightInfo roleLightInfo1 = LightInfo.DefaultAsset;

        public EnverimentModifyType GetEnvType()
        {
            return EnverimentModifyType.Lighting;
        }
    }


    [Serializable]
    public class FogModify : IEnverimnentModify
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
    }
}