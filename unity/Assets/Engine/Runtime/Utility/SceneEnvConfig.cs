#if UNITY_EDITOR
using System;
using System.Collections.Generic;

using UnityEngine;
using UnityEngine.Rendering;
namespace CFEngine
{

    public class SceneEnvConfig : ScriptableObject
    {

        //IBL
        public string EnveriomentCubePath;
        public float hdrScale = 4.6f;
        public float hdrPow = 0.1f;
        public float hdrAlpha = 0.5f;
        public float lightmapShadowMask = 0.25f;
        public float shadowIntensity = 0.1f;
        //Camera Light
        public float CameraLightAtten = 1f;
        public float CameraLightSqDistance = 10.0f;
        //Lighting
        public LightingModify lighting = new LightingModify();
        public string SkyboxMatPath;
        public AmbientModify ambient = new AmbientModify();
        // public float AmbientMax = 1.0f;

        public Vector3 sunDir = new Vector3(0, -1, 0);
        //Shadow
        public float shadowScale = 0.5f;
        public float shadowDepthBias = -0.03f;
        public float shadowNormalBias = 2.5f;
        public float shadowSmoothMin = 4f;
        public float shadowSmoothMax = 1f;
        public float shadowSampleSize = 1.2f;
        public float shadowPower = 2f;
        public bool fogEnable = true;
        public FogModify fog = new FogModify();
        public bool enableWind = true;
        public RandomWindModify randomWind = new RandomWindModify();
        public WaveWindModify waveWind = new WaveWindModify();
        public float interactiveParam = 1;

        public CameraClearFlags clearFlag = CameraClearFlags.Skybox;
        public Color clearColor = Color.black;

    }
}
#endif