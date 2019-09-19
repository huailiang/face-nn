using CFUtilPoolLib;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace XEngine
{
    [DisallowMultipleComponent, ExecuteInEditMode]
    [RequireComponent(typeof(Camera))]
    public class RenderingEnvironment : MonoBehaviour
    {
        //IBL
        public string EnveriomentCubePath;
        public float hdrScale = 4.6f;
        public float hdrPow = 0.1f;
        public float hdrAlpha = 0.5f;
        [XEngine.RangeAttribute(0, 1)]
        public float lightmapShadowMask = 0.25f;
        [XEngine.RangeAttribute(0, 1)]
        public float shadowIntensity = 0.1f;
        //Lighting
        public LightingModify lighting = null;
        public string SkyboxMatPath;
        public Vector3 sunDir = new Vector3(0, -1, 0);

        public AmbientModify ambient = null;
        //Shadow
        public float shadowDepthBias = -0.03f;
        public float shadowNormalBias = 2.5f;
        public float shadowSmoothMin = 4f;
        public float shadowSmoothMax = 1f;
        public float shadowSampleSize = 1.2f;
        public float shadowPower = 2f;
        public FogModify fog = null;
        public bool fogEnable = true;
        [NoSerialized]
        public SceneData sceneData = null;

        private ResHandle EnveriomentCube;
        private ResHandle SkyBoxMat;
        private Cubemap SkyBox;
        //shadow
        [System.NonSerialized]
        public Vector3 cameraForward;
        [System.NonSerialized]
        public Vector3 lightProjectRight;
        [System.NonSerialized]
        public Vector3 lightProjectUp;
        [System.NonSerialized]
        public Vector3 lightProjectForward;
        [System.NonSerialized]
        public Vector3 translatePos;

#if UNITY_EDITOR
        public static bool isPreview = false;
        [System.NonSerialized]
        public float shadowOrthoSize;
        [System.NonSerialized]
        public bool saveingFile = false;
#endif

        void Awake()
        {
            SceneData.GlobalSceneData = sceneData;
        }

        private void SetLightInfo(ref LightInfo li, int dirKey, int colorKey, bool setLightColor = true)
        {
            Shader.SetGlobalVector(dirKey, li.lightDir);
            if (setLightColor)
            {
                Vector4 lightColorIntensity;
                lightColorIntensity = new Vector4(
                    Mathf.Pow(li.lightColor.r * li.lightDir.w, 2.2f),
                    Mathf.Pow(li.lightColor.g * li.lightDir.w, 2.2f),
                    Mathf.Pow(li.lightColor.b * li.lightDir.w, 2.2f), shadowIntensity);
                Shader.SetGlobalVector(colorKey, lightColorIntensity);
            }
        }
        
        private void ProcessResCb(ref ResHandle resHandle, ref Vector4Int param)
        {
            if (resHandle.obj != null)
            {
                if (resHandle.obj is Cubemap)
                {
                    EnveriomentCube.Set(ref resHandle);
                    return;
                }
                else if (resHandle.obj is Material)
                {
                    SkyBoxMat.Set(ref resHandle);
                    return;
                }
                LoadMgr.singleton.Destroy(ref resHandle);
            }
        }

        public void LoadRes(bool loadEnvCube = true, bool loadSkyBox = true)
        {
            ProcessLoadCb processResCb = ProcessResCb;
            if (loadEnvCube && !string.IsNullOrEmpty(EnveriomentCubePath))
            {
                string suffix = EnveriomentCubePath.EndsWith("HDR") ? ".exr" : ".tga";
                string path = string.Format("{0}/{1}{2}", AssetsConfig.GlobalAssetsConfig.ResourcePath, EnveriomentCubePath, suffix);
                EnveriomentCube.obj = AssetDatabase.LoadAssetAtPath<Cubemap>(path);
            }
            if (loadSkyBox && !string.IsNullOrEmpty(SkyboxMatPath))
            {
                string path = string.Format("{0}/{1}.mat", AssetsConfig.GlobalAssetsConfig.ResourcePath, SkyboxMatPath);
                SkyBoxMat.obj = AssetDatabase.LoadAssetAtPath<Material>(path);
            }
        }

        public void InitRender(UnityEngine.Camera camera)
        {
            sceneData.CameraRef = camera;
            sceneData.CameraTransCache = camera.transform;
            if (lighting == null)
                lighting = new LightingModify();
            if (ambient == null)
                ambient = new AmbientModify();
            if (fog == null)
                fog = new FogModify();

            lighting.needUpdate = true;
            ambient.needUpdate = true;
            fog.needUpdate = true;

            RuntimeUtilities.EnableKeyword(ShaderIDs.Weather_ThunderKeyWord, false);
            RuntimeUtilities.EnableKeyword(ShaderIDs.Weather_RainbowKeyWord, false);
            RuntimeUtilities.EnableKeyword(ShaderIDs.Weather_RainEffectKeyWord, false);
            RuntimeUtilities.EnableKeyword(ShaderIDs.Weather_StarKeyWord, false);
            RuntimeUtilities.EnableKeyword(ShaderIDs.Weather_WeatherKeyWord, false);

            Shader.SetGlobalColor(ShaderIDs.Env_EffectParameter, Color.white);
            LoadRes();
            UpdateEnv();
            
            EnverinmentExtra ee = GetComponent<EnverinmentExtra>();
            if (ee != null) ee.Init();
        }

        public void Uninit()
        {
            LoadMgr.singleton.Destroy(ref EnveriomentCube, false);
            SkyBox = null;
            LoadMgr.singleton.Destroy(ref SkyBoxMat, false);
        }

        public void UpdateSkyBox()
        {
            if (SkyBoxMat.obj is Material)
            {
                Material mat = SkyBoxMat.obj as Material;
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
        }

        public void UpdateEnv()
        {
            //IBL
            float maxMipmap = 1;
            if (EnveriomentCube.obj is Cubemap)
            {
                Cubemap envCube = EnveriomentCube.obj as Cubemap;
                Shader.SetGlobalTexture(ShaderIDs.Env_Cubemap, envCube);

                if (envCube != null)
                {
                    maxMipmap = envCube.mipmapCount;
                }
            }
            Shader.SetGlobalVector(ShaderIDs.Env_CubemapParam, new Vector4(hdrScale, hdrPow, hdrAlpha, maxMipmap));
            Shader.SetGlobalVector(ShaderIDs.Env_LightmapScale, new Vector4(1.0f / lightmapShadowMask, lightmapShadowMask, shadowIntensity, 0));

            if (ambient.needUpdate)
            {
                UpdateSkyBox();
            }
            if (lighting.needUpdate)
            {
                SetLightInfo(ref lighting.roleLightInfo0, ShaderIDs.Env_DirectionalLightDir, ShaderIDs.Env_DirectionalLightColor);
                SetLightInfo(ref lighting.roleLightInfo1, ShaderIDs.Env_DirectionalLightDir1, ShaderIDs.Env_DirectionalLightColor1);
            }
            if (ambient.needUpdate)
                Shader.SetGlobalColor(ShaderIDs.Env_AmbientParam, new Vector4(ambient.AmbientMax, 0, 0, 0));
            //Shadow
            Shader.SetGlobalVector(ShaderIDs.Env_ShadowBias, new Vector4(shadowDepthBias, shadowNormalBias, 0, 0));
            Shader.SetGlobalVector(ShaderIDs.Env_ShadowSmooth, new Vector4(shadowSmoothMin * -0.0001f, shadowSmoothMax * 0.0001f, shadowSampleSize, shadowPower));
            if (sceneData.ShadowRT != null)
            {
                int halfSize = (int)(sceneData.ShadowRT.width * 0.5f);
                Shader.SetGlobalVector(ShaderIDs.Env_ShadowMapSize, new Vector4(halfSize, 1.0f / halfSize, 0, 0));
                Shader.SetGlobalTexture(ShaderIDs.Env_ShadowMapTex, sceneData.ShadowRT);
            }
            //Fog
            if (fog.needUpdate)
            {
                Vector4 HeightFogParameters = new Vector4();
                HeightFogParameters.x = fog.Density;
                HeightFogParameters.y = fog.SkyboxHeight;
                HeightFogParameters.z = fog.EndHeight;
                HeightFogParameters.w = fog.StartDistance;

                Shader.SetGlobalVector(ShaderIDs.Env_HeightFogParameters, HeightFogParameters);
                Shader.SetGlobalVector(ShaderIDs.Env_HeighFogColorParameter0, fog.Color0.linear);
                Shader.SetGlobalVector(ShaderIDs.Env_HeighFogColorParameter1, fog.Color1.linear);
                Shader.SetGlobalVector(ShaderIDs.Env_HeighFogColorParameter2, fog.Color2.linear);
            }
        }

        private void UpdateCameraInternal()
        {
            if (sceneData.CameraRef != null)
            {
                GeometryUtility.CalculateFrustumPlanes(sceneData.CameraRef, SceneData.frustumPlanes);
                Shader.SetGlobalVector(ShaderIDs.Env_GameViewCameraPos, sceneData.cameraPos);
            }
        }

  
        public void ManualUpdate()
        {
            if (sceneData.CameraRef == null)
            {
                InitRender(this.GetComponent<Camera>());
            }
            UpdateCameraInternal();
        }
        

        void Update()
        {
            if (!saveingFile)
            {
                if (lighting != null && fog != null)
                {
                    UpdateEnv();
                    bool hasFog = fogEnable;
                    if (SceneView.lastActiveSceneView != null)
                    {
                        var sceneViewState = SceneView.lastActiveSceneView.sceneViewState;
                        hasFog &= sceneViewState.showFog;
                    }
                    Shader.SetGlobalFloat(ShaderIDs.Env_FogDisable, !hasFog ? 1.0f : 0.0f);
                }
                ManualUpdate();
            }
        }

        public void SyncGameCamera()
        {
            if (SceneView.lastActiveSceneView != null && SceneView.lastActiveSceneView.camera != null &&
                sceneData.CameraTransCache != null)
            {
                Transform t = SceneView.lastActiveSceneView.camera.transform;
                sceneData.CameraTransCache.position = t.position;
                sceneData.CameraTransCache.rotation = t.rotation;
                sceneData.CameraTransCache.localScale = t.localScale;
            }
        }

        public void SyncLight(Light l, ref LightInfo li)
        {
            li.lightDir = l.transform.rotation * -Vector3.forward;
            li.lightColor = l.color;
            li.lightDir.w = (l.enabled && l.gameObject.activeInHierarchy) ? l.intensity : 0;
        }

    }

}
