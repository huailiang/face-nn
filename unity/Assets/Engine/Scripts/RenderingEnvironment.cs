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
        [CFRange(0, 1)]
        public float lightmapShadowMask = 0.25f;
        [CFRange(0, 1)]
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
        [CFNoSerialized]
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
        [System.NonSerialized]
        public bool fitWorldSpace = true;

#if UNITY_EDITOR
        [System.NonSerialized]
        public bool textureFolder = true;
        [System.NonSerialized]
        public bool envLighingFolder = true;
        [System.NonSerialized]
        public bool fogFolder = true;
        public static bool isPreview = false;
        [System.NonSerialized]
        public float shadowOrthoSize;
        [System.NonSerialized]
        public bool saveingFile = false;

#endif

        void Awake()
        {
            SceneData.GlobalSceneData = sceneData;
            RenderingManager.instance.renderingEnvironment = this;
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

        private void CalcLightMatrix(Bounds shadowBound, Vector3 looktargetCenter, float centerOffset, float shadowScale, float size, out Matrix4x4 viewMatrix, out Matrix4x4 projMatrix)
        {
            Vector3 lightPos = looktargetCenter - lightProjectForward * 10;
            translatePos = shadowBound.center - lightProjectForward * 10;
            translatePos.y = lightPos.y;
            translatePos.x = lightPos.x * centerOffset + translatePos.x * (1 - centerOffset);
            translatePos.z = lightPos.z * centerOffset + translatePos.z * (1 - centerOffset);

            Matrix4x4 translate = Matrix4x4.Translate(-translatePos);
            viewMatrix = Matrix4x4.identity;

            viewMatrix.SetColumn(0, lightProjectRight);
            viewMatrix.SetColumn(1, lightProjectUp);
            viewMatrix.SetColumn(2, -lightProjectForward);

            viewMatrix = viewMatrix.transpose;
            viewMatrix *= translate;

            Vector3 shadowBoxSize = shadowBound.size;
            float scale = shadowBoxSize.x > shadowBoxSize.z ? shadowBoxSize.x : shadowBoxSize.z;
            scale = (scale + 0.01f) / size;
            float shadowSize = size * scale * shadowScale;

            Matrix4x4 ortho = Matrix4x4.Ortho(-shadowSize, shadowSize, -shadowSize, shadowSize, 0.1f, 50);
            projMatrix = ortho;
#if UNITY_EDITOR
            shadowOrthoSize = shadowSize;
#endif
        }

        private void UpdateLightVPWorldSpace()
        {
            lightProjectUp = Vector3.forward;
            lightProjectRight = Vector3.Cross(lightProjectUp, lightProjectForward);
            lightProjectUp = Vector3.Cross(lightProjectForward, lightProjectRight);
            Vector4 shadowParam = sceneData.shadowParam;
            CalcLightMatrix(sceneData.shadowBound, sceneData.lockAtPos, shadowParam.x, shadowParam.w, shadowParam.y, out sceneData.shadowViewMatrix, out sceneData.shadowProjMatrix);
            Shader.SetGlobalMatrix(ShaderIDs.Env_ShadowMapView, GL.GetGPUProjectionMatrix(sceneData.shadowProjMatrix, false) * sceneData.shadowViewMatrix);
            if (sceneData.enableShadowCsm)
            {
                float dist = sceneData.shadowParam.z - sceneData.shadowParam.y;
                Vector3 center = sceneData.lockAtPos;
                if (cameraForward.z > 0)
                {
                    if (cameraForward.z >= Mathf.Abs(cameraForward.x))
                    {
                        center += dist * Vector3.forward;
                    }
                    else if (cameraForward.x > 0)
                    {
                        center += dist * Vector3.right;
                    }
                    else
                    {
                        center -= dist * Vector3.right;
                    }
                }
                else
                {
                    if (Mathf.Abs(cameraForward.z) >= Mathf.Abs(cameraForward.x))
                    {
                        center -= dist * Vector3.forward;
                    }
                    else if (cameraForward.x > 0)
                    {
                        center += dist * Vector3.right;
                    }
                    else
                    {
                        center -= dist * Vector3.right;
                    }
                }
                CalcLightMatrix(sceneData.shadowBound1, center, shadowParam.x, shadowParam.w, shadowParam.z, out sceneData.shadowViewMatrix1, out sceneData.shadowProjMatrix1);
                Shader.SetGlobalMatrix(ShaderIDs.Env_ShadowMapView1, GL.GetGPUProjectionMatrix(sceneData.shadowProjMatrix1, false) * sceneData.shadowViewMatrix1);
            }
        }

        private void UpdateLightVPCameraSpace()
        {
            Vector3 looktargetCenter = sceneData.currentEntityPos;
            float delta = (looktargetCenter - sceneData.cameraPos).sqrMagnitude;
            if (delta < 0.01f)
            {
                looktargetCenter += Vector3.up;
                lightProjectUp = cameraForward;
            }
            else
            {
                lightProjectUp = sceneData.currentEntityPos - sceneData.cameraPos;
            }

            lightProjectUp.y = 0;
            lightProjectUp.Normalize();
            lightProjectRight = Vector3.Cross(lightProjectUp, lightProjectForward);
            lightProjectUp = Vector3.Cross(lightProjectForward, lightProjectRight);

            Vector4 shadowParam = sceneData.shadowParam;
            CalcLightMatrix(sceneData.shadowBound, sceneData.currentEntityPos, sceneData.shadowParam.x, shadowParam.w, sceneData.shadowParam.y, out sceneData.shadowViewMatrix, out sceneData.shadowProjMatrix);
            Shader.SetGlobalMatrix(ShaderIDs.Env_ShadowMapView, GL.GetGPUProjectionMatrix(sceneData.shadowProjMatrix, false) * sceneData.shadowViewMatrix);
            if (sceneData.enableShadowCsm)
            {
                Vector3 forward = cameraForward;
                forward.y = 0;
                forward.Normalize();
                float dist = sceneData.shadowParam.z - sceneData.shadowParam.y;
                Vector3 center = sceneData.currentEntityPos + dist * forward;
                CalcLightMatrix(sceneData.shadowBound1, center, sceneData.shadowParam.x, shadowParam.w, sceneData.shadowParam.z, out sceneData.shadowViewMatrix1, out sceneData.shadowProjMatrix1);
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
#if UNITY_EDITOR
            lighting.needUpdate = true;
            ambient.needUpdate = true;
            fog.needUpdate = true;
#endif

            RuntimeUtilities.EnableKeyword(ShaderIDs.Weather_ThunderKeyWord, false);
            RuntimeUtilities.EnableKeyword(ShaderIDs.Weather_RainbowKeyWord, false);
            RuntimeUtilities.EnableKeyword(ShaderIDs.Weather_RainEffectKeyWord, false);
            RuntimeUtilities.EnableKeyword(ShaderIDs.Weather_StarKeyWord, false);
            RuntimeUtilities.EnableKeyword(ShaderIDs.Weather_WeatherKeyWord, false);

            Shader.SetGlobalColor(ShaderIDs.Env_EffectParameter, Color.white);
            LoadRes();
            UpdateEnv();

#if UNITY_EDITOR

            EnverinmentExtra ee = GetComponent<EnverinmentExtra> ();
            if (ee != null)
            {
                ee.Init ();
            }
#endif

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
#if UNITY_EDITOR
            if (ambient.needUpdate)
#endif
            {
                UpdateSkyBox();
            }



#if UNITY_EDITOR
         if (lighting.needUpdate)
#endif
            {
                SetLightInfo(ref lighting.roleLightInfo0, ShaderIDs.Env_DirectionalLightDir, ShaderIDs.Env_DirectionalLightColor);
                SetLightInfo(ref lighting.roleLightInfo1, ShaderIDs.Env_DirectionalLightDir1, ShaderIDs.Env_DirectionalLightColor1);
            }
#if UNITY_EDITOR
                if (ambient.needUpdate)
#endif
            Shader.SetGlobalColor(ShaderIDs.Env_AmbientParam, new Vector4(ambient.AmbientMax, 0, 0, 0));


#if UNITY_EDITOR
            if (lighting.needUpdate)
#endif
            {
                SetLightInfo(ref lighting.sceneLightInfo0, ShaderIDs.Env_DirectionalSceneLightDir, ShaderIDs.Env_DirectionalSceneLightColor);
                SetLightInfo(ref lighting.sceneLightInfo1, ShaderIDs.Env_DirectionalSceneLightDir1, ShaderIDs.Env_DirectionalSceneLightColor1);
            }
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
#if UNITY_EDITOR
            if (fog.needUpdate)
#endif
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

                float dist = -5;
                Vector3 sunPos = sunDir * 10 + sceneData.CameraTransCache.position;
                int inrange = 1;
                for (int i = 0; i < SceneData.frustumPlanes.Length; ++i)
                {
                    Plane p = SceneData.frustumPlanes[i];
                    float distance2Plane = XCommon.Point2PlaneDistance(ref sunPos, p.normal, p.distance);
                    if (distance2Plane <= 0)
                    {
                        if (distance2Plane < dist)
                        {
                            inrange = -1;
                            break;
                        }
                    }
                }
                RenderingManager.instance.sunForward = new Vector4(sunDir.x, sunDir.y, sunDir.z, inrange);
                if (sceneData.enableShadow)
                {
                    lightProjectForward = -lighting.roleLightInfo0.lightDir;
                    cameraForward = sceneData.CameraTransCache.forward;

                    if (fitWorldSpace)
                    {
                        UpdateLightVPWorldSpace();
                    }
                    else
                    {
                        UpdateLightVPCameraSpace();
                    }
                }
            }
        }

        private void UpdateEnvArea(float deltaTime)
        {
            if (sceneData.globalObjectsRef.IsValid() &&
                sceneData.currentEnv != null && sceneData.envLerpTime > -0.5f)
            {
                float lerpPercent = 0;
                int state = 0;
                if (sceneData.envLerpLength <= 0)
                {
                    state = 2;
                    sceneData.envLerpTime = sceneData.envLerpLength;
                    lerpPercent = 1;
                }
                else
                {
                    if (sceneData.envLerpTime < 0.001f)
                    {
                        state = 1;//start
                    }
                    else if (sceneData.envLerpTime > sceneData.envLerpLength)
                    {
                        state = 2;
                        sceneData.envLerpTime = sceneData.envLerpLength;
                        lerpPercent = 1;
                    }
                    else
                    {
                        lerpPercent = sceneData.envLerpTime / sceneData.envLerpLength;
                    }
                }

                EnverinmentContext context;
                context.env = this;
                context.envMgr = RenderingManager.instance;
                for (int i = sceneData.currentEnv.envModifyStart; i < sceneData.currentEnv.envModifyEnd; ++i)
                {
                    var envLerp = sceneData.globalObjectsRef.Get<ISceneObject>(i) as IEnverimnentLerp;
                    if (envLerp != null)
                    {
                        envLerp.Lerp(ref context, lerpPercent, sceneData.intoEnvArea, state);
                    }
                }
                if (state == 2)
                {
                    sceneData.currentEnv = null;
                    sceneData.envLerpTime = -1;
                }
                else
                    sceneData.envLerpTime += deltaTime;
            }
        }

        public void ManualUpdate()
        {
#if UNITY_EDITOR
            if (sceneData.CameraRef == null)
            {
                InitRender (this.GetComponent<Camera> ());
            }
#endif
            UpdateCameraInternal();
            float deltaTime = Time.deltaTime;
            UpdateEnvArea(deltaTime);
        }

#if UNITY_EDITOR

        void Update()
        {
            if (!saveingFile)
            {
                if (lighting != null && fog != null )
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

        public void SyncGameCamera ()
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

        public void SyncLight (Light l, ref LightInfo li)
        {
            li.lightDir = l.transform.rotation * -Vector3.forward;
            li.lightColor = l.color;
            li.lightDir.w = (l.enabled && l.gameObject.activeInHierarchy) ? l.intensity : 0;
        }

#endif
    }
}
