using CFUtilPoolLib;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace XEngine
{

    [DisallowMultipleComponent, ExecuteInEditMode]
    [RequireComponent(typeof(Camera))]
    public class Environment : MonoBehaviour
    {
        //IBL
        public float hdrScale = 4.6f;
        public float hdrPow = 0.1f;
        public float hdrAlpha = 0.5f;
        public int iblLevel = 1;

        [XEngine.RangeAttribute(0, 1)]
        public float lightmapShadowMask = 0.25f;
        [XEngine.RangeAttribute(0, 1)]
        public float shadowIntensity = 0.1f;
        //Lighting
        public LightingModify lighting = null;
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
        public Cubemap envCube;
        public Material SkyBoxMat;
        private Cubemap SkyBox;
        //shadow
        [System.NonSerialized]
        public Vector3 lightProjectRight;
        [System.NonSerialized]
        public Vector3 lightProjectUp;
        [System.NonSerialized]
        public Vector3 lightProjectForward;
        [System.NonSerialized]
        public Vector3 translatePos;
        [System.NonSerialized]
        public float shadowOrthoSize;
        public bool lightingFolder = true;
        public Light roleLight0;
        [System.NonSerialized]
        public TransformRotationGUIWrapper roleLight0Rot;
        public Light roleLight1;
        [System.NonSerialized]
        public TransformRotationGUIWrapper roleLight1Rot;
        public bool shadowFolder = true;
        public float shadowMapLevel = 0.25f;
        public bool shadowBound = false;
        public Transform lookTarget;
        public bool drawShadowLighing = false;

        [System.NonSerialized]
        public RenderTexture shadowMap = null;
        private List<Renderer> shadowCasters = new List<Renderer>();
        private CommandBuffer shadowMapCb = null;
        private List<RenderBatch> shadowRenderBatchs = new List<RenderBatch>();
        private Material shadowMat = null;


        public bool debugFolder = true;
        public bool showObjects = false;

        public ShaderDebugContext debugContext = new ShaderDebugContext();

        public static int[] debugShaderIDS = new int[]
        {
            Shader.PropertyToID ("_GlobalDebugMode"),
            Shader.PropertyToID ("_DebugDisplayType"),
            Shader.PropertyToID ("_SplitAngle"),
            Shader.PropertyToID ("_SplitPos"),
        };


        void Awake()
        {
            SceneData.GlobalSceneData = sceneData;
            Shader.SetGlobalFloat("_GlobalDebugMode", 0);
        }

        public void Update()
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
            if (sceneData.CameraRef == null)
            {
                InitRender(this.GetComponent<Camera>());
            }
            if (sceneData.CameraRef != null)
            {
                GeometryUtility.CalculateFrustumPlanes(sceneData.CameraRef, SceneData.frustumPlanes);
                Shader.SetGlobalVector(ShaderIDs.Env_GameViewCameraPos, sceneData.cameraPos);
            }
            SyncLightInfo();
            BuildShadowMap();
            debugContext.Refresh();
        }

        void OnDestroy()
        {
            if (shadowMapCb != null)
            {
                shadowMapCb.Release();
                shadowMapCb = null;
            }
            if (shadowMap != null)
            {
                if (Application.isPlaying)
                    UnityEngine.Object.Destroy(shadowMap);
                else
                    UnityEngine.Object.DestroyImmediate(shadowMap);
                shadowMap = null;
            }
        }

        public void InitRender(UnityEngine.Camera camera)
        {
            sceneData.CameraRef = camera;
            sceneData.CameraTransCache = camera.transform;
            if (lighting == null) lighting = new LightingModify();
            if (fog == null) fog = new FogModify();

            UpdateEnv();
            if (shadowMapCb == null)
                shadowMapCb = new CommandBuffer { name = "Editor Shadow Map Cb" };
        }


        private void PrepareTransformGui(Light light, ref TransformRotationGUIWrapper wrapper)
        {
            if (light != null && (wrapper == null || wrapper.t != light.transform))
            {
                wrapper = EditorCommon.GetTransformRotatGUI(light.transform);
            }
        }

        private void BuildShadowMap()
        {
            if (shadowMap != null && shadowMat != null)
            {
                shadowMapCb.Clear();
                shadowMapCb.ClearRenderTarget(true, true, Color.clear, 1.0f);
                shadowMapCb.SetViewProjectionMatrices(sceneData.shadowViewMatrix, sceneData.shadowProjMatrix);

                if (shadowCasters.Count > 0)
                {
                    for (int i = 0; i < shadowCasters.Count; ++i)
                    {
                        Renderer render = shadowCasters[i];
                        if (render != null &&
                            render.enabled &&
                            render.shadowCastingMode != ShadowCastingMode.Off)
                            shadowMapCb.DrawRenderer(render, shadowMat, 0, 0);
                    }
                    Graphics.SetRenderTarget(shadowMap);
                    Graphics.ExecuteCommandBuffer(shadowMapCb);
                    Graphics.SetRenderTarget(null);
                }
            }
        }

        void OnDrawGizmos()
        {
            Color color = Gizmos.color;
            if (drawShadowLighing)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireSphere(translatePos, 0.3f);
                Vector3 targetPos = translatePos + lightProjectForward * 10;
                Gizmos.DrawWireSphere(targetPos, 0.3f);
                Vector3 leftUp = translatePos + lightProjectUp * shadowOrthoSize - lightProjectRight * shadowOrthoSize;
                Vector3 rightUp = translatePos + lightProjectUp * shadowOrthoSize + lightProjectRight * shadowOrthoSize;
                Vector3 leftBottom = translatePos - lightProjectUp * shadowOrthoSize - lightProjectRight * shadowOrthoSize;
                Vector3 rightBottom = translatePos - lightProjectUp * shadowOrthoSize + lightProjectRight * shadowOrthoSize;
                Gizmos.DrawLine(leftBottom, rightBottom);
                Gizmos.DrawLine(rightBottom, rightUp);
                Gizmos.DrawLine(rightUp, leftUp);
                Gizmos.DrawLine(leftUp, leftBottom);

                leftUp = targetPos + lightProjectUp * shadowOrthoSize - lightProjectRight * shadowOrthoSize;
                rightUp = targetPos + lightProjectUp * shadowOrthoSize + lightProjectRight * shadowOrthoSize;
                leftBottom = targetPos - lightProjectUp * shadowOrthoSize - lightProjectRight * shadowOrthoSize;
                rightBottom = targetPos - lightProjectUp * shadowOrthoSize + lightProjectRight * shadowOrthoSize;
                Gizmos.DrawLine(leftBottom, rightBottom);
                Gizmos.DrawLine(rightBottom, rightUp);
                Gizmos.DrawLine(rightUp, leftUp);
                Gizmos.DrawLine(leftUp, leftBottom);
                Handles.ArrowHandleCap(100, translatePos, Quaternion.LookRotation(lightProjectForward), 1, EventType.Repaint);
            }
            if (shadowBound)
            {
                Gizmos.DrawWireCube(sceneData.shadowBound.center, sceneData.shadowBound.size);
            }
            Gizmos.color = color;
        }

        public void UpdateSkyBox()
        {
            if (SkyBoxMat != null)
            {
                RenderSettings.skybox = SkyBoxMat;
                SkyBox = SkyBoxMat.GetTexture(ShaderIDs.Env_SkyCubeTex) as Cubemap;
                if (SkyBox != null)
                {
                    Shader.SetGlobalTexture(ShaderIDs.Env_SkyCube, SkyBox);
                }
            }
        }

        public void UpdateEnv()
        {
            //IBL
            Shader.SetGlobalTexture(ShaderIDs.Env_Cubemap, envCube);
            Shader.SetGlobalVector(ShaderIDs.Env_CubemapParam, new Vector4(hdrScale, hdrPow, hdrAlpha, iblLevel));
            Shader.SetGlobalVector(ShaderIDs.Env_LightmapScale, new Vector4(1.0f / lightmapShadowMask, lightmapShadowMask, shadowIntensity, 0));

            UpdateSkyBox();

            SetLightInfo(ref lighting.roleLightInfo0, ShaderIDs.Env_DirectionalLightDir, ShaderIDs.Env_DirectionalLightColor);
            SetLightInfo(ref lighting.roleLightInfo1, ShaderIDs.Env_DirectionalLightDir1, ShaderIDs.Env_DirectionalLightColor1);

            Shader.SetGlobalColor(ShaderIDs.Env_AmbientParam, new Vector4(1.0f, 0, 0, 0));
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

        private void SetLightInfo(ref LightInfo li, int dirKey, int colorKey)
        {
            Shader.SetGlobalVector(dirKey, li.lightDir);
            Vector4 lightColorIntensity;
            lightColorIntensity = new Vector4(
                Mathf.Pow(li.lightColor.r * li.lightDir.w, 2.2f),
                Mathf.Pow(li.lightColor.g * li.lightDir.w, 2.2f),
                Mathf.Pow(li.lightColor.b * li.lightDir.w, 2.2f), shadowIntensity);
            Shader.SetGlobalVector(colorKey, lightColorIntensity);
        }

        public void SyncLightInfo()
        {
            PrepareTransformGui(roleLight0, ref roleLight0Rot);
            PrepareTransformGui(roleLight1, ref roleLight1Rot);
            SyncLight(roleLight0, ref lighting.roleLightInfo0);
            SyncLight(roleLight1, ref lighting.roleLightInfo1);
        }

        public void SyncLight(Light l, ref LightInfo li)
        {
            if (l != null)
            {
                li.lightDir = l.transform.rotation * -Vector3.forward;
                li.lightColor = l.color;
                li.lightDir.w = (l.enabled && l.gameObject.activeInHierarchy) ? l.intensity : 0;
            }
        }

    }

}