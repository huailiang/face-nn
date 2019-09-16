#if UNITY_EDITOR
using CFUtilPoolLib;
using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace XEngine
{
    public enum DrawType
    {
        Both,
        Draw,
        Cull,
    }

    public delegate void OnDrawGizmoCb();
    [DisallowMultipleComponent, ExecuteInEditMode]
    [RequireComponent(typeof(RenderingEnvironment))]
    public class EnverinmentExtra : MonoBehaviour
    {
        public bool lightingFolder = true;
        public Light roleLight0;
        [System.NonSerialized]
        public TransformRotationGUIWrapper roleLight0Rot;
        public Light roleLight1;
        [System.NonSerialized]
        public TransformRotationGUIWrapper roleLight1Rot;


        public bool shadowFolder = true;
        public bool fitWorldSpace = true;
        public float shadowMapLevel = 0.25f;
        public bool shadowBound = false;
        public GameObject shadowCasterProxy = null;
        public Transform lookTarget;
        public bool drawShadowLighing = false;

        [System.NonSerialized]
        public RenderTexture shadowMap = null;
        private List<Renderer> shadowCasters = new List<Renderer>();
        private CommandBuffer shadowMapCb = null;
        private List<RenderBatch> shadowRenderBatchs = new List<RenderBatch>();
        private Material shadowMat = null;


        public bool debugFolder = true;
        public int quadIndex = -1;
        public DrawType drawType = DrawType.Both;
        public bool showObjects = false;

        public ShaderDebugContext debugContext = new ShaderDebugContext();
        public static int[] debugShaderIDS = new int[]
        {
            Shader.PropertyToID ("_GlobalDebugMode"),
            Shader.PropertyToID ("_DebugDisplayType"),
            Shader.PropertyToID ("_SplitAngle"),
            Shader.PropertyToID ("_SplitPos"),
        };
        public static int[] ppDebugShaderIDS = new int[]
        {
            Shader.PropertyToID ("_PPDebugMode"),
            Shader.PropertyToID ("_PPDebugDisplayType"),
            Shader.PropertyToID ("_PPSplitAngle"),
            Shader.PropertyToID ("_PPSplitPos"),
        };
        
        [NonSerialized]
        private CommandBuffer[] commandBuffer = null;
        private CommandBuffer editorCommandBuffer;


        [NonSerialized]
        public RenderingEnvironment re;
        Camera mainCamera;
        Camera sceneViewCamera;
        private static OnDrawGizmoCb m_drawGizmo = null;


        void Awake()
        {
            GetSceneViewCamera();
        }

        void Start()
        {
            Shader.SetGlobalFloat("_GlobalDebugMode", 0);
        }

        void OnDestroy()
        {
            if (commandBuffer != null)
            {
                if (SceneView.lastActiveSceneView != null && SceneView.lastActiveSceneView.camera != null)
                {
                    SceneView.lastActiveSceneView.camera.RemoveAllCommandBuffers();
                }

                foreach (CommandBuffer cb in commandBuffer)
                {
                    cb.Release();
                }
                commandBuffer = null;
            }
            RefreshLightmap(false);
            if (editorCommandBuffer != null)
            {
                editorCommandBuffer.Release();
                editorCommandBuffer = null;
            }
            if (shadowMapCb != null)
            {
                shadowMapCb.Release();
                shadowMapCb = null;
            }
            if (shadowMap != null)
            {
                RuntimeUtilities.Destroy(shadowMap);
                shadowMap = null;
            }
        }

        void Update()
        {
            if (re == null)
                re = this.GetComponent<RenderingEnvironment>();
            SyncLightInfo();

            UpdateShadowCaster();
            BuildShadowMap();

            if (re != null)
            {
                re.fitWorldSpace = fitWorldSpace;
            }
            debugContext.Refresh();
        }

        public void Init()
        {
            if (re == null)
                re = this.GetComponent<RenderingEnvironment>();
            SceneData.updateSceneView = SceneView_Update;
            SceneData.editorSetRes = SetRes;
            if (shadowMapCb == null)
                shadowMapCb = new CommandBuffer { name = "Editor Shadow Map Cb" };


            shadowMat = AssetsConfig.GlobalAssetsConfig.ShadowCaster;
            UpdateShadowCaster();
        }

        private void GetSceneViewCamera()
        {
            if (sceneViewCamera == null)
            {
                commandBuffer = new CommandBuffer[3];
                CommandBuffer commandBufferBeforeOpaque = new CommandBuffer { name = "SceneView Before Opaque" };
                commandBuffer[(int)ECommandBufferType.BeforeOpaque] = commandBufferBeforeOpaque;
                CommandBuffer commandBufferAfterOpaque = new CommandBuffer { name = "SceneView After Opaque" };
                commandBuffer[(int)ECommandBufferType.AfterOpaque] = commandBufferAfterOpaque;
                CommandBuffer commandBufferAfterTransparent = new CommandBuffer { name = "SceneView After Transparent" };
                commandBuffer[(int)ECommandBufferType.AfterForwardAlpha] = commandBufferAfterTransparent;
                if (SceneView.lastActiveSceneView != null && SceneView.lastActiveSceneView.camera != null)
                {
                    sceneViewCamera = SceneView.lastActiveSceneView.camera;
                    sceneViewCamera.RemoveAllCommandBuffers();
                    sceneViewCamera.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, commandBufferBeforeOpaque);
                    sceneViewCamera.AddCommandBuffer(CameraEvent.AfterForwardOpaque, commandBufferAfterOpaque);
                    sceneViewCamera.AddCommandBuffer(CameraEvent.AfterForwardAlpha, commandBufferAfterTransparent);
                }
            }
        }

        #region lighting
        private void PrepareTransformGui(Light light, ref TransformRotationGUIWrapper wrapper)
        {
            if (light != null && (wrapper == null || wrapper.t != light.transform))
            {
                wrapper = EditorCommon.GetTransformRotatGUI(light.transform);
            }
        }

        private void SyncLightInfo(Light light, ref LightInfo li, Light inversLight, ref TransformRotationGUIWrapper wrapper)
        {
            if (light != null)
            {
                re.SyncLight(light, ref li);
                if (inversLight != null)
                {
                    li.lightDir = inversLight.transform.rotation * Vector3.forward;
                }
            }
        }

        public void SyncLightInfo()
        {
            if (re != null)
            {
                PrepareTransformGui(roleLight0, ref roleLight0Rot);
                PrepareTransformGui(roleLight1, ref roleLight1Rot);
                SyncLightInfo(roleLight0, ref re.lighting.roleLightInfo0, null, ref roleLight0Rot);
                SyncLightInfo(roleLight1, ref re.lighting.roleLightInfo1, null, ref roleLight1Rot);
            }
        }

        public void RefreshLightmap(bool preview)
        {
            RenderingEnvironment.isPreview = preview;

            if (preview)
            {
                Shader.EnableKeyword("LIGHTMAP_ON");
                Shader.EnableKeyword("_CUSTOM_LIGHTMAP_ON");
                Shader.SetGlobalVector(ShaderManager._ShaderKeyLightMapEnable, Vector4.one);
            }
            else
            {
                Shader.DisableKeyword("LIGHTMAP_ON");
                Shader.DisableKeyword("_CUSTOM_LIGHTMAP_ON");
                Shader.SetGlobalVector(ShaderManager._ShaderKeyLightMapEnable, Vector4.zero);
            }
        }
        #endregion

        private void UpdateShadowCaster()
        {
            if (lookTarget == null)
            {
                GameObject go = GameObject.Find("LookTarget");
                lookTarget = go != null ? go.transform : null;
            }
            if (lookTarget != null)
            {
                re.sceneData.currentEntityPos = lookTarget.position;
            }
            else
            {
                if (re.sceneData.CameraRef != null)
                    re.sceneData.currentEntityPos = re.cameraForward * 10 + re.sceneData.cameraPos;
            }
            shadowCasters.Clear();
            if (shadowCasterProxy == null)
            {
                shadowCasterProxy = GameObject.Find("ShadowCaster");
            }
            shadowRenderBatchs.Clear();
            bool first = true;
            Bounds shadowBound = new Bounds(Vector3.zero, Vector3.zero);
            if (shadowCasterProxy != null)
            {
                shadowCasterProxy.GetComponentsInChildren<Renderer>(false, shadowCasters);
                if (shadowCasters.Count > 0)
                {
                    for (int i = 0; i < shadowCasters.Count; ++i)
                    {
                        Renderer render = shadowCasters[i];
                        if (render != null &&
                            render.enabled &&
                            render.shadowCastingMode != ShadowCastingMode.Off &&
                            render.sharedMaterial != null)
                        {
                            RenderBatch rb = new RenderBatch();
                            rb.render = render;
                            rb.mat = render.sharedMaterial;
                            rb.mpbRef = null;
                            rb.passID = 0;
                            shadowRenderBatchs.Add(rb);
                            if (first)
                            {
                                shadowBound = render.bounds;
                                first = false;
                            }
                            else
                                shadowBound.Encapsulate(render.bounds);
                        }
                    }
                }
            }
            re.sceneData.shadowBound = shadowBound;
        }

        private void BuildShadowMap()
        {
            if (shadowMap != null && shadowMat != null)
            {
                shadowMapCb.Clear();
                shadowMapCb.ClearRenderTarget(true, true, Color.clear, 1.0f);
                shadowMapCb.SetViewProjectionMatrices(re.sceneData.shadowViewMatrix, re.sceneData.shadowProjMatrix);

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
        

        void SetRes(System.Object obj, int type)
        {
            if (type == 0)
            {
                shadowMap = obj as RenderTexture;
            }
        }


        void OnDrawGizmos()
        {
            Color color = Gizmos.color;
            if (mainCamera == null)
                mainCamera = GetComponent<Camera>();
            if (drawShadowLighing)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireSphere(re.translatePos, 0.3f);
                Vector3 targetPos = re.translatePos + re.lightProjectForward * 10;
                Gizmos.DrawWireSphere(targetPos, 0.3f);
                Vector3 leftUp = re.translatePos + re.lightProjectUp * re.shadowOrthoSize - re.lightProjectRight * re.shadowOrthoSize;
                Vector3 rightUp = re.translatePos + re.lightProjectUp * re.shadowOrthoSize + re.lightProjectRight * re.shadowOrthoSize;
                Vector3 leftBottom = re.translatePos - re.lightProjectUp * re.shadowOrthoSize - re.lightProjectRight * re.shadowOrthoSize;
                Vector3 rightBottom = re.translatePos - re.lightProjectUp * re.shadowOrthoSize + re.lightProjectRight * re.shadowOrthoSize;
                Gizmos.DrawLine(leftBottom, rightBottom);
                Gizmos.DrawLine(rightBottom, rightUp);
                Gizmos.DrawLine(rightUp, leftUp);
                Gizmos.DrawLine(leftUp, leftBottom);

                leftUp = targetPos + re.lightProjectUp * re.shadowOrthoSize - re.lightProjectRight * re.shadowOrthoSize;
                rightUp = targetPos + re.lightProjectUp * re.shadowOrthoSize + re.lightProjectRight * re.shadowOrthoSize;
                leftBottom = targetPos - re.lightProjectUp * re.shadowOrthoSize - re.lightProjectRight * re.shadowOrthoSize;
                rightBottom = targetPos - re.lightProjectUp * re.shadowOrthoSize + re.lightProjectRight * re.shadowOrthoSize;
                Gizmos.DrawLine(leftBottom, rightBottom);
                Gizmos.DrawLine(rightBottom, rightUp);
                Gizmos.DrawLine(rightUp, leftUp);
                Gizmos.DrawLine(leftUp, leftBottom);
                Handles.ArrowHandleCap(100, re.translatePos, Quaternion.LookRotation(re.lightProjectForward), 1, EventType.Repaint);
            }
            if (shadowBound)
            {
                Gizmos.DrawWireCube(re.sceneData.shadowBound.center, re.sceneData.shadowBound.size);
            }

            if (m_drawGizmo != null)
            {
                m_drawGizmo();
            }
            Gizmos.color = color;
        }

        public void SceneView_Update()
        {
            GetSceneViewCamera();
            if (commandBuffer != null)
            {
                for (int i = 0; i < commandBuffer.Length; ++i)
                {
                    CommandBuffer cb = commandBuffer[i];
                    cb.Clear();
                }
            }
        }
        
    }

}

#endif
