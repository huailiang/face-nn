#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
namespace CFEngine
{
    public enum LightingEditMode
    {
        Enverinment,
        UIScene,
    }
    public enum QuadTreeLevel
    {
        None,
        Level0,
        Level1,
        Level2,
        Level3
    }
    public enum DrawType
    {
        Both,
        Draw,
        Cull,
    }
    public enum LightMask
    {
        Role = 0x0001,
        Scene = 0x0002,
        Sun = 0x0004,
    }
    public enum LightingMode
    {
        SimpleLoop,
        Voxel
    }
    enum OpType
    {
        OpNone,
        OpCollectLights
    }

    [Serializable]
    public class LightBlockInfo
    {
        public int chunkID = -1;
        public int blockId = -1;
        public int objCount = 0;
        public int xzIndex;
        public int yIndex;
        public Vector3 offset;
        public List<Light> lights = new List<Light> ();
    }
    struct LightDataInfo
    {
        public Vector4 lightPos; //xyz:world pos w:range
        public Vector4 lightColor; //xyz:color w: not use
    }
    struct LightHeadIndex
    {
        public uint blockStartIndex;
        public float minY;
    }

    [System.Serializable]
    public class ChunkLightInfo
    {
        public Light light;
        public int index;
    }

    [System.Serializable]
    public class ChunkLightVerticalBlock
    {
        public int yIndex;
        public Vector3 offset;
        public List<ChunkLightInfo> lightInfos = new List<ChunkLightInfo> ();
    }

    [System.Serializable]
    public class ChunkLightIndex
    {
        public int xzIndex;
        public float minY = 2000;
        public int minYIndex = 1000;
        public int maxYIndex = 0;
        public List<ChunkLightVerticalBlock> lightVerticalBlock = new List<ChunkLightVerticalBlock> ();
    }

    [System.Serializable]
    public class ChunkLightBlockInfo
    {
        public int chunkID = -1;
        public List<ChunkLightIndex> lightIndexs = new List<ChunkLightIndex> ();

        public List<Light> lights = new List<Light> ();
    }

    public class MeshVertex
    {
        public Vector3[] vertices;
        public int[] triangles;
    }

    [Serializable]
    public class LightLoopContext
    {
        public int chunkWidth = 100;
        public int chunkHeight = 100;
        public int widthCount = 100;
        public int heightCount = 100;
        public int xCount = 200;
        public int zCount = 200;
        public int lightGridSize = 5;
        public Dictionary<int, LightBlockInfo> lightBlocks = new Dictionary<int, LightBlockInfo> ();

        public Dictionary<Mesh, MeshVertex> processMesh = new Dictionary<Mesh, MeshVertex> ();

        public static int FindChunkIndex (Vector3 pos, float width, float height, int xxCount, int zzCount, out int x, out int z)
        {
            x = (int) Mathf.Clamp (pos.x / width, 0, xxCount - 1);
            z = (int) Mathf.Clamp (pos.z / width, 0, zzCount - 1);
            return x + z * xxCount;
        }
    }

    public delegate void OnDrawGizmoCb ();
    [DisallowMultipleComponent, ExecuteInEditMode]
    [RequireComponent (typeof (RenderingEnvironment))]
    public class EnverinmentExtra : MonoBehaviour
    {
#region lighting
        public bool lightingFolder = true;
        public Light roleLight0;
        [System.NonSerialized]
        public TransformRotationGUIWrapper roleLight0Rot;

        public Light roleLight1;
        [System.NonSerialized]
        public TransformRotationGUIWrapper roleLight1Rot;

        public static LightingEditMode editMode = LightingEditMode.Enverinment;

        public Light bakeSceneLight0;

        [System.NonSerialized]
        public TransformRotationGUIWrapper bakeSceneLight0Rot;

        public Light bakeSceneLight1;

        [System.NonSerialized]
        public TransformRotationGUIWrapper bakeSceneLight1Rot;

        public Light sceneRuntimeLight0;

        [System.NonSerialized]
        public TransformRotationGUIWrapper sceneRuntimeLight0Rot;

        public Light sceneRuntimeLight1;

        [System.NonSerialized]
        public TransformRotationGUIWrapper sceneRuntimeLight1Rot;

        private Light sceneLight0;
        private TransformRotationGUIWrapper sceneLight0Rot;
        private Light sceneLight1;
        private TransformRotationGUIWrapper sceneLight1Rot;

        public Light sunLight;
        [System.NonSerialized]
        public TransformRotationGUIWrapper sunLightRot;
        private GameObject dummyLightGo;

#endregion

#region toggle
        public bool fastEditLight = false;
        public bool fastEditEnvLight = false;
        public bool fastEditWind = false;
        public bool useUnityLighting = false;
#endregion

#region testObj
        public bool testObjFolder = true;
        public Light pointLight;
        private float cycle = 0.0f;
        private float sign = 1.0f;
        public Transform roleDummy;
        public float interactiveParam;
        public bool debugEnvArea = false;
        public static int envObjStart;
        public static int envObjEnd;
        public List<EnverinmentArea> envObjects = new List<EnverinmentArea> ();
        [NonSerialized]
        public EnverinmentArea lastArea = null;
        [NonSerialized]
        public EnverinmentArea currentArea = null;
        [NonSerialized]
        public float envLerpTime = -1;
        [NonSerialized]
        public bool intoEnvArea = true;
#endregion

#region fastRun
        public bool fastRunFolder = true;
        public bool loadGameAtHere = false;
        public bool useCurrentScene = false;
        public bool gotoScene = false;
        public int sceneID = -1;
        public bool replaceStartScene = false;
        public bool useStaticBatch = false;
#endregion

#region freeCamera
        public bool freeCameraFolder = true;
        static Texture2D ms_invisibleCursor;

        public bool forceUpdateFreeCamera = false;
        public bool forceIgnore = false;
        public bool holdRightMouseCapture = false;

        public float lookSpeed = 5f;
        public float moveSpeed = 5f;
        public float sprintSpeed = 50f;

        bool m_inputCaptured;
        float m_yaw;
        float m_pitch;
#endregion

#region debugShadow

        public bool shadowFolder = true;
        public bool fitWorldSpace = true;
        public float shadowMapLevel = 0.25f;
        public bool shadowBound = false;
        public GameObject shadowCasterProxy = null;
        public Transform lookTarget;
        public bool drawShadowLighing = false;

        [System.NonSerialized]
        public RenderTexture shadowMap = null;

        private List<Renderer> shadowCasters = new List<Renderer> ();
        private CommandBuffer shadowMapCb = null;

        private List<RenderBatch> shadowRenderBatchs = new List<RenderBatch> ();

        private Material shadowMat = null;

#endregion

#region debug
        public bool debugFolder = true;
        public bool drawFrustum = false;
        public bool drawInvisibleObj = false;
        public bool drawLodGrid = false;
        public bool drawTerrainGrid = false;
        public QuadTreeLevel quadLevel = QuadTreeLevel.None;
        public int quadIndex = -1;
        public DrawType drawType = DrawType.Both;
        public bool showObjects = false;

        public bool drawPointLight = false;
        public bool drawWind = false;
        public bool drawTerrainHeight = false;

        public bool drawLightBox = false;

        public ShaderDebugContext debugContext = new ShaderDebugContext();
        public bool isDebugLayer = false;
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
        public bool updateSceneObject = false;
#endregion

#region runtime objects

        [NonSerialized]
        public Dictionary<int, SceneChunk> chunks = new Dictionary<int, SceneChunk> ();
        [NonSerialized]
        public QuadTree quadTreeRef;
        [NonSerialized]
        public List<SceneObject> sceneObjects = new List<SceneObject> ();

        private CommandBuffer[] commandBuffer = null;

        private CommandBuffer editorCommandBuffer;
        private List<RenderBatch> renderBatches = new List<RenderBatch> ();

        private Material[] sceneMats;

        private Dictionary<uint, SceneDynamicObject> dynamicObjects;

#endregion

#region voxelLights

        public bool voxelLightFolder = true;
        public LightingMode lightMode = LightingMode.SimpleLoop;
        public List<ChunkLightBlockInfo> chunkLightBlocks = new List<ChunkLightBlockInfo> ();
        public List<Light> lights = new List<Light> ();
        public int minLightCount = 0;
        public int maxLightCount = 0;
        public LightLoopContext lightLoopContext = new LightLoopContext ();

        private static EditorCommon.EnumTransform funFindLight = FindLight;
        private static EditorCommon.EnumTransform funCollectLight = CollectLights;
        private static EditorCommon.EnumTransform funIntersectLightObjects = IntersectLightObjects;
        private ComputeBuffer lightInfoBuffer;
        private ComputeBuffer lightIndexBuffer;
        private ComputeBuffer verticalBlockIndexBuffer;
        private ComputeBuffer lightIndexHeadBuffer;
        public int previewLightCount = 0;
        private Color[] lightInfoColor = new Color[5]
        {
            new Color (1, 1, 1, 0.8f),
            new Color (0, 0, 1, 0.8f),
            new Color (0, 1, 1, 0.8f),
            new Color (1, 1, 0, 0.8f),
            new Color (1, 0, 0, 0.8f),
        };
        private bool isInit = false;
#endregion

#region lightmap
#endregion

#region misc
        [NonSerialized]
        public RenderingEnvironment re;
        Camera mainCamera;
        Camera sceneViewCamera;
        private List<TerrainObject> terrainObjects = new List<TerrainObject> ();
        private OpType opType = OpType.OpNone;
        public static bool drawSceneBounds = false;
        private static OnDrawGizmoCb m_drawGizmo = null;
#endregion

        void Awake ()
        {
            if (!ms_invisibleCursor)
            {
                ms_invisibleCursor = new Texture2D (1, 1);
                ms_invisibleCursor.SetPixel (0, 0, new Color32 (0, 0, 0, 0));
            }

            GetSceneViewCamera ();
            forceUpdateFreeCamera |= XDebug.debugRoamingScene;
        }

        void Start ()
        {
            Shader.SetGlobalFloat("_GlobalDebugMode", 0);
            if (loadGameAtHere && Application.isPlaying)
            {
                if (null == XInterfaceMgr.singleton.GetInterface<IEntrance> (0))
                {
                    UnityEngine.SceneManagement.SceneManager.LoadScene (0);
                    if (useCurrentScene)
                    {
                        XDebug.editorSceneReplace = UnityEngine.SceneManagement.SceneManager.GetActiveScene ().name;
                    }
                    else
                    {
                        XDebug.editorSceneReplace = "";
                    }
                    if (gotoScene)
                        XDebug.sceneID = sceneID;
                    else
                        XDebug.sceneID = -1;
                    if (replaceStartScene)
                        XDebug.startSceneID = sceneID;

                    XDebug.startPos = transform.position;
                    XDebug.startRot = transform.rotation.eulerAngles;

                    XDebug.singleton.SetFlag (EDebugFlag.EDebug_StaticBatch, useStaticBatch);
                }
            }
        }

        void OnDestroy ()
        {
            if (commandBuffer != null)
            {
                if (SceneView.lastActiveSceneView != null && SceneView.lastActiveSceneView.camera != null)
                {
                    SceneView.lastActiveSceneView.camera.RemoveAllCommandBuffers ();
                }

                foreach (CommandBuffer cb in commandBuffer)
                {
                    cb.Release ();
                }
                commandBuffer = null;
            }
            RefreshLightmap (false);
            if (editorCommandBuffer != null)
            {
                editorCommandBuffer.Release ();
                editorCommandBuffer = null;
            }
            if (shadowMapCb != null)
            {
                shadowMapCb.Release ();
                shadowMapCb = null;
            }
            if (shadowMap != null)
            {
                RuntimeUtilities.Destroy (shadowMap);
                shadowMap = null;
            }
        }
        void Update ()
        {
            if (re == null)
                re = this.GetComponent<RenderingEnvironment> ();
            SyncLightInfo ();

        
                if (roleDummy != null)
                {
                    Vector3 pos = roleDummy.position;
                    re.sceneData.currentEntityPos = pos;
                    if (debugEnvArea && envObjects != null)
                    {
                        EnverinmentArea.TestArea (this, new Vector2 (pos.x, pos.z));
                    }
                    UpdateArea ();
                }

                if (!isInit)
                {
                    InitScene ();
                    isInit = true;
                }

                switch (opType)
                {
                    case OpType.OpCollectLights:
                        InnerCollectLights ();
                        break;
                }
                opType = OpType.OpNone;
            

            if ((Application.isPlaying ||
                    forceUpdateFreeCamera) && !forceIgnore)
                UpdateFreeCamera ();

         
                UpdateShadowCaster ();
                BuildShadowMap ();
                if (pointLight != null && pointLight.type == LightType.Point)
                {
                    Vector3 pos = pointLight.transform.position;
                    float range = pointLight.range * pointLight.range;
                    float intensity = pointLight.intensity;
                    cycle += Time.deltaTime * sign;
                    if (Mathf.Abs (cycle) > UnityEngine.Random.Range (5, 8))
                    {
                        sign = -sign;
                    }
                    intensity += Mathf.Sin (Time.time) * Mathf.PerlinNoise (cycle, Mathf.Cos (Time.time)) * intensity * 0.8f;
                    // intensity += Mathf.Sin(cycle * Mathf.Cos(Time.time)) * intensity * 0.8f;
                    Vector4 color = new Vector4 (Mathf.Pow (pointLight.color.r * intensity, 2.2f),
                        Mathf.Pow (pointLight.color.g * intensity, 2.2f),
                        Mathf.Pow (pointLight.color.b * intensity, 2.2f), range != 0 ? 1 / range : 1);
                    Shader.SetGlobalColor (ShaderManager._ShaderKeyPointLightDir, new Vector4 (pos.x, pos.y, pos.z, range));

                    Shader.SetGlobalColor (ShaderManager._ShaderKeyPointLightColor, color);
                }
            
            if (re != null)
            {
                re.fitWorldSpace = fitWorldSpace;
            }
            debugContext.Refresh();
        }
        public void Init ()
        {
            if (re == null)
                re = this.GetComponent<RenderingEnvironment> ();
            SceneData.updateSceneView = SceneView_Update;
            SceneData.editorEditChunk = EditChunk;
            SceneData.editorSetRes = SetRes;

                re.sceneData.enableShadow = true;
                if (shadowMapCb == null)
                    shadowMapCb = new CommandBuffer { name = "Editor Shadow Map Cb" };
                if (re.sceneData.ShadowRT == null)
                {
                    if (shadowMap == null)
                    {
                    shadowMap = new RenderTexture (512, 512, 16, RenderTextureFormat.RG16, RenderTextureReadWrite.Linear)
                    {
                    name = "Shadowmap",
                    hideFlags = HideFlags.DontSave,
                    filterMode = FilterMode.Bilinear,
                    wrapMode = TextureWrapMode.Clamp,
                    anisoLevel = 0,
                    autoGenerateMips = false,
                    useMipMap = false
                        };
                        shadowMap.Create ();
                    }
                }

                shadowMat = AssetsConfig.GlobalAssetsConfig.ShadowCaster;
                UpdateShadowCaster ();
            
        }
        private void GetSceneViewCamera ()
        {
            if (sceneViewCamera == null)
            {
                commandBuffer = new CommandBuffer[3];
                CommandBuffer commandBufferBeforeOpaque = new CommandBuffer { name = "SceneView Before Opaque" };
                commandBuffer[(int) ECommandBufferType.BeforeOpaque] = commandBufferBeforeOpaque;
                CommandBuffer commandBufferAfterOpaque = new CommandBuffer { name = "SceneView After Opaque" };
                commandBuffer[(int) ECommandBufferType.AfterOpaque] = commandBufferAfterOpaque;
                CommandBuffer commandBufferAfterTransparent = new CommandBuffer { name = "SceneView After Transparent" };
                commandBuffer[(int) ECommandBufferType.AfterForwardAlpha] = commandBufferAfterTransparent;
                if (SceneView.lastActiveSceneView != null && SceneView.lastActiveSceneView.camera != null)
                {
                    sceneViewCamera = SceneView.lastActiveSceneView.camera;
                    sceneViewCamera.RemoveAllCommandBuffers ();
                    sceneViewCamera.AddCommandBuffer (CameraEvent.BeforeForwardOpaque, commandBufferBeforeOpaque);
                    sceneViewCamera.AddCommandBuffer (CameraEvent.AfterForwardOpaque, commandBufferAfterOpaque);
                    sceneViewCamera.AddCommandBuffer (CameraEvent.AfterForwardAlpha, commandBufferAfterTransparent);
                }
            }
        }
#region debug
        public void RefreshDebug(bool isPP)
        {
            Shader.SetGlobalFloat(ShaderIDs.DebugLayer, 0);
            debugContext.shaderID = EnverinmentExtra.debugShaderIDS;
            debugContext.Reset();
            debugContext.Refresh();
            debugContext.shaderID = EnverinmentExtra.ppDebugShaderIDS;
            debugContext.Reset();
            debugContext.Refresh();
            debugContext.shaderID = isPP ? EnverinmentExtra.ppDebugShaderIDS : EnverinmentExtra.debugShaderIDS;
        }
#endregion
#region lighting
        private void PrepareTransformGui (Light light, ref TransformRotationGUIWrapper wrapper)
        {
            if (light != null && (wrapper == null || wrapper.t != light.transform))
            {
                wrapper = EditorCommon.GetTransformRotatGUI (light.transform);
            }
        }

        private void SyncLightInfo (Light light, ref LightInfo li, Light inversLight, ref TransformRotationGUIWrapper wrapper)
        {
            if (light != null)
            {
                re.SyncLight (light, ref li);
                if (inversLight != null)
                {
                    li.lightDir = inversLight.transform.rotation * Vector3.forward;
                }
            }
        }

        public void SyncLightInfo (Light light, ref TransformRotationGUIWrapper rot, ref Vector3 data, bool invers = false)
        {
            if (light != null)
            {
                if (invers)
                    data = light.transform.rotation * Vector3.forward;
                else
                    data = light.transform.rotation * -Vector3.forward;

                if (rot == null || rot.t != light.transform)
                {
                    rot = EditorCommon.GetTransformRotatGUI (light.transform);
                }
            }
        }
        public void SyncLightInfo ()
        {
                useUnityLighting = true;

            if (useUnityLighting)
            {
                bool isRuntime =  RenderingEnvironment.isPreview;
                if (isRuntime)
                {
                    sceneLight0 = sceneRuntimeLight0;
                    sceneLight0Rot = sceneRuntimeLight0Rot;
                    sceneLight1 = sceneRuntimeLight1;
                    sceneLight1Rot = sceneRuntimeLight0Rot;
                }
                else
                {
                    sceneLight0 = bakeSceneLight0;
                    sceneLight0Rot = bakeSceneLight0Rot;
                    sceneLight1 = bakeSceneLight1;
                    sceneLight1Rot = bakeSceneLight1Rot;
                }

                if (re != null)
                {
                    PrepareTransformGui (roleLight0, ref roleLight0Rot);
                    PrepareTransformGui (roleLight1, ref roleLight1Rot);
                    PrepareTransformGui (bakeSceneLight0, ref bakeSceneLight0Rot);
                    PrepareTransformGui (bakeSceneLight1, ref bakeSceneLight1Rot);
                    PrepareTransformGui (sceneRuntimeLight0, ref sceneRuntimeLight0Rot);
                    PrepareTransformGui (sceneRuntimeLight1, ref sceneRuntimeLight1Rot);
                    SyncLightInfo (roleLight0, ref re.lighting.roleLightInfo0, null, ref roleLight0Rot);
                    SyncLightInfo (roleLight1, ref re.lighting.roleLightInfo1, null, ref roleLight1Rot);
                    SyncLightInfo (sceneLight0, ref re.lighting.sceneLightInfo0, null, ref sceneLight0Rot);
                    SyncLightInfo (sceneLight1, ref re.lighting.sceneLightInfo1, null, ref sceneLight1Rot);
                    if (isRuntime)
                    {
                        sceneRuntimeLight0Rot = sceneLight0Rot;
                        sceneRuntimeLight1Rot = sceneLight1Rot;
                    }
                    else
                    {
                        bakeSceneLight0Rot = sceneLight0Rot;
                        bakeSceneLight1Rot = sceneLight1Rot;
                    }
                    SyncLightInfo (sunLight, ref sunLightRot, ref re.sunDir);
                }
            }
        }

        private Light CreateLight (string name, ref LightInfo li, int mask)
        {
            GameObject go = new GameObject (name);
            Light l = go.AddComponent<Light> ();
            l.transform.rotation = Quaternion.LookRotation (-li.lightDir);
            l.type = LightType.Directional;
            l.color = li.lightColor;
            l.intensity = li.lightDir.w;
            if (dummyLightGo == null)
            {
                dummyLightGo = new GameObject ("DummyLights");
            }
            l.cullingMask = mask;
            go.transform.parent = dummyLightGo.transform;
            return l;
        }

        public void PrepareLights (ref Light light, ref LightInfo lightInfo, string name)
        {
            if (light == null)
            {
                light = CreateLight (name, ref lightInfo, 1 << GameObjectLayerHelper.InVisiblityLayer);

            }
        }
        public void PrepareLights (ref Light light, ref Vector3 dir, string name)
        {
            if (light == null)
            {
                GameObject go = new GameObject (name);
                light = go.AddComponent<Light> ();
                light.transform.rotation = Quaternion.LookRotation (-dir);
                light.type = LightType.Directional;
                if (dummyLightGo == null)
                {
                    dummyLightGo = new GameObject ("DummyLights");
                }
                light.cullingMask = 1 << GameObjectLayerHelper.InVisiblityLayer;
                go.transform.parent = dummyLightGo.transform;
            }
        }
        public void PrepareLights (LightMask lightMask)
        {
            if (re != null)
            {
                if ((lightMask & LightMask.Role) != 0)
                {
                        PrepareLights (ref roleLight0, ref re.lighting.roleLightInfo0, "roleLight0");

                        PrepareLights (ref roleLight1, ref re.lighting.roleLightInfo1, "roleLight1");
                }
                if ((lightMask & LightMask.Scene) != 0)
                {
                    PrepareLights (ref sceneRuntimeLight0, ref re.lighting.sceneLightInfo0, "runtimeSceneLight0");
                    PrepareLights (ref sceneRuntimeLight1, ref re.lighting.sceneLightInfo1, "runtimeSceneLight1");
                }
                if ((lightMask & LightMask.Sun) != 0)
                {
                    PrepareLights (ref sunLight, ref re.sunDir, "sunLight");
                }
            }
        }

        public void CleanLights ()
        {
            if (roleLight0 != null)
            {
                GameObject.DestroyImmediate (roleLight0.gameObject);
                roleLight0Rot = null;
            }
            if (roleLight1 != null)
            {
                GameObject.DestroyImmediate (roleLight1.gameObject);
                roleLight1Rot = null;
            }
            if (sceneRuntimeLight0 != null)
            {
                GameObject.DestroyImmediate (sceneRuntimeLight0.gameObject);
                sceneRuntimeLight0Rot = null;
            }
            if (sceneRuntimeLight1 != null)
            {
                GameObject.DestroyImmediate (sceneRuntimeLight1.gameObject);
                sceneRuntimeLight1Rot = null;
            }
            if (sunLight != null)
            {
                GameObject.DestroyImmediate (sunLight.gameObject);
                sunLightRot = null;
            }
        }
        private void SyncLightInfo (Light light, ref LightInfo li)
        {
            if (light != null)
            {
                light.transform.rotation = Quaternion.LookRotation (-li.lightDir);
                light.intensity = li.lightDir.w;
                light.color = li.lightColor;
            }
        }

        public void RefreshLightmap (bool preview)
        {
            RenderingEnvironment.isPreview = preview;
            EnableShadow (!preview);

            if (preview)
            {
                Shader.EnableKeyword ("LIGHTMAP_ON");
                Shader.EnableKeyword ("_CUSTOM_LIGHTMAP_ON");
                Shader.SetGlobalVector (ShaderManager._ShaderKeyLightMapEnable, Vector4.one);
            }
            else
            {
                Shader.DisableKeyword ("LIGHTMAP_ON");
                Shader.DisableKeyword ("_CUSTOM_LIGHTMAP_ON");
                Shader.SetGlobalVector (ShaderManager._ShaderKeyLightMapEnable, Vector4.zero);
            }
        }

        public void InitScene ()
        {
            terrainObjects.Clear ();

            InitLighting ();
        }

        private void InitLighting ()
        {
            if (lightMode == LightingMode.SimpleLoop)
            {
                //InitSimpleLoopLighting ();
            }
            else
            {
                RefreshVoxelLighting ();
            }
        }

        private static void FindLight (Transform trans, object param)
        {
            if (trans.gameObject.activeInHierarchy)
            {
                Light light = trans.GetComponent<Light> ();
                if (light != null)
                {
                    if (light.enabled)
                    {
                        if (light.type == LightType.Point)
                        {
                            EnverinmentExtra ee = param as EnverinmentExtra;
                            ee.lights.Add (light);

                        }
                    }
                }
                EditorCommon.EnumChildObject (trans, param, funFindLight);
            }
        }

        public void InitSimpleLoopLighting ()
        {
            lights.Clear ();
            string path = AssetsConfig.EditorGoPath[0] + "/" + AssetsConfig.EditorGoPath[(int) AssetsConfig.EditorSceneObjectType.Light];
            EditorCommon.EnumTargetObject (path, funFindLight, this);
        }

        public void RefreshVoxelLighting ()
        {
            int lineBlockCount = lightLoopContext.chunkWidth / lightLoopContext.lightGridSize * lightLoopContext.widthCount;
            Shader.SetGlobalInt ("_LineBlockCount", lineBlockCount);
            Shader.SetGlobalVector ("_WorldChunkOffset", new Vector4 (0, 0, 1.0f / lightLoopContext.lightGridSize, 0));
            int chunkCount = lightLoopContext.widthCount * lightLoopContext.heightCount;
            if (lights.Count > 0 && chunkCount > 0)
            {
                LightDataInfo[] lightDataInfos = new LightDataInfo[lights.Count];
                for (int i = 0; i < lights.Count; ++i)
                {
                    var light = lights[i];
                    Vector3 pos = light.transform.position;
                    lightDataInfos[i] = new LightDataInfo ()
                    {
                        lightPos = new Vector4 (
                        pos.x, pos.y, pos.z,
                        1 / light.range),
                        lightColor = new Vector4 (
                        Mathf.Pow (light.color.r * light.intensity, 2.2f),
                        Mathf.Pow (light.color.g * light.intensity, 2.2f),
                        Mathf.Pow (light.color.b * light.intensity, 2.2f),
                        0),
                    };
                }

                LightHeadIndex[] lightIndexHead = new LightHeadIndex[lineBlockCount * lineBlockCount * chunkCount];
                for (int i = 0; i < lightIndexHead.Length; ++i)
                {
                    lightIndexHead[i] = new LightHeadIndex ()
                    {
                        minY = 2000f
                    };

                }
                List<uint> verticalBlockIndex = new List<uint> ();
                List<uint> lightIndex = new List<uint> ();
                for (int i = 0; i < chunkLightBlocks.Count; ++i)
                {
                    var clb = chunkLightBlocks[i];
                    clb.lightIndexs.Sort ((a, b) => { return a.xzIndex - b.xzIndex; });

                    for (int j = 0; j < clb.lightIndexs.Count; ++j)
                    {
                        var cli = clb.lightIndexs[j];
                        lightIndexHead[cli.xzIndex] = new LightHeadIndex ()
                        {
                            blockStartIndex = (uint) verticalBlockIndex.Count,
                            minY = cli.minY
                        };
                        if (cli.minYIndex <= cli.maxYIndex)
                        {
                            int count = cli.maxYIndex - cli.minYIndex + 1;
                            int startIndex = verticalBlockIndex.Count;
                            for (int k = 0; k < count; ++k)
                            {
                                verticalBlockIndex.Add (2000000);
                            }

                            cli.lightVerticalBlock.Sort ((a, b) => { return a.yIndex - b.yIndex; });

                            for (int k = 0; k < cli.lightVerticalBlock.Count; ++k)
                            {
                                var clvb = cli.lightVerticalBlock[k];
                                int offset = clvb.yIndex - cli.minYIndex;
                                if (offset >= count)
                                    Debug.LogError ("Y outof range");
                                verticalBlockIndex[startIndex + offset] = (uint) lightIndex.Count;
                                lightIndex.Add ((uint) clvb.lightInfos.Count);
                                for (int ii = 0; ii < clvb.lightInfos.Count; ++ii)
                                {
                                    lightIndex.Add ((uint) clvb.lightInfos[ii].index);
                                }
                            }
                        }
                    }
                }
                if (lightInfoBuffer != null)
                {
                    lightInfoBuffer.Dispose ();
                }
                lightInfoBuffer = new ComputeBuffer (lightDataInfos.Length, Marshal.SizeOf (typeof (LightDataInfo)));
                lightInfoBuffer.SetData (lightDataInfos);

                Shader.SetGlobalBuffer ("_LightInfos", lightInfoBuffer);
                if (lightIndexBuffer != null)
                {
                    lightIndexBuffer.Dispose ();
                }
                lightIndexBuffer = new ComputeBuffer (lightIndex.Count, sizeof (uint));
                lightIndexBuffer.SetData (lightIndex);
                Shader.SetGlobalBuffer ("_LightIndex", lightIndexBuffer);

                if (verticalBlockIndexBuffer != null)
                {
                    verticalBlockIndexBuffer.Dispose ();
                }
                if (verticalBlockIndex.Count > 0)
                {
                    verticalBlockIndexBuffer = new ComputeBuffer (verticalBlockIndex.Count, sizeof (uint));
                    verticalBlockIndexBuffer.SetData (verticalBlockIndex);
                    Shader.SetGlobalBuffer ("_VerticalBlockIndex", verticalBlockIndexBuffer);
                }

                if (lightIndexHeadBuffer != null)
                {
                    lightIndexHeadBuffer.Dispose ();
                }
                if (lightIndexHead.Length > 0)
                {
                    lightIndexHeadBuffer = new ComputeBuffer (lightIndexHead.Length, Marshal.SizeOf (typeof (LightHeadIndex)));
                    lightIndexHeadBuffer.SetData (lightIndexHead);
                    Shader.SetGlobalBuffer ("_LightIndexHead", lightIndexHeadBuffer);
                }
            }
        }

        private static void CollectLights (Transform trans, object param)
        {
            if (trans.gameObject.activeInHierarchy)
            {
                Light light = trans.GetComponent<Light> ();
                if (light != null)
                {
                    if (light.enabled)
                    {
                        LightLoopContext llc = param as LightLoopContext;
                        if (light.type == LightType.Point)
                        {

                            Vector3 pos = light.transform.position;
                            {
                                Vector3 min = pos - new Vector3 (light.range, light.range, light.range);
                                Vector3 max = pos + new Vector3 (light.range, light.range, light.range);
                                int xzCount = llc.zCount * llc.xCount;
                                int minx;
                                int minz;
                                int minindex = LightLoopContext.FindChunkIndex (min, llc.lightGridSize, llc.lightGridSize, llc.xCount, llc.zCount, out minx, out minz);
                                int maxx;
                                int maxz;
                                int maxindex = LightLoopContext.FindChunkIndex (max, llc.lightGridSize, llc.lightGridSize, llc.xCount, llc.zCount, out maxx, out maxz);
                                int miny = (int) min.y / llc.lightGridSize;
                                int maxy = (int) max.y / llc.lightGridSize;
                                if (miny < 0)
                                {
                                    miny = 0;
                                }
                                float range2 = light.range * light.range;
                                for (int z = minz; z <= maxz; ++z)
                                {
                                    for (int x = minx; x <= maxx; ++x)
                                    {
                                        float h = -1;
                                        RaycastHit hitinfo;
                                        Vector3 p = new Vector3 (x * llc.lightGridSize, max.y, z * llc.lightGridSize);
                                        if (Physics.Raycast (p, Vector3.down, out hitinfo, 1000, 1 << GameObjectLayerHelper.TerrainLayer))
                                        {
                                            h = hitinfo.point.y;
                                            for (int y = miny; y <= maxy; ++y)
                                            {
                                                float height = (y + 1) * llc.lightGridSize;
                                                if (height > h && h >= 0)
                                                {
                                                    Vector3 offsetPos = new Vector3 (x * llc.lightGridSize, y * llc.lightGridSize, z * llc.lightGridSize);
                                                    if (offsetPos.x < pos.x)
                                                    {
                                                        offsetPos.x += llc.lightGridSize;
                                                    }
                                                    if (offsetPos.y < pos.y)
                                                    {
                                                        offsetPos.y += llc.lightGridSize;
                                                    }
                                                    if (offsetPos.z < pos.z)
                                                    {
                                                        offsetPos.z += llc.lightGridSize;
                                                    }
                                                    float d2 = (offsetPos - pos).sqrMagnitude;
                                                    if (d2 <= range2)
                                                    {
                                                        int id = y * xzCount + z * llc.xCount + x;
                                                        LightBlockInfo blockInfo;
                                                        if (!llc.lightBlocks.TryGetValue (id, out blockInfo))
                                                        {
                                                            blockInfo = new LightBlockInfo ()
                                                            {
                                                                blockId = id,
                                                                offset = new Vector3 (x * llc.lightGridSize, y * llc.lightGridSize, z * llc.lightGridSize),
                                                                xzIndex = z * llc.xCount + x,
                                                                yIndex = y,
                                                            };
                                                            int xx;
                                                            int zz;
                                                            llc.lightBlocks.Add (id, blockInfo);
                                                            blockInfo.chunkID = LightLoopContext.FindChunkIndex (blockInfo.offset, llc.chunkWidth, llc.chunkHeight, llc.widthCount, llc.heightCount, out xx, out zz);
                                                        }
                                                        blockInfo.lights.Add (light);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                        }
                    }
                }
                //else
                {
                    EditorCommon.EnumChildObject (trans, param, funCollectLight);
                }
            }
        }

        private static void IntersectLightObjects (Transform trans, object param)
        {
            if (trans.gameObject.activeInHierarchy)
            {
                ObjectCombine oc = trans.GetComponent<ObjectCombine> ();
                if (oc != null)
                {
                    // oc.lights.Clear ();
                    if (oc.IsRenderValid () && oc.mesh != null)
                    {
                        LightLoopContext llc = param as LightLoopContext;
                        Matrix4x4 matrix = trans.localToWorldMatrix;
 
                        if (oc.mesh.isReadable)
                        {
                            Vector3[] vertices = oc.mesh.vertices;
                            int[] triangles = oc.mesh.triangles;
                            for (int i = 0; i < triangles.Length; i += 3)
                            {
                                Vector3 p0 = matrix.MultiplyPoint (vertices[triangles[i]]);
                                Vector3 p1 = matrix.MultiplyPoint (vertices[triangles[i + 1]]);
                                Vector3 p2 = matrix.MultiplyPoint (vertices[triangles[i + 2]]);
                                float dist01 = Vector3.Distance (p0, p1);
                                float dist02 = Vector3.Distance (p0, p2);
                                int count01 = (int) (dist01 / llc.lightGridSize) + 1;
                                int count02 = (int) (dist02 / llc.lightGridSize) + 1;
                                float percent01 = 1.0f / count01;
                                float percent02 = 1.0f / count02;

                                for (int step01 = 0; step01 <= count01; ++step01)
                                {
                                    float p01 = percent01 * step01;
                                    Vector3 v01 = Vector3.Lerp (p0, p1, p01);
                                    for (int step02 = 0; step02 <= count02; ++step02)
                                    {
                                        float p02 = percent02 * step02;
                                        Vector3 v = Vector3.Lerp (v01, p2, p02);
                                        int xzCount = llc.zCount * llc.xCount;
                                        int x;
                                        int z;
                                        int index = LightLoopContext.FindChunkIndex (v, llc.lightGridSize, llc.lightGridSize, llc.xCount, llc.zCount, out x, out z);
                                        int y = (int) v.y / llc.lightGridSize;
                                        int id = y * xzCount + z * llc.xCount + x;
                                        LightBlockInfo blockInfo;
                                        if (llc.lightBlocks.TryGetValue (id, out blockInfo))
                                        {
                                            blockInfo.objCount++;
                  
                                        }
                                        //pc.points.Add(v);
                                    }
                                }
                            }
                         
                        }
                        else
                        {
                            Debug.LogErrorFormat ("mesh not readable:{0}", oc.name);
                        }

                    }
                }
                // else
                {
                    EditorCommon.EnumChildObject (trans, param, funIntersectLightObjects);
                }
            }
        }
        public void CollectLights ()
        {
            opType = OpType.OpCollectLights;
        }
        private void InnerCollectLights ()
        {
            lightLoopContext.lightBlocks.Clear ();
            lightLoopContext.processMesh.Clear ();
            // lightLoopContext.points.Clear ();
            chunkLightBlocks.Clear ();
            lights.Clear ();
            string path = AssetsConfig.EditorGoPath[0] + "/" + AssetsConfig.EditorGoPath[(int) AssetsConfig.EditorSceneObjectType.Light];
            EditorCommon.EnumTargetObject (path, funCollectLight, lightLoopContext);
            path = AssetsConfig.EditorGoPath[0] + "/" + AssetsConfig.EditorGoPath[(int) AssetsConfig.EditorSceneObjectType.StaticPrefab];
            EditorCommon.EnumTargetObject (path, funIntersectLightObjects, lightLoopContext);
            path = AssetsConfig.EditorGoPath[0] + "/" + AssetsConfig.EditorGoPath[(int) AssetsConfig.EditorSceneObjectType.Prefab];
            EditorCommon.EnumTargetObject (path, funIntersectLightObjects, lightLoopContext);

            maxLightCount = 0;
            minLightCount = 10000;
            float avg = 0;
            int blockCount = 0;
            var it = lightLoopContext.lightBlocks.GetEnumerator ();
            while (it.MoveNext ())
            {
                var value = it.Current.Value;
                if (value.objCount > 0)
                {
                    var chunkLightBlock = chunkLightBlocks.Find ((clb) => { return clb.chunkID == value.chunkID; });
                    if (chunkLightBlock == null)
                    {
                        chunkLightBlock = new ChunkLightBlockInfo ()
                        {
                        chunkID = value.chunkID,
                        };
                        chunkLightBlocks.Add (chunkLightBlock);
                    }
                    for (int i = 0; i < value.lights.Count; ++i)
                    {
                        var l = value.lights[i];

                        int lightIndex = lights.IndexOf (l);
                        if (lightIndex < 0)
                        {
                            lightIndex = lights.Count;
                            lights.Add (l);
                        }
                        var chunkLightIndex = chunkLightBlock.lightIndexs.Find ((cli) => { return cli.xzIndex == value.xzIndex; });
                        if (chunkLightIndex == null)
                        {
                            chunkLightIndex = new ChunkLightIndex ()
                            {
                            xzIndex = value.xzIndex,
                            };
                            chunkLightBlock.lightIndexs.Add (chunkLightIndex);
                        }
                        float y = value.yIndex * lightLoopContext.lightGridSize;
                        if (y < chunkLightIndex.minY)
                        {
                            chunkLightIndex.minY = y;
                        }
                        if (value.yIndex < chunkLightIndex.minYIndex)
                        {
                            chunkLightIndex.minYIndex = value.yIndex;
                        }
                        if (value.yIndex > chunkLightIndex.maxYIndex)
                        {
                            chunkLightIndex.maxYIndex = value.yIndex;
                        }
                        var lightVerticalBlock = chunkLightIndex.lightVerticalBlock.Find ((lvb) => { return lvb.yIndex == value.yIndex; });
                        if (lightVerticalBlock == null)
                        {
                            lightVerticalBlock = new ChunkLightVerticalBlock ()
                            {
                            yIndex = value.yIndex,
                            offset = value.offset
                            };
                            chunkLightIndex.lightVerticalBlock.Add (lightVerticalBlock);
                        }
                        lightVerticalBlock.lightInfos.Add (new ChunkLightInfo ()
                        {
                            light = l,
                                index = lightIndex,
                        });
                    }
                    if (value.lights.Count > maxLightCount)
                    {
                        maxLightCount = value.lights.Count;
                    }
                    if (value.lights.Count < minLightCount)
                    {
                        minLightCount = value.lights.Count;
                    }
                    avg += value.lights.Count;
                    blockCount++;
                }
            }
            chunkLightBlocks.Sort ((x, y) => { return x.chunkID.CompareTo (y.chunkID); });

            InitLighting ();
            lightLoopContext.lightBlocks.Clear ();
            lightLoopContext.processMesh.Clear ();
            if (blockCount > 0)
                Debug.LogErrorFormat ("light count max:{0} min:{1} avg:{2}", maxLightCount, minLightCount, avg / blockCount);
        }
        public void SaveEnvConfig (SceneEnvConfig sec)
        {
            if (re != null)
            {
                SyncLightInfo (sceneRuntimeLight0, ref re.lighting.sceneLightInfo0, null, ref sceneRuntimeLight0Rot);
                SyncLightInfo (sceneRuntimeLight1, ref re.lighting.sceneLightInfo1, null, ref sceneRuntimeLight1Rot);

                re.SaveEnvConfig (sec);
            }
        }

        public void LoadEnvConfig (SceneEnvConfig sec)
        {

            SyncLightInfo (roleLight0, ref sec.lighting.roleLightInfo0);
            SyncLightInfo (roleLight1, ref sec.lighting.roleLightInfo1);
            SyncLightInfo (sceneRuntimeLight0, ref sec.lighting.sceneLightInfo0);
            SyncLightInfo (sceneRuntimeLight1, ref sec.lighting.sceneLightInfo1);
            if (re != null)
            {
                re.LoadEnvConfig (sec);
            }
        }

        public void EnableShadow (bool enable)
        {
            if (bakeSceneLight0 != null)
            {
                bakeSceneLight0.shadows = enable ? LightShadows.Soft : LightShadows.None;
            }
        }
#endregion

#region testEnvArea
        public void RefreshEnvArea ()
        {
                envObjects.Clear ();
                string path = AssetsConfig.EditorGoPath[0] + "/" + AssetsConfig.EditorGoPath[(int) AssetsConfig.EditorSceneObjectType.Enverinment];
                GameObject envGo = GameObject.Find (path);
                if (envGo != null)
                {
                    envGo.transform.GetComponentsInChildren<EnverinmentArea> (true, envObjects);
                }
                if (lastArea != null)
                    lastArea.active = false;
                lastArea = null;
                re.lighting.needUpdate = true;
                re.ambient.needUpdate = true;
                re.fog.needUpdate = true;
            
        }

        private void UpdateArea ()
        {
            if (currentArea != null)
            {
                if (envLerpTime > -0.5f)
                {
                    int state = 0;
                    if (envLerpTime < 0.001f)
                    {
                        state = 1; //start
                    }
                    else if (envLerpTime > 1)
                    {
                        state = 2;
                        envLerpTime = 1;
                    }

                    EnverinmentContext context;
                    context.env = re;
                    context.envMgr = RenderingManager.instance;
                    for (int i = 0; i < currentArea.envModifyList.Length; ++i)
                    {
                        var wrapper = currentArea.envModifyList[i];
                        if (wrapper != null && wrapper.envLerp != null)
                        {
                            wrapper.envLerp.Lerp (ref context, envLerpTime, intoEnvArea, state);
                        }
                    }
                    if (state == 2)
                    {
                        envLerpTime = -1;
                    }
                    else
                        envLerpTime += Time.deltaTime;
                }
                else
                {
                    EnverinmentContext context;
                    context.env = re;
                    context.envMgr = RenderingManager.instance;
                    for (int i = 0; i < currentArea.envModifyList.Length; ++i)
                    {
                        var wrapper = currentArea.envModifyList[i];
                        if (wrapper != null && wrapper.envLerp != null)
                        {
                            wrapper.Update ();
                            wrapper.envLerp.Lerp (ref context, 1, intoEnvArea, 2);
                        }
                    }
                }

            }
        }
#endregion

#region freeCamera
        void OnValidate ()
        {
        }

        void CaptureInput ()
        {
            Cursor.lockState = CursorLockMode.Locked;

            Cursor.SetCursor (ms_invisibleCursor, Vector2.zero, CursorMode.ForceSoftware);
            m_inputCaptured = true;

            m_yaw = transform.eulerAngles.y;
            m_pitch = transform.eulerAngles.x;
        }
        void ReleaseInput ()
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.SetCursor (null, Vector2.zero, CursorMode.Auto);
            m_inputCaptured = false;
        }
        void OnApplicationFocus (bool focus)
        {
            if (m_inputCaptured && !focus)
                ReleaseInput ();
        }
        void UpdateFreeCamera ()
        {
            if (!m_inputCaptured)
            {
                if (!holdRightMouseCapture && Input.GetMouseButtonDown (0))
                    CaptureInput ();
                else if (holdRightMouseCapture && Input.GetMouseButtonDown (1))
                    CaptureInput ();
            }

            if (!m_inputCaptured)
                return;

            if (m_inputCaptured)
            {
                if (!holdRightMouseCapture && Input.GetKeyDown (KeyCode.Escape))
                    ReleaseInput ();
                else if (!holdRightMouseCapture && Input.GetKeyDown (KeyCode.Q))
                {
                    ReleaseInput ();
                    forceUpdateFreeCamera = false;
                }
                else if (holdRightMouseCapture && Input.GetMouseButtonUp (1))
                    ReleaseInput ();

            }

            var rotStrafe = Input.GetAxis ("Mouse X");
            var rotFwd = Input.GetAxis ("Mouse Y");

            m_yaw = (m_yaw + lookSpeed * rotStrafe) % 360f;
            m_pitch = (m_pitch - lookSpeed * rotFwd) % 360f;
            transform.rotation = Quaternion.AngleAxis (m_yaw, Vector3.up) * Quaternion.AngleAxis (m_pitch, Vector3.right);

            var speed = Time.deltaTime * (Input.GetKey (KeyCode.LeftShift) ? sprintSpeed : moveSpeed);
            var forward = speed * Input.GetAxis ("Vertical");
            var right = speed * Input.GetAxis ("Horizontal");
            var up = speed * ((Input.GetKey (KeyCode.E) ? 1f : 0f) - (Input.GetKey (KeyCode.Q) ? 1f : 0f));
            transform.position += transform.forward * forward + transform.right * right + Vector3.up * up;

        }
#endregion
#region shadow
        private void UpdateShadowCaster ()
        {

            if (lookTarget == null)
            {
                GameObject go = GameObject.Find ("LookTarget");
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
            shadowCasters.Clear ();
            if (shadowCasterProxy == null)
            {
                shadowCasterProxy = GameObject.Find ("ShadowCaster");
            }
            shadowRenderBatchs.Clear ();
            bool first = true;
            Bounds shadowBound = new Bounds (Vector3.zero, Vector3.zero);
            if (shadowCasterProxy != null)
            {
                shadowCasterProxy.GetComponentsInChildren<Renderer> (false, shadowCasters);
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
                            RenderBatch rb = new RenderBatch ();
                            rb.render = render;
                            rb.mat = render.sharedMaterial;
                            rb.mpbRef = null;
                            rb.passID = 0;
                            shadowRenderBatchs.Add (rb);
                            if (first)
                            {
                                shadowBound = render.bounds;
                                first = false;
                            }
                            else
                                shadowBound.Encapsulate (render.bounds);
                        }
                    }
                }
            }
            re.sceneData.shadowBound = shadowBound;
        }

        private void BuildShadowMap ()
        {
            if (shadowMap != null && shadowMat != null)
            {
                shadowMapCb.Clear ();
                shadowMapCb.ClearRenderTarget (true, true, Color.clear, 1.0f);
                shadowMapCb.SetViewProjectionMatrices (re.sceneData.shadowViewMatrix, re.sceneData.shadowProjMatrix);

                if (shadowCasters.Count > 0)
                {
                    for (int i = 0; i < shadowCasters.Count; ++i)
                    {
                        Renderer render = shadowCasters[i];
                        if (render != null &&
                            render.enabled &&
                            render.shadowCastingMode != ShadowCastingMode.Off)
                            shadowMapCb.DrawRenderer (render, shadowMat, 0, 0);
                    }
                    Graphics.SetRenderTarget (shadowMap);
                    Graphics.ExecuteCommandBuffer (shadowMapCb);
                    Graphics.SetRenderTarget (null);
                }

            }
        }
        public void PrepareShadowCaster ()
        {
            GameObject go = GameObject.Find ("LookTarget");
            if (go == null)
            {
                lookTarget = new GameObject ("LookTarget").transform;
            }
            go = GameObject.Find ("ShadowCaster");
            if (go == null)
            {
                shadowCasterProxy = new GameObject ("ShadowCaster");
            }
        }
#endregion
#region gizmo

        public static void RegisterDrawGizmo (OnDrawGizmoCb cb)
        {
            m_drawGizmo = cb;
        }
        void EditChunk (SceneChunk chunk, bool add)
        {
            if (add)
            {
                chunks[chunk.chunkId] = chunk;
            }
            else
            {
                chunks.Remove (chunk.chunkId);
            }
        }
        void SetRes (System.Object obj, int type)
        {
            if (type == 0)
            {
                shadowMap = obj as RenderTexture;
            }
            else if (type == 1)
            {
                quadTreeRef = obj as QuadTree;
            }
            // else if (type == 2)
            // {
            //     envAreasRef = obj as EnvArea[];
            // }
            else if (type == 3)
            {
                sceneMats = obj as Material[];
            }
            else if (type == 4)
            {
                dynamicObjects = obj as Dictionary<uint, SceneDynamicObject>;
            }

        }

        void DrawBox (bool draw, Bounds aabb)
        {
            switch (drawType)
            {
                case DrawType.Both:
                    {
                        if (draw)
                        {
                            Gizmos.color = Color.red;
                        }
                        else
                            Gizmos.color = Color.magenta;
                        Gizmos.DrawWireCube (aabb.center, aabb.size);
                    }
                    break;
                case DrawType.Draw:
                    {
                        if (draw)
                        {
                            Gizmos.color = Color.red;
                            Gizmos.DrawWireCube (aabb.center, aabb.size);
                        }

                    }
                    break;
                case DrawType.Cull:
                    {
                        if (!draw)
                        {
                            Gizmos.color = Color.magenta;
                            Gizmos.DrawWireCube (aabb.center, aabb.size);
                        }
                    }
                    break;
            }

        }

        void DrawLightHandle (Transform t, Vector3 centerPos, float right, float up, Light l, string text)
        {
            if (l != null)
            {
                Vector3 pos = centerPos + t.right * right + t.up * up;
                EditorGUI.BeginChangeCheck ();
                Transform lt = l.transform;
                Quaternion rot = Handles.RotationHandle (lt.rotation, pos);
                if (EditorGUI.EndChangeCheck ())
                {
                    lt.rotation = rot;
                }

                Handles.color = l.color;
                Handles.ArrowHandleCap (100, pos, rot, 2 * l.intensity, EventType.Repaint);
                Handles.Label (pos, text);
            }
        }
        void DrawSceneObjectBox (SceneChunk sc, SceneQuadTreeNode node)
        {
            if (sc.sceneObjects.IsValid () && sc.chunkState >= SceneChunk.ESceneChunkState.ChunkDataLoadFinish)
            {
                SceneQuadBlock sqb = sc.sceneObjects.SafeGet<SceneQuadBlock> (node.sceneQuadBlockIndex);
                if (sqb != null)
                {
                    ushort soStart = 0;
                    ushort soCount = 0;
                    if (sc.SafeGetSceneObjectGroupIndex (sqb.sceneObjectGroupIndex, ref soStart, ref soCount))
                    {
                        for (ushort i = 0; i < soCount; ++i)
                        {
                            int soIndex = sc.GetSceneObjectIndex (soStart, i) + sc.sceneObjecStart;
                            SceneObject so = sc.sceneObjects.SafeGet<SceneObject> (soIndex);
                            if (so != null && so.asset.obj != null && so.mpRef != null)
                            {
                                DrawBox (so.draw, so.aabb);
                                if (updateSceneObject)
                                    sceneObjects.Add (so);
                            }
                        }
                    }
                }

            }
        }
        void OnDrawGizmos ()
        {
            Color color = Gizmos.color;
            if (mainCamera == null)
                mainCamera = GetComponent<Camera> ();
            if (mainCamera != null)
            {
                if (drawFrustum)
                    CameraEditorUtils.DrawFrustumGizmo (mainCamera);
            }
            if (drawWind)
            {
                Gizmos.color = Color.red;
                Vector3 pos = re.randomWind.WindPos;
                Gizmos.DrawWireSphere (pos, 0.3f);

                Gizmos.color = Color.blue;
                Vector3 forward = re.randomWind.rotation * Vector3.forward;

                Gizmos.DrawLine (pos, pos + forward * re.randomWind.m_windDir.w * 10);

            }
            if (drawShadowLighing)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireSphere (re.translatePos, 0.3f);
                Vector3 targetPos = re.translatePos + re.lightProjectForward * 10;
                Gizmos.DrawWireSphere (targetPos, 0.3f);
                Vector3 leftUp = re.translatePos + re.lightProjectUp * re.shadowOrthoSize - re.lightProjectRight * re.shadowOrthoSize;
                Vector3 rightUp = re.translatePos + re.lightProjectUp * re.shadowOrthoSize + re.lightProjectRight * re.shadowOrthoSize;
                Vector3 leftBottom = re.translatePos - re.lightProjectUp * re.shadowOrthoSize - re.lightProjectRight * re.shadowOrthoSize;
                Vector3 rightBottom = re.translatePos - re.lightProjectUp * re.shadowOrthoSize + re.lightProjectRight * re.shadowOrthoSize;
                Gizmos.DrawLine (leftBottom, rightBottom);
                Gizmos.DrawLine (rightBottom, rightUp);
                Gizmos.DrawLine (rightUp, leftUp);
                Gizmos.DrawLine (leftUp, leftBottom);

                leftUp = targetPos + re.lightProjectUp * re.shadowOrthoSize - re.lightProjectRight * re.shadowOrthoSize;
                rightUp = targetPos + re.lightProjectUp * re.shadowOrthoSize + re.lightProjectRight * re.shadowOrthoSize;
                leftBottom = targetPos - re.lightProjectUp * re.shadowOrthoSize - re.lightProjectRight * re.shadowOrthoSize;
                rightBottom = targetPos - re.lightProjectUp * re.shadowOrthoSize + re.lightProjectRight * re.shadowOrthoSize;
                Gizmos.DrawLine (leftBottom, rightBottom);
                Gizmos.DrawLine (rightBottom, rightUp);
                Gizmos.DrawLine (rightUp, leftUp);
                Gizmos.DrawLine (leftUp, leftBottom);
                Handles.ArrowHandleCap (100, re.translatePos, Quaternion.LookRotation (re.lightProjectForward), 1, EventType.Repaint);
            }
            if (shadowBound)
            {
                Gizmos.DrawWireCube (re.sceneData.shadowBound.center, re.sceneData.shadowBound.size);
            }

            if (quadTreeRef != null)
            {
                if (updateSceneObject)
                {
                    sceneObjects.Clear ();
                }
                int level = (int) quadLevel;
                if (level != 0)
                {
                    var it = chunks.GetEnumerator ();
                    while (it.MoveNext ())
                    {
                        var sc = it.Current.Value;
                        if (sc.usedChunkStartIndex >= 0)
                        {
                            int quadTreeOffset = sc.usedChunkStartIndex * QuadTree.blockNodeCount;
                            int start = 0;
                            int end = 1;
                            if (level == 2)
                            {
                                start = 1;
                                end = 5;
                            }
                            else if (level == 3 || level == 4)
                            {
                                start = 5;
                                end = 21;
                            }
                            var nodeList = quadTreeRef.nodeList;

                            if (level != 4)
                            {
                                for (int i = start; i < end; ++i)
                                {
                                    SceneQuadTreeNode node = nodeList[quadTreeOffset + i];
                                    if (node != null)
                                    {
                                        DrawBox (node.draw, node.aabb);
                                    }
                                }
                            }
                        }
                    }
                }

                if (drawLodGrid)
                {
                    var nodeList = quadTreeRef.nodeList;

                    var it = chunks.GetEnumerator ();
                    while (it.MoveNext ())
                    {
                        var chunk = it.Current.Value;
                        if (chunk.usedChunkStartIndex >= 0)
                        {
                            int quadTreeOffset = chunk.usedChunkStartIndex * QuadTree.blockNodeCount;

                            for (int i = 5; i < QuadTree.blockNodeCount; ++i)
                            {
                                SceneQuadTreeNode node = nodeList[quadTreeOffset + i];
                                if (node != null && node.lodLevel == 0)
                                {
                                    Gizmos.color = Color.yellow;
                                    Gizmos.DrawWireCube (node.aabb.center, node.aabb.size);
                                }
                            }
                            if (drawTerrainGrid)
                            {
                                var to = chunk.terrainObject;
                                Gizmos.color = Color.cyan;
                                SceneQuadTreeNode node0 = nodeList[quadTreeOffset + 1];
                                if (node0 != null && node0.lodLevel == 0)
                                    Gizmos.DrawWireCube (to.aabb0.center, to.aabb0.size);
                                SceneQuadTreeNode node1 = nodeList[quadTreeOffset + 2];
                                if (node1 != null && node1.lodLevel == 0)
                                    Gizmos.DrawWireCube (to.aabb1.center, to.aabb1.size);
                                SceneQuadTreeNode node2 = nodeList[quadTreeOffset + 3];
                                if (node2 != null && node2.lodLevel == 0)
                                    Gizmos.DrawWireCube (to.aabb2.center, to.aabb2.size);
                                SceneQuadTreeNode node3 = nodeList[quadTreeOffset + 4];
                                if (node3 != null && node3.lodLevel == 0)
                                    Gizmos.DrawWireCube (to.aabb3.center, to.aabb3.size);
                            }
                        }
                    }
                }

                if (drawPointLight)
                {
                    if (XFxMgr.fxPointLight != null)
                    {
                        var point = XFxMgr.fxPointLight;
                        Gizmos.color = new Color (point.color.x, point.color.y, point.color.z, 1);
                        Gizmos.DrawWireSphere (new Vector3 (point.pos.x, point.pos.y, point.pos.z), point.pos.w);
                    }
                    else if (re.sceneData.scenePointLight != null)
                    {
                        var point = re.sceneData.scenePointLight;
                        Gizmos.color = new Color (point.color.x, point.color.y, point.color.z, 1);
                        Gizmos.DrawWireSphere (new Vector3 (point.pos.x, point.pos.y, point.pos.z), point.pos.w);
                    }
                }
            }
            if (debugEnvArea)
            {
                    for (int i = 0; i < envObjects.Count; ++i)
                    {
                        var envObject = envObjects[i];
                        Transform t = envObject.transform;

                        for (int j = 0; j < envObject.areaList.Count; ++j)
                        {
                            var areaBox = envObject.areaList[j];
                            if (envObject == lastArea)
                            {
                                Color boxColor = Color.Lerp (new Color (1, 1, 1, 0.2f), Color.white, lastArea.blinkTime);
                                lastArea.blinkTime += Time.deltaTime;
                                if (lastArea.blinkTime > 1.0f)
                                {
                                    lastArea.blinkTime = 0;
                                }
                                Gizmos.color = boxColor;
                            }
                            else
                                Gizmos.color = envObject.color;

                            Vector3 worldPos = areaBox.center + t.position;
                            Quaternion rot = Quaternion.Euler (0, areaBox.rotY, 0);
                            Gizmos.matrix = Matrix4x4.TRS (worldPos, rot, Vector3.one);
                            Gizmos.DrawWireCube (Vector3.zero, areaBox.size);
                            Gizmos.matrix = Matrix4x4.identity;
                        }
                    }
                
            }
            if (drawTerrainHeight)
            {
                int chunkId = re.sceneData.chunkID.y * re.sceneData.xChunkCount + re.sceneData.chunkID.x;
                SceneChunk sc;
                if (chunks.TryGetValue (chunkId, out sc))
                {
                    if (sc.heights.IsCreated)
                    {
                        Gizmos.color = Color.green;
                        float xOffset = sc.x * re.sceneData.ChunkWidth;
                        float zOffset = sc.z * re.sceneData.ChunkHeight;
                        Vector3 v0 = Vector3.zero;
                        Vector3 v1 = Vector3.zero;
                        Vector3 v2 = Vector3.zero;

                        for (int z = 0; z < SceneData.terrainGridCount; ++z)
                        {
                            for (int x = 0; x < SceneData.terrainGridCount; ++x)
                            {
                                int p0 = z * (SceneData.terrainGridCount + 1) + x;
                                int p1 = z * (SceneData.terrainGridCount + 1) + x + 1;
                                int p2 = (z + 1) * (SceneData.terrainGridCount + 1) + x + 1;
                                int p3 = (z + 1) * (SceneData.terrainGridCount + 1) + x;
                                float h0 = sc.heights[p0];
                                float h1 = sc.heights[p1];
                                float h2 = sc.heights[p2];
                                float h3 = sc.heights[p3];

                                float x0 = x * SceneData.terrainGridSize + xOffset;
                                float x1 = (x + 1) * SceneData.terrainGridSize + xOffset;
                                float z0 = z * SceneData.terrainGridSize + zOffset;
                                float z1 = (z + 1) * SceneData.terrainGridSize + zOffset;
                                if (re.sceneData.terrainVertex.w == 1)
                                {
                                    if (re.sceneData.terrainVertex.x == p2 &&
                                        re.sceneData.terrainVertex.y == p0 &&
                                        re.sceneData.terrainVertex.z == p1)
                                    {
                                        v0 = new Vector3 (x0, h0, z0);
                                        v1 = new Vector3 (x1, h1, z0);
                                        v2 = new Vector3 (x1, h2, z1);
                                    }
                                }
                                else
                                {
                                    if (re.sceneData.terrainVertex.x == p2 &&
                                        re.sceneData.terrainVertex.y == p3 &&
                                        re.sceneData.terrainVertex.z == p0)
                                    {
                                        v0 = new Vector3 (x0, h0, z0);
                                        v1 = new Vector3 (x1, h2, z1);
                                        v2 = new Vector3 (x0, h3, z1);
                                    }
                                }

                                Gizmos.DrawLine (new Vector3 (x0, h0, z0), new Vector3 (x1, h2, z1));
                                Gizmos.DrawLine (new Vector3 (x1, h2, z1), new Vector3 (x0, h3, z1));
                                Gizmos.DrawLine (new Vector3 (x0, h3, z1), new Vector3 (x0, h0, z0));

                                Gizmos.DrawLine (new Vector3 (x0, h0, z0), new Vector3 (x1, h1, z0));
                                Gizmos.DrawLine (new Vector3 (x1, h1, z0), new Vector3 (x1, h2, z1));
                                Gizmos.DrawLine (new Vector3 (x1, h2, z1), new Vector3 (x0, h0, z0));
                            }
                        }
                        Gizmos.color = Color.yellow;
                        Gizmos.DrawLine (v0, v1);
                        Gizmos.DrawLine (v1, v2);
                        Gizmos.DrawLine (v2, v0);
                    }
                }
            }
            if (drawLightBox)
            {
                Vector3 size = new Vector3 (lightLoopContext.lightGridSize / 2.0f, lightLoopContext.lightGridSize / 2.0f, lightLoopContext.lightGridSize / 2.0f);
                for (int i = 0; i < chunkLightBlocks.Count; ++i)
                {
                    var clb = chunkLightBlocks[i];
                    //if (saveChunkContext.toMap.ContainsKey(clb.chunkID))
                    {
                        for (int j = 0; j < clb.lightIndexs.Count; ++j)
                        {
                            var cli = clb.lightIndexs[j];
                            for (int k = 0; k < cli.lightVerticalBlock.Count; ++k)
                            {
                                var clvb = cli.lightVerticalBlock[k];
                                if (previewLightCount == 0 || clvb.lightInfos.Count == previewLightCount)
                                {
                                    int debugIndex = Mathf.Clamp (clvb.lightInfos.Count / 5, 0, 4);
                                    Gizmos.color = lightInfoColor[debugIndex];
                                    Gizmos.DrawWireCube (clvb.offset + size, size);
                                }
                            }
                        }

                    }
                }
            }
            if (m_drawGizmo != null && drawSceneBounds)
            {
                m_drawGizmo ();
            }

            Gizmos.color = color;
        }

        private void DrawObject (SceneObject so, ComputeBuffer argBuffer, Material inVisibleMat)
        {
        }

        public void SceneView_Update ()
        {
            GetSceneViewCamera ();
            if (commandBuffer != null)
            {
                for (int i = 0; i < commandBuffer.Length; ++i)
                {
                    CommandBuffer cb = commandBuffer[i];
                    cb.Clear ();
                }
            }
        }

        public void RenderMesh (Mesh mesh, Matrix4x4 matrix, Material material, MaterialPropertyBlock mpb)
        {

            RenderBatch rb = new RenderBatch ();
            rb.mesh = mesh;
            rb.matrix = matrix;
            rb.mat = material;
            rb.mpbRef = mpb;
            rb.passID = 0;
            renderBatches.Add (rb);
            RenderingManager.instance.SetPostAlphaCommand (renderBatches);
            // editorCommandBuffer.DrawMesh(mesh, matrix, material, 0, 0);
        }
#endregion
    }
}
#endif
