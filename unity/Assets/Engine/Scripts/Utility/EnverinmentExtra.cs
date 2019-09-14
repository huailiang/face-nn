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

    public class MeshVertex
    {
        public Vector3[] vertices;
        public int[] triangles;
    }

    [Serializable]
    public class LightLoopContext
    {
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
        public bool drawLodGrid = false;
        public bool drawTerrainGrid = false;
        public int quadIndex = -1;
        public DrawType drawType = DrawType.Both;
        public bool showObjects = false;

        public bool drawPointLight = false;
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


#region misc
        [NonSerialized]
        public RenderingEnvironment re;
        Camera mainCamera;
        Camera sceneViewCamera;
        public static bool drawSceneBounds = false;
        private static OnDrawGizmoCb m_drawGizmo = null;
#endregion

        void Awake ()
        {
            GetSceneViewCamera ();
        }

        void Start ()
        {
            Shader.SetGlobalFloat("_GlobalDebugMode", 0);
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
        
            UpdateShadowCaster ();
            BuildShadowMap ();
            
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
            if (shadowMapCb == null)
                shadowMapCb = new CommandBuffer { name = "Editor Shadow Map Cb" };
         

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
            if (re != null)
            {
                PrepareTransformGui (roleLight0, ref roleLight0Rot);
                PrepareTransformGui (roleLight1, ref roleLight1Rot);
                SyncLightInfo (roleLight0, ref re.lighting.roleLightInfo0, null, ref roleLight0Rot);
                SyncLightInfo (roleLight1, ref re.lighting.roleLightInfo1, null, ref roleLight1Rot);
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
            l.cullingMask = mask;
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
                light.cullingMask = 1 << GameObjectLayerHelper.InVisiblityLayer;
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
                                                            llc.lightBlocks.Add (id, blockInfo);
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
            }
        }

        private static void IntersectLightObjects (Transform trans, object param)
        {
            if (trans.gameObject.activeInHierarchy)
            {
                ObjectCombine oc = trans.GetComponent<ObjectCombine> ();
                if (oc != null)
                {
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
            }
        }

        public void SaveEnvConfig (SceneEnvConfig sec)
        {
            if (re != null)
            {
                re.SaveEnvConfig (sec);
            }
        }

        public void LoadEnvConfig (SceneEnvConfig sec)
        {
            SyncLightInfo (roleLight0, ref sec.lighting.roleLightInfo0);
            SyncLightInfo (roleLight1, ref sec.lighting.roleLightInfo1);
            if (re != null)
            {
                re.LoadEnvConfig (sec);
            }
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
  

        void OnDrawGizmos ()
        {
            Color color = Gizmos.color;
            if (mainCamera == null)
                mainCamera = GetComponent<Camera> ();
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
            }
            
            if (m_drawGizmo != null && drawSceneBounds)
            {
                m_drawGizmo ();
            }
            Gizmos.color = color;
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
        }
#endregion
    }
}
#endif
