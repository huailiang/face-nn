#if UNITY_EDITOR
using CFUtilPoolLib;
using UnityEngine;
namespace XEngine
{
    public interface IQuadTreeObject
    {
        int BlockId { get; }

        int QuadNodeId { get; set; }

        Bounds bounds { get; }
    }

    [DisallowMultipleComponent, ExecuteInEditMode]
    public class ObjectCombine : MonoBehaviour, IQuadTreeObject
    {
        // public Renderer render;
        public Material material;
        public Mesh mesh;
        public Bounds aabb;
        //block info
        public int chunkID = -1;
        public int chunkLevel = 2; //0,1,2
        public int blockId = 0;
        public bool forceLocalObject = false;
        //lod
        public float lod0Dist = 10;
        public float lod1Dist = 100;
        //lightmap
        public float lightmapScale = 1.0f;
        public int lightmapChunkId = -1;
        public string fileID = "";
        public Texture lightmap;
        public Vector4 lightmapUVST;
        
        //static batch
        public int batchMeshIndex = -1;
        public short subMeshIndex = -1;
        public short staticBatchGameObjectIndex = -1;
        public int priority = 0;
        
        [System.NonSerialized]
        public MaterialPropertyBlock mpb = null;
        [System.NonSerialized]
        public static bool isPreview = false;

        [System.NonSerialized]
        private Renderer render;



        public int BlockId { get { return blockId; } }
        public int QuadNodeId { get; set; }
        public Bounds bounds { get { return aabb; } }

        public Renderer GetRenderer()
        {
            if (render == null)
            {
                render = GetComponent<Renderer>();
            }
            return render;
        }
        public bool IsRenderValid()
        {
            return GetRenderer() != null && GetRenderer().enabled && GetRenderer().gameObject.activeInHierarchy;
        }

        public void UpdateLightmap()
        {
            if (mpb == null)
            {
                mpb = new MaterialPropertyBlock();
            }
            if (lightmap != null)
            {
                mpb.SetTexture(ShaderManager._ShaderKeyLightmap, lightmap);
                mpb.SetVector(ShaderManager._ShaderKeyLightmapST, lightmapUVST);
            }
            if (GetRenderer() != null)
            {
                GetRenderer().SetPropertyBlock(mpb);
            }
        }

        public void ManualUpdate()
        {
            if (GetRenderer() != null)
            {
                if (isPreview)
                {
                    if (mpb != null)
                    {
                        GetRenderer().SetPropertyBlock(mpb);
                    }
                    else
                    {
                        GetRenderer().SetPropertyBlock(null);
                    }
                }
            }
        }
    }
}
#endif