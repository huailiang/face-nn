#if UNITY_EDITOR
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
namespace CFEngine
{
    [DisallowMultipleComponent, ExecuteInEditMode]
    public class MaterialMeshGroup : MonoBehaviour
    {
        public Material mat;
        public List<Mesh> meshs = new List<Mesh>();

        [System.NonSerialized]
        public Texture2D tex0;
        [System.NonSerialized]
        public Texture2D tex1;
        [System.NonSerialized]
        public Vector2 previewSize;
        [System.NonSerialized]
        public Rect previewRect;
    }
}
#endif