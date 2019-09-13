#if UNITY_EDITOR
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
namespace CFEngine
{
    public enum AtlasSize
    {
        E512x512,
        E1024x1024,
        E2048x2048,
    }

    [System.Serializable]
    public class MaterialCombineTex2D
    {
        public Vector2Int atlasPos = new Vector2Int(-1, -1);
        public Vector2Int atlasSize = new Vector2Int(1, 1);
        public SpriteSize spriteWidth = SpriteSize.E256x256;
        public SpriteSize spriteHeight = SpriteSize.E256x256;
        public MaterialMeshGroup mmgRef = null;

    }

    [DisallowMultipleComponent, ExecuteInEditMode]
    public class MaterialCombine : MonoBehaviour
    {
        public List<MaterialCombineTex2D> materialTex2D = new List<MaterialCombineTex2D>();
        public Texture2D atlas0;
        public Texture2D atlas1;
        public AtlasSize atlasWidth = AtlasSize.E1024x1024;
        public AtlasSize atlasHeight = AtlasSize.E1024x1024;
        public SpriteSize spriteSize = SpriteSize.E256x256;
        public bool hasAlpha = false;
        public bool textureArray = true;
    }

    public static class AtlasUtility
    {
        public static int GetAtlasSize(this AtlasSize size)
        {
            switch (size)
            {
                case AtlasSize.E512x512:
                    return 512;
                case AtlasSize.E1024x1024:
                    return 1024;
                case AtlasSize.E2048x2048:
                    return 2048;
                default:
                    return 1024;
            }
        }
        public static int GetSpriteSize(this SpriteSize size)
        {
            switch (size)
            {
                case SpriteSize.E128x128:
                    return 128;
                case SpriteSize.E256x256:
                    return 256;
                case SpriteSize.E512x512:
                    return 512;
                default:
                    return 256;
            }
        }
    }
}
#endif