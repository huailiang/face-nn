#if UNITY_EDITOR
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
namespace CFEngine
{
    [DisallowMultipleComponent, ExecuteInEditMode]
    public class NormalCombine : MonoBehaviour
    {
        public Texture2D normal0;
        public Texture2D normal1;
        public SpriteSize normalSize = SpriteSize.E256x256;
        public Texture2D normal2;
        public Texture2D blend;
        public Texture2D combineNormal;
        public Texture2D combineBlend;

        public int GetNormalSize()
        {
            switch (normalSize)
            {
                case SpriteSize.E64x64:
                    return 64;
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