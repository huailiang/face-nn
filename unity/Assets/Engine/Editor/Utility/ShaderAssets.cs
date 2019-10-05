using CFUtilPoolLib;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace XEngine.Editor
{
    public struct ShaderPropertyValue
    {
        public int shaderID;
        public string shaderKeyName;
        public Vector4 value;
    }

    public struct ShaderTexPropertyValue
    {
        public int shaderID;
        public string shaderKeyName;
        public Texture value;
        public string path;
    }

    internal struct MaterialContext
    {
        public List<ShaderTexPropertyValue> textureValue;
        public short customMatIndex;
        public uint flag;
        public List<ShaderPropertyValue> shaderIDs;

        public void SetFlag(EMatFlag f, bool add)
        {
            if (add)
            {
                flag |= (uint)f;
            }
            else
            {
                flag &= ~((uint)f);
            }
        }

        public bool HasFlag(EMatFlag f)
        {
            return (flag & (uint)f) != 0;
        }

        public static MaterialContext GetContext()
        {
            MaterialContext context = new MaterialContext();
            context.textureValue = new List<ShaderTexPropertyValue>();
            context.shaderIDs = new List<ShaderPropertyValue>();
            context.customMatIndex = -1;
            return context;
        }
    }

    internal class ShaderAssets
    {

        internal class ShaderValue
        {
            public ShaderValue(string n, ShaderUtil.ShaderPropertyType t)
            {
                name = n;
                type = t;
            }
            public string name = "";
            public ShaderUtil.ShaderPropertyType type = ShaderUtil.ShaderPropertyType.Float;

            public virtual void SetValue(Material mat) { }

            public static void GetShaderValue(Material mat, List<ShaderValue> shaderValueLst)
            {
                Shader shader = mat.shader;
                int count = ShaderUtil.GetPropertyCount(shader);
                for (int i = 0; i < count; ++i)
                {
                    ShaderValue sv = null;
                    string name = ShaderUtil.GetPropertyName(shader, i);
                    ShaderUtil.ShaderPropertyType type = ShaderUtil.GetPropertyType(shader, i);
                    switch (type)
                    {
                        case ShaderUtil.ShaderPropertyType.Color:
                            sv = new ShaderColorValue(name, type, mat);
                            break;
                        case ShaderUtil.ShaderPropertyType.Vector:
                            sv = new ShaderVectorValue(name, type, mat);
                            break;
                        case ShaderUtil.ShaderPropertyType.Float:
                            sv = new ShaderFloatValue(name, type, mat);
                            break;
                        case ShaderUtil.ShaderPropertyType.Range:
                            sv = new ShaderFloatValue(name, type, mat);
                            break;
                        case ShaderUtil.ShaderPropertyType.TexEnv:
                            sv = new ShaderTexValue(name, type, mat);
                            break;
                    }
                    shaderValueLst.Add(sv);
                }
                ShaderKeyWordValue keyword = new ShaderKeyWordValue(mat);
                shaderValueLst.Add(keyword);
            }
        }

        internal class ShaderIntValue : ShaderValue
        {
            public ShaderIntValue(string n, ShaderUtil.ShaderPropertyType t, Material mat) : base(n, t)
            {
                value = mat.GetInt(n);
            }
            public int value = 0;
            public override void SetValue(Material mat)
            {
                mat.SetInt(name, value);
            }
        }

        internal class ShaderFloatValue : ShaderValue
        {
            public ShaderFloatValue(string n, ShaderUtil.ShaderPropertyType t, Material mat) : base(n, t)
            {
                value = mat.GetFloat(n);
            }
            public float value = 0;
            public override void SetValue(Material mat)
            {
                mat.SetFloat(name, value);
            }
        }

        internal class ShaderVectorValue : ShaderValue
        {
            public ShaderVectorValue(string n, ShaderUtil.ShaderPropertyType t, Material mat) : base(n, t)
            {
                value = mat.GetVector(n);
            }
            public Vector4 value = Vector4.zero;
            public override void SetValue(Material mat)
            {
                mat.SetVector(name, value);
            }
        }
        internal class ShaderColorValue : ShaderValue
        {
            public ShaderColorValue(string n, ShaderUtil.ShaderPropertyType t, Material mat) : base(n, t)
            {
                value = mat.GetColor(n);
            }
            public Color value = Color.black;
            public override void SetValue(Material mat)
            {
                mat.SetColor(name, value);
            }
        }

        internal class ShaderTexValue : ShaderValue
        {
            public ShaderTexValue(string n, ShaderUtil.ShaderPropertyType t, Material mat) : base(n, t)
            {
                value = mat.GetTexture(n);
                offset = mat.GetTextureOffset(n);
                scale = mat.GetTextureScale(n);
            }
            public Texture value = null;
            public Vector2 offset = Vector2.zero;
            public Vector2 scale = Vector2.one;

            public override void SetValue(Material mat)
            {
                mat.SetTexture(name, value);
            }
        }

        internal class ShaderKeyWordValue : ShaderValue
        {
            public ShaderKeyWordValue(Material mat) : base("", ShaderUtil.ShaderPropertyType.Float)
            {
                if (mat.shaderKeywords != null)
                {
                    string[] tmp = mat.shaderKeywords;
                    keywordValue = new string[tmp.Length];

                    for (int i = 0; i < tmp.Length; ++i)
                    {
                        keywordValue[i] = tmp[i];
                    }
                }
                blendMode = GetBlendMode(mat);
            }
            public string[] keywordValue = null;
            public BlendMode blendMode = BlendMode.Opaque;
            public override void SetValue(Material mat)
            {
                mat.shaderKeywords = keywordValue;
                SetupMaterialWithBlendMode(mat, blendMode);
            }
        }

        public static void SetupMaterialWithBlendMode(Material material, BlendMode blendMode, bool resetRenderQueue = true, int renderQueue = -1)
        {
            switch (blendMode)
            {
                case BlendMode.Opaque:
                    material.SetOverrideTag("RenderType", "");
                    if (material.HasProperty("_SrcBlend"))
                        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
                    if (material.HasProperty("_DstBlend"))
                        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.Zero);
                    if (material.HasProperty("_ZWrite"))
                        material.SetInt("_ZWrite", 1);
                    material.DisableKeyword("_ALPHA_TEST");
                    material.DisableKeyword("_ALPHA_PREMULT");
                    if (resetRenderQueue)
                    {
                        if (renderQueue != -1)
                            material.renderQueue = renderQueue;
                        else
                            material.renderQueue = -1;
                    }
                    break;
                case BlendMode.Cutout:
                    material.SetOverrideTag("RenderType", "TransparentCutout");
                    if (material.HasProperty("_SrcBlend"))
                        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
                    if (material.HasProperty("_DstBlend"))
                        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.Zero);
                    if (material.HasProperty("_ZWrite"))
                        material.SetInt("_ZWrite", 1);
                    material.EnableKeyword("_ALPHA_TEST");
                    if (resetRenderQueue)
                    {
                        if (renderQueue != -1)
                            material.renderQueue = renderQueue;
                        else
                            material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.AlphaTest;
                    }
                    break;
                case BlendMode.CutoutTransparent:
                    material.SetOverrideTag("RenderType", "TransparentCutout");
                    if (material.HasProperty("_SrcBlend"))
                        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                    if (material.HasProperty("_DstBlend"))
                        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                    if (material.HasProperty("_ZWrite"))
                        material.SetInt("_ZWrite", 1);
                    material.EnableKeyword("_ALPHA_TEST");
                    if (resetRenderQueue)
                    {
                        if (renderQueue != -1)
                            material.renderQueue = renderQueue;
                        else
                            material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.AlphaTest;
                    }
                    break;
                case BlendMode.Transparent:
                    material.SetOverrideTag("RenderType", "Transparent");
                    if (material.HasProperty("_SrcBlend"))
                        material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                    if (material.HasProperty("_DstBlend"))
                        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                    if (material.HasProperty("_ZWrite"))
                        material.SetInt("_ZWrite", 0);
                    material.DisableKeyword("_ALPHA_TEST");
                    if (resetRenderQueue)
                    {
                        if (renderQueue != -1)
                            material.renderQueue = renderQueue;
                        else
                            material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
                    }
                    break;
            }
        }

        internal static BlendMode GetBlendMode(Material material)
        {
            bool alphaTest = material.IsKeywordEnabled("_ALPHA_TEST");
            bool alphaBlend = material.HasProperty("_DstBlend") &&
                material.GetInt("_DstBlend") == (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha;
            if (!alphaTest && !alphaBlend)
            {
                return BlendMode.Opaque;
            }
            else if (alphaTest && !alphaBlend)
            {
                return BlendMode.Cutout;
            }
            else if (alphaTest && alphaBlend)
            {
                return BlendMode.CutoutTransparent;
            }
            return BlendMode.Transparent;
        }

        internal static void AddMaterialProperty(ref MaterialContext context, Texture tex, int shaderID)
        {
            ShaderTexPropertyValue stpv = new ShaderTexPropertyValue()
            {
                shaderID = shaderID,
                value = tex,
                path = tex.name
            };
            context.textureValue.Add(stpv);
        }

        internal static void AddMaterialProperty(ref MaterialContext context, Vector4 param, int shaderID)
        {
            ShaderPropertyValue stpv = new ShaderPropertyValue()
            {
                shaderID = shaderID,
                value = param,
            };
            context.shaderIDs.Add(stpv);
        }

        internal static bool AddMaterialProperty(ref MaterialContext context, Material material, ShaderProperty sp)
        {
            if (material.HasProperty(sp.shaderProperty))
            {
                if (sp.isTex)
                {
                    Texture tex = material.GetTexture(sp.shaderProperty);
                    if (tex != null)
                    {
                        ShaderTexPropertyValue stpv = new ShaderTexPropertyValue()
                        {
                            shaderID = sp.shaderID,
                            value = tex,
                            path = tex.name
                        };
                        context.textureValue.Add(stpv);
                        return true;
                    }
                    else
                    {
                        Debug.LogErrorFormat("null tex property:{0} mat:{1}", sp.shaderProperty, material.name);
                        return false;
                    }
                }
                else
                {
                    Vector4 param = material.GetVector(sp.shaderProperty);
                    ShaderPropertyValue spv = new ShaderPropertyValue()
                    {
                        shaderID = sp.shaderID,
                        value = param,
                    };
                    context.shaderIDs.Add(spv);
                    return true;
                }
            }
            return false;
        }

        static string[] shaderNameKeys = null;
        static ShaderProperty shareSp = new ShaderProperty();
        internal static bool AddMaterialProperty(
            ref MaterialContext context,
            Material material,
            ShaderValue sv,
            List<ShaderProperty> shaderPropertyKey)
        {
            if (sv.name == "_Cutoff" ||
                sv.name == "_SrcBlend" ||
                sv.name == "_DstBlend" ||
                sv.name == "_ZWrite" ||
                sv.name == "_DebugMode")
                return false;

            ShaderProperty sp = shaderPropertyKey.Find((x) => { return x.shaderProperty == sv.name; });
            if (sp != null)
            {
                return AddMaterialProperty(ref context, material, sp);
            }
            else
            {
                if (shaderNameKeys == null)
                {
                    shaderNameKeys = System.Enum.GetNames(typeof(EShaderKeyID));
                    for (int i = 0; i < shaderNameKeys.Length; ++i)
                    {
                        shaderNameKeys[i] = "_" + shaderNameKeys[i];
                    }
                }
                for (int i = 0; i < shaderNameKeys.Length; ++i)
                {
                    if (shaderNameKeys[i] == sv.name)
                    {
                        shareSp.shaderID = i;
                        shareSp.isTex = sv is ShaderTexValue;
                        shareSp.shaderProperty = sv.name;
                        return AddMaterialProperty(ref context, material, shareSp);
                    }
                }
                Debug.LogErrorFormat("null property:{0} mat:{1}", sv.name, material.name);
                return false;
            }
        }

        static string[] keyWords = new string[]
        {
        "_PBS_FROM_PARAM",
        "_PBS_HALF_FROM_PARAM",
        "_PBS_M_FROM_PARAM ",
        "_PBS_NO_IBL",
        "_OVERLAY",
        "_ETX_EFFECT",
        "_NEED_BOX_PROJECT_REFLECT",
        "_INSTANCE",
        "_PARALLAX_EFFECT",
        "_TERRAIN_LODCULL",
        "_SHADOW_MAP",
        "_SELF_SHADOW_MAP",
        "_ALPHA_FROM_COLOR",
        "_CUSTOM_LIGHTMAP_ON",
        };

        static KeywordFlags[] keyWordFlag = new KeywordFlags[]
        {
        KeywordFlags._PBS_FROM_PARAM,
        KeywordFlags._PBS_HALF_FROM_PARAM,
        KeywordFlags._PBS_M_FROM_PARAM,
        KeywordFlags._PBS_NO_IBL,
        KeywordFlags._OVERLAY,
        KeywordFlags._ETX_EFFECT,
        KeywordFlags._NEED_BOX_PROJECT_REFLECT,
        KeywordFlags._INSTANCE,
        KeywordFlags._PARALLAX_EFFECT,
        KeywordFlags._TERRAIN_LODCULL,
        KeywordFlags._SHADOW_MAP,
        KeywordFlags._SELF_SHADOW_MAP,
        KeywordFlags._ALPHA_FROM_COLOR
        };

        public static void RefeshMat(Material mat, BlendMode blendMode, KeywordFlags keyWord, bool enableLightMap, bool enableShadow)
        {
            if (mat != null)
            {
                mat.shaderKeywords = null;
                for (int i = 0; i < keyWordFlag.Length; ++i)
                {
                    KeywordFlags flag = keyWordFlag[i];
                    if (((uint)(keyWord & flag)) != 0)
                    {
                        mat.EnableKeyword(keyWords[i]);
                    }
                }
                if (enableLightMap)
                    mat.EnableKeyword("_CUSTOM_LIGHTMAP_ON");
                if (enableShadow)
                    mat.EnableKeyword("_SHADOW_MAP");
                SetupMaterialWithBlendMode(mat, blendMode);
            }
        }

    }

}