#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace XEngine
{
    public enum BlendMode
    {
        Opaque,
        Cutout,
        CutoutTransparent,
        Transparent,
    }

    public enum KeywordFlags
    {
        _PBS_FROM_PARAM = 0x00000001,
        _PBS_HALF_FROM_PARAM = 0x00000002,
        _PBS_M_FROM_PARAM = 0x00000004,
        _PBS_NO_IBL = 0x00000100,
        _OVERLAY = 0x00000200,
        _ETX_EFFECT = 0x00000400,
        _NEED_BOX_PROJECT_REFLECT = 0x00000800,
        _INSTANCE = 0x00001000,
        _PARALLAX_EFFECT = 0x00002000,
        _TERRAIN_LODCULL = 0x01000000,
        _SHADOW_MAP = 0x10000000,
        _SELF_SHADOW_MAP = 0x20000000,
        _ALPHA_FROM_COLOR = 0x40000000,
    }

    public enum DebugDisplayType
    {
        Split,
        Full
    }

    [System.Serializable]
    public class ShaderDebugContext
    {
        public int debugMode = 0;
        public DebugDisplayType debugDisplayType = DebugDisplayType.Split;
        public float splitLeft = -1;
        public float splitRight = 1;
        [XEngine.RangeAttribute(-45, 45)]
        public float splitAngle = 0;
        public float splitPos = 0;
        [NonSerialized]
        public bool modeModify = false;
        [NonSerialized]
        public bool typeModify = false;
        [NonSerialized]
        public bool angleModify = false;
        [NonSerialized]
        public bool posModify = false;
        [NonSerialized]
        public int[] shaderID = null;
        public void Refresh()
        {
            if (shaderID != null)
            {
                if (angleModify)
                    CalcSplitLeftRight();
                if (modeModify)
                {
                    Shader.SetGlobalFloat(shaderID[0], (float)debugMode);
                    modeModify = false;
                }
                if (typeModify)
                {
                    Shader.SetGlobalInt(shaderID[1], (int)debugDisplayType);
                    typeModify = false;
                }
                if (angleModify)
                {
                    float k = Mathf.Tan(Mathf.Deg2Rad * (90 - splitAngle));
                    Shader.SetGlobalVector(shaderID[2], new Vector2(k, k < 0 ? -1 : 1));
                    angleModify = false;
                }
                if (posModify)
                {
                    float k = Mathf.Tan(Mathf.Deg2Rad * (90 - splitAngle));
                    float b = -k * splitPos;
                    Shader.SetGlobalFloat(shaderID[3], b);
                    posModify = false;
                }
            }
        }

        private void CalcSplitLeftRight()
        {
            float k = Mathf.Tan(Mathf.Deg2Rad * (90 - splitAngle));
            float b = 1 + k;
            splitLeft = -b / k;
            splitRight = -splitLeft;
        }
    }


    [System.Serializable]
    public class ShaderProperty
    {
        public string shaderProperty;
        public bool isTex;
        public int shaderID;
    }

    public class AssetsConfig : ScriptableObject
    {
        public enum ShaderPropertyType
        {
            CustomFun = 0,
            Tex,
            Color,
            Vector,
            Keyword, //Keyword
            Custom,
            CustomGroup,
            RenderQueue,
        }

        public enum DependencyType
        {
            Or,
            And,
        }

        [System.Serializable]
        public class ShaderPropertyDependency
        {
            public bool isNor = false;
            public DependencyType dependencyType = DependencyType.Or;
            public List<string> dependencyShaderProperty = new List<string>();

            public void Clone(ShaderPropertyDependency src)
            {
                isNor = src.isNor;
                dependencyType = src.dependencyType;
                dependencyShaderProperty.Clear();
                dependencyShaderProperty.AddRange(src.dependencyShaderProperty);
            }
        }

        [System.Serializable]
        public class ShaderCustomProperty
        {
            public bool valid = false;
            public string desc;
            public float defaultValue = 0.0f;
            public float min = 0.0f;
            public float max = 1.0f;
            public string subGroup = "";
            public int indexInGroup = -1;
        }

        [System.Serializable]
        public class ShaderFeature
        {
            public bool hide = false;
            public bool readOnly = false;
            public string name = "empty";
            public string shaderGroupName = "";
            public int indexInGroup = -1;
            public string propertyName = "";
            public ShaderPropertyType type = ShaderPropertyType.Custom;
            public ShaderCustomProperty[] customProperty = new ShaderCustomProperty[4]
            {
                new ShaderCustomProperty(),
                new ShaderCustomProperty(),
                new ShaderCustomProperty(),
                new ShaderCustomProperty(),
            };
            public ShaderPropertyDependency dependencyPropertys = new ShaderPropertyDependency();
        }

        [System.Serializable]
        public class ShaderInfo
        {
            public Shader shader;
            public List<string> shaderFeatures = new List<string>();
        }

        public static string[] shaderDebugNames = null;
        public List<string> ShaderGroupInfo = new List<string>()
        {
            "Default",
            "Base",
            "Pbs",
            "Custom",
            "Rim",
            "Skin",
            "Debug",
            "None",
        };
        public List<ShaderFeature> ShaderFeatures = new List<ShaderFeature>
        {
            new ShaderFeature () { name = "BlendMode", shaderGroupName = "Default", type = ShaderPropertyType.CustomFun, propertyName = "" },
            new ShaderFeature () { name = "Debug", shaderGroupName = "Debug", propertyName = "_DebugMode", type = ShaderPropertyType.CustomFun },
            new ShaderFeature () { name = "BaseTex", shaderGroupName = "Base", propertyName = "_BaseTex", type = ShaderPropertyType.Tex },
            new ShaderFeature () { name = "BaseFromColor", shaderGroupName = "Base", propertyName = "_BASE_FROM_COLOR", type = ShaderPropertyType.Keyword },
            new ShaderFeature () { name = "AlphaFromColor", shaderGroupName = "Base", propertyName = "_ALPHA_FROM_COLOR", type = ShaderPropertyType.Keyword },
            new ShaderFeature () { name = "ColorBlend", shaderGroupName = "Base", propertyName = "_ColorR", type = ShaderPropertyType.Color },
            new ShaderFeature () { name = "PbsFromParam", shaderGroupName = "Pbs", propertyName = "_PBS_FROM_PARAM", type = ShaderPropertyType.Keyword },
            new ShaderFeature () { name = "MagicParam", shaderGroupName = "Pbs", propertyName = "_MagicParam", type = ShaderPropertyType.Custom },
            new ShaderFeature () { name = "PbsNoEnv", shaderGroupName = "Pbs", propertyName = "_PBS_NO_IBL", type = ShaderPropertyType.Keyword },
            new ShaderFeature () { name = "UVST", shaderGroupName = "Scene", propertyName = "_uvST", type = ShaderPropertyType.Vector },
            new ShaderFeature () { readOnly = true, name = "AtlasUVST", shaderGroupName = "Scene", propertyName = "_AtlasUVST", type = ShaderPropertyType.Vector },
            new ShaderFeature () { name = "MainColor", shaderGroupName = "Scene", propertyName = "_MainColor", type = ShaderPropertyType.Color },
            new ShaderFeature () { name = "Emission", shaderGroupName = "Custom", propertyName = "_EMISSION", type = ShaderPropertyType.Keyword },
            new ShaderFeature () { name = "Skin", shaderGroupName = "Custom", propertyName = "_SkinSpecularScatter", type = ShaderPropertyType.Custom },
            new ShaderFeature ()
            {
            name = "Color", shaderGroupName = "Base", propertyName = "_Color", type = ShaderPropertyType.Color,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "BaseFromColor", "AlphaFromColor" } }
            },
            new ShaderFeature ()
            {
            name = "PbsTex", shaderGroupName = "Pbs", propertyName = "_PBSTex", type = ShaderPropertyType.Tex,
            dependencyPropertys = new ShaderPropertyDependency () { isNor = true, dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "PbsFromColor" } }
            },
            new ShaderFeature ()
            {
            name = "PbsHalfFromParam", shaderGroupName = "Pbs", propertyName = "_PBS_HALF_FROM_PARAM", type = ShaderPropertyType.Keyword,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "PbsFromColor" } }
            },
            new ShaderFeature ()
            {
            name = "PbsParam", shaderGroupName = "Pbs", propertyName = "_PbsParam", type = ShaderPropertyType.Custom,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "PbsFromColor", "PbsHalfFromColor" } }
            },
            new ShaderFeature ()
            {
            name = "EmissionTex", shaderGroupName = "Custom", propertyName = "_EffectTex", type = ShaderPropertyType.Tex,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "Emission" } }
            },
            new ShaderFeature ()
            {
            name = "EmissionColor", shaderGroupName = "Custom", propertyName = "_EmissionColor", type = ShaderPropertyType.Color,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "Emission" } }
            },
            new ShaderFeature () { name = "Overlay", shaderGroupName = "Custom", propertyName = "_OVERLAY", type = ShaderPropertyType.Keyword },
            new ShaderFeature ()
            {
            name = "OverLayParam", shaderGroupName = "Custom", propertyName = "_OverLayParam", type = ShaderPropertyType.Vector,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "Overlay" } }
            },
            new ShaderFeature ()
            {
            name = "OverlayColor", shaderGroupName = "Custom", propertyName = "_OverlayColor", type = ShaderPropertyType.Color,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "Overlay" } }
            },
        };
        public List<ShaderInfo> ShaderInfos = new List<ShaderInfo>();
        private static AssetsConfig g_AssetsConfig;
        public static AssetsConfig GlobalAssetsConfig
        {
            get
            {
                if (g_AssetsConfig == null)
                {
                    g_AssetsConfig = AssetDatabase.LoadAssetAtPath<AssetsConfig>("Assets/Engine/Editor/EditorResources/AssetsConfig.asset");
                }
                if (g_AssetsConfig == null)
                {
                    g_AssetsConfig = ScriptableObject.CreateInstance<AssetsConfig>();
                }
                return g_AssetsConfig;
            }
        }

        public static Dictionary<string, ShaderFeature> GetShaderFeatureList()
        {
            Dictionary<string, ShaderFeature> shaderFeatures = new Dictionary<string, ShaderFeature>();

            for (int i = 0; i < GlobalAssetsConfig.ShaderFeatures.Count; ++i)
            {
                ShaderFeature sf = GlobalAssetsConfig.ShaderFeatures[i];
                shaderFeatures[sf.name] = sf;
            }
            return shaderFeatures;
        }
        public TextAsset debugFile;
        public static void RefreshShaderDebugNames(bool force = false)
        {
            if (shaderDebugNames == null || force)
            {
                if (GlobalAssetsConfig.debugFile != null)
                {
                    string path = AssetDatabase.GetAssetPath(GlobalAssetsConfig.debugFile);
                    bool parse = false;
                    using (FileStream fs = new FileStream(path, FileMode.Open))
                    {
                        List<string> debugTypeStr = new List<string>();
                        StreamReader sr = new StreamReader(fs);
                        while (!sr.EndOfStream)
                        {
                            string line = sr.ReadLine();
                            if (parse)
                            {
                                if (line.StartsWith("//DEBUG_END"))
                                {
                                    parse = false;
                                }
                                else
                                {
                                    string[] str = line.Split(' ');
                                    if (str.Length >= 3 && str[0] == "#define")
                                    {
                                        string debugStr = str[1];
                                        debugStr = debugStr.Replace("Debug_", "");
                                        debugTypeStr.Add(debugStr);
                                    }
                                }
                            }
                            else
                            {
                                if (line.StartsWith("//DEBUG_START"))
                                {
                                    parse = true;
                                }
                            }
                        }
                        shaderDebugNames = debugTypeStr.ToArray();
                    }
                }
            }
        }
    }
}

#endif