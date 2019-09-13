#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Reflection;
using System.IO;
using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;

namespace CFEngine
{
    public enum BlendMode
    {
        Opaque,
        Cutout,
        CutoutTransparent,
        Transparent, // Physically plausible transparency mode, implemented as alpha pre-multiply
    }
    public enum SpriteSize
    {
        E1x1,
        E2x2,
        E4x4,
        E8x8,
        E16x16,
        E32x32,
        E64x64,
        E128x128,
        E256x256,
        E512x512,
        E1024x1024,
        E2048x2048,
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
        _SPLAT1 = 0x00100000,
        _SPLAT2 = 0x00200000,
        _SPLAT3 = 0x00400000,
        _SPLAT4 = 0x00800000,
        _TERRAIN_LODCULL = 0x01000000,
        _SHADOW_MAP = 0x10000000,
        _SELF_SHADOW_MAP = 0x20000000,
        _ALPHA_FROM_COLOR = 0x40000000,
    }

    public enum ELightMapMode
    {
        None,
        LightmapMat,
        LightmapKeyWord
    }

    public enum EBlendType
    {
        None = 0x0000,
        Opaque = 0x0001,
        Cutout = 0x0002,
        CutoutTransparent = 0x0004,
        Transparent = 0x0008,
    }
    public interface IRectSelect
    {
        Rect SelectRect { get; set; }
        string Name { get; set; }
    }
    public enum PPDebugMode
    {
        None,
        Depth,
        Lut,
        Bloom,
        GodRay,
        // DofSetup,
        // DofFlatten0,
        // DofFlatten1,
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
        [CFRange(-45, 45)]
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
        public void Reset()
        {
            debugMode = 0;
            splitAngle = 0;
            splitPos = 0;
            modeModify = true;
            typeModify = true;
            angleModify = true;
            posModify = true;
        }
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
    public struct SceneLoadInfo
    {
        public string name;
        public int count;
    }

    public enum ESceneMaterial : int
    {
        SceneCommon = 0,
        SceneCommonPbs,
        SceneCommonCutout,
        SceneCommonPbsCutout,

        SceneOverlay,
        SceneOverlayPbs,
        SceneOverlayCutout,
        SceneOverlayPbsCutout,

        SceneEmission,
        SceneEmissionPbs,
        SceneEmissionCutout,
        SceneEmissionPbsCutout,

        SceneBoxReflect,
        SceneBoxReflectPbs,

        SceneTree,

        TerrainChunk0,
        TerrainChunk1,
        TerrainChunk2,
        TerrainChunk3,

        SceneMassive,
        SceneMassiveTransparent,

        SceneTransparent,
        SceneMetallic,
        SceneMetallicCutout,
        ScenePOM,
        LavaSat,
        SceneWater,
        IceCave,
        Lava,
        SceneCommonPbs2,
        SceneCommonPbsCutout2,
        Num,
        CustomMat = 0xFF,
    }
    public enum TexFilterType
    {
        Or,
        And,
        Nor,
    }
    public enum TexFlag
    {
        None,
        IgnoreImport = 0x00000001
    }

    [System.Serializable]
    public class TexCompressFilter
    {
        public TexFilterType type = TexFilterType.Or;
        // public bool isNor = false;
        public string str;
    }

    [System.Serializable]
    public class TexImportSetting
    {
        public bool folder = false;
        public SpriteSize maxTextureSize = SpriteSize.E1024x1024;
        public TextureImporterFormat format = TextureImporterFormat.RGB24;
        public TextureImporterFormat alphaFormat = TextureImporterFormat.RGBA32;      
    }

    [System.Serializable]
    public class TexCompressConfig
    {
        public bool folder = false;       
        public bool vaild = true;
        public string name = "";
        public int priority = 0;
        public List<TexCompressFilter> compressFilters = new List<TexCompressFilter>();
        public TextureImporterType importType = TextureImporterType.Default;
        public TextureImporterShape importShape = TextureImporterShape.Texture2D;
        public bool sRGB = true;
        public bool mipMap = false;        
        public FilterMode filterMode = FilterMode.Bilinear;  
        public TextureWrapMode wrapMode = TextureWrapMode.Repeat;  
        public int anisoLevel = -1;
        public TexImportSetting iosSetting = new TexImportSetting()
        {
            format = TextureImporterFormat.ASTC_RGB_5x5,
            alphaFormat = TextureImporterFormat.ASTC_RGBA_5x5
        };
        public TexImportSetting androidSetting = new TexImportSetting()
        {
            format = TextureImporterFormat.ETC2_RGB4,
            alphaFormat = TextureImporterFormat.ETC2_RGBA8Crunched
        };
        public TexImportSetting standaloneSetting = new TexImportSetting()
        {
            format = TextureImporterFormat.DXT1,
            alphaFormat = TextureImporterFormat.DXT5
        };

        public override string ToString()
        {
            return name;
        }

    }

    [System.Serializable]
    public class ShaderProperty
    {
        public bool folder;
        public string shaderProperty;
        public bool isTex;
        public int shaderID;
    }

    public enum FindPropertyType
    {
        None,
        CommonProperty,
        FindAll
    }

    [System.Serializable]
    public class MatShaderType
    {
        public bool folder;
        public string name;
        public Shader shader;
        public string macro;
        public List<ShaderProperty> shaderPropertys = new List<ShaderProperty> ();
        public ESceneMaterial matOffset = ESceneMaterial.SceneCommon;
        public FindPropertyType findPropertyType = FindPropertyType.None;
        public bool hasPbs;
        public uint pbsOffset;
        public bool hasCutout;
        public bool hasTransparent;
        public bool hasTransparentCout;
        public uint renderTypeOffset;
        public uint matFlag;
        public void SetFlag (EMatFlag f, bool add)
        {
            if (add)
            {
                matFlag |= (uint) f;
            }
            else
            {
                matFlag &= ~((uint) f);
            }
        }

        public bool HasFlag (EMatFlag f)
        {
            return (matFlag & (uint) f) != 0;
        }

        public void Clone (MatShaderType src)
        {
            name = src.name;
            shader = src.shader;
            macro = src.macro;
            shaderPropertys.Clear ();
            for (int i = 0; i < src.shaderPropertys.Count; ++i)
            {
                var sp = src.shaderPropertys[i];
                shaderPropertys.Add (new ShaderProperty ()
                {
                    shaderProperty = sp.shaderProperty,
                        isTex = sp.isTex,
                        shaderID = sp.shaderID,
                });
            }

            matOffset = src.matOffset;
            findPropertyType = src.findPropertyType;
            pbsOffset = src.pbsOffset;
            renderTypeOffset = src.renderTypeOffset;
            hasCutout = src.hasCutout;
            hasTransparent = src.hasTransparent;
            hasTransparentCout = src.hasTransparentCout;
            renderTypeOffset = src.renderTypeOffset;
            matFlag = src.matFlag;

        }
    }

    public class AssetsConfig : ScriptableObject
    {
        public enum EditorSceneObjectType
        {
            EditorScene,
            Light,
            Collider,
            Enverinment,
            Effect,
            DynamicObject,
            Prefab,
            StaticPrefab,
            Instance,
            // ExceptBakeMesh,
            MeshTerrain,
            UnityTerrain,
            Num,
        }
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
            public List<string> dependencyShaderProperty = new List<string> ();

            public void Clone (ShaderPropertyDependency src)
            {
                isNor = src.isNor;
                dependencyType = src.dependencyType;
                dependencyShaderProperty.Clear ();
                dependencyShaderProperty.AddRange (src.dependencyShaderProperty);
            }
        }
       [System.Serializable]
        public class ShaderCustomProperty
        {
            public bool folder = false;
            public bool valid = false;
            public string desc;
            public float defaultValue = 0.0f;
            public float min = 0.0f;
            public float max = 1.0f;
            public string subGroup = "";
            [NonSerialized]
            public int shaderGroupIndex = 0;
            public int indexInGroup = -1;
        }

        [System.Serializable]
        public class ShaderFeature : IRectSelect
        {
            public bool hide = false;
            public bool readOnly = false;
            public string name = "empty";
            public string shaderGroupName = "";
            [NonSerialized]
            public int shaderGroupIndex = 0;
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
            public ShaderPropertyDependency dependencyPropertys = new ShaderPropertyDependency ();
            public bool folder = false;

            [System.NonSerialized]
            private Rect rect;
            public Rect SelectRect { get { return rect; } set { rect = value; } }
            public string Name { get { return name; } set { name = value; } }

            public void Clone (ShaderFeature src)
            {

                hide = src.hide;
                readOnly = src.readOnly;
                // name = src.name;
                shaderGroupName = src.shaderGroupName;
                propertyName = src.propertyName;
                type = src.type;
                dependencyPropertys.Clone (src.dependencyPropertys);

            }
        }

        [System.Serializable]
        public class ShaderInfo
        {
            public bool folder = false;
            public Shader shader;
            public List<string> shaderFeatures = new List<string> ();
            public bool customGroupDraw = false;
        }

        [System.Serializable]
        public class DummyMaterialInfo : IRectSelect
        {
            public bool folder = false;
            public string name = "";
            [System.NonSerialized]
            public int enumIndex = -1;
            public Shader shader;
            public Material mat;
            public Material mat1;
            public string ext1;
            public Material mat2;
            public string ext2;
            public Material mat3;
            public string ext3;
            public ELightMapMode lightmapMode = ELightMapMode.None;
            public EBlendType blendType = EBlendType.None;
            public bool shadowMat;
            public uint flag;

            [System.NonSerialized]
            private Rect rect;
            public Rect SelectRect { get { return rect; } set { rect = value; } }
            public string Name { get { return name; } set { name = value; } }
        }
        public string ResourcePath = "Assets/BundleRes";
        public string SceneLibPath = "Assets/Scenes/Scenelib";
        public string EditorResourcePath = "Assets/Editor/EditorResources";
        public string EngineResPath = "Assets/Engine/Editor/EditorResources";
        public string Bandpose_Str = "Bandpose";
        public string Fbx_Ext = ".fbx";
        public string Creature_Path = "Assets/Creatures";
        public string SharedMaterialPrefix = "SharedMaterials_";
        public string Creature_Bandpose_Format_Path = "{0}/{1}/Bandpose";
        public string Creature_Material_Format_Path = "{0}/{1}/Bandpose/Materials_{2}";
        public string Creature_SharedMaterial_Format_Path = "{0}/{1}/Bandpose/{2}/Materials";
        public string Creature_Material_Format_Folder = "Materials_{0}";
        public string Creature_Aniamtion_Format_Path = "{0}/{1}/Animation/";
        public string ResourceAnimationPath = "Assets/BundleRes/Animation";
        public string ResourceAnimation = "Animation";
        public string BaseTex_Format_Path = "{0}/{1}_base.tga";
        public string PbsTex_Format_Path = "{0}/{1}_pbs.tga";
        public string BaseTex_Suffix = "_base";
        public string PbsTex_Suffix = "_pbs";
        public string DummyMatFolder = "MatShader";
        public string Table_Path = "Assets/Table/";
        public string Table_Bytes_Path = "Assets/BundleRes/Table/";
        public string TerrainDirStr = "Terrain";
        public string ConfigStr = "_Config";
        public string SceneConfigStr = "_SceneConfig";

        public string SceneChunkStr = "Chunk_";
        public string TerrainBlockStr = "Block_";
        public string TerrainBlendTexStr = "_Blend";
        public string TerrainBlendRTStr = "_BlendRT";
        public string TerrainSplatTexStr = "_Splat";
        public string TerrainSplatConfigStr = "_SplatConfig";
        public string TerrainSplatConfigRefStr = "_SplatConfigRef";

        public string AtlasDirStr = "Assets/Scenes/modlelib/atlas";
        public string AtlasConfigStr = "_AtlasConfig";
        public string UIImagePath = "UI/UISource/Background";
        public string SceneCombineObjName = "GameObject";

        public string SpriteAtlasExt = ".spriteatlas";

        public string ReadableMeshSuffix = "_readable.asset";

        public string EditorSceneRes = "/EditorSceneRes";

        public string[] ResourceShaders = new string[]
        {
            "Assets/Engine/Shaders/PBS"
        };

        public string[] MaterialShaderMap = new string[]
        {
            "_face",
            "Custom/PBS/Skin",
            "_hair",
            "Custom/Hair/HairTest01",
            "_lower|_upper|_weapon",
            "Custom/PBS/Role",
            "",
            "Custom/PBS/Entity"
        };

        public static string[] EditorGoPath = new string[(int) EditorSceneObjectType.Num]
        {
            "EditorScene",
            "Light",
            "Collider",
            "Enverinment",
            "Effects",
            "DynamicObjects",
            "Prefabs",
            "StaticPrefabs",
            "Instance",
            "MeshTerrain",
            "UnityTerrain",
        };

        public string[] MehsAutoExportType = new string[]
        {
            "Assets/Scenes",
        };

        public LightmapParameters[] LightmapParam;
        public Material TerrainEditMat;
        public Material TerrainMergeMat;
        public Material TerrainMeshMat;
        public Material TerrainBakeBaseMap;
        public Material PreviewMat;
        public Material[] TerrainPreviewMat;
        public Material Outline;

        public Material ShadowCaster;
        public Material CombineNormal;

        public Material PreveiwTransparent;

        public static string[] shaderDebugNames = null;
        public static string[] shaderPPDebugNames = Enum.GetNames(typeof(PPDebugMode));
        [HideInInspector]
        public bool commonFolder = false;
        public Shader ScenePreview;
        // public Shader TerrainPreview;
        public Shader TextureBake;
        [HideInInspector]
        public bool texCompressConfigFolder = false;
        public List<TexCompressConfig> texCompressConfig = new List<TexCompressConfig>();
        [HideInInspector]
        public bool groupFolder = false;
        public List<string> ShaderGroupInfo = new List<string> ()
        {
            "Default",
            "Base",
            "Pbs",
            "Scene",
            "Custom",
            "Rim",
            "Skin",
            "Debug",
            "None",
        };
        [HideInInspector]
        public bool shaderFeatureFolder = false;
        public List<ShaderFeature> ShaderFeatures = new List<ShaderFeature>
        {
            new ShaderFeature () { name = "BlendMode", shaderGroupName = "Default", type = ShaderPropertyType.CustomFun, propertyName = "" },
            new ShaderFeature () { name = "Debug", shaderGroupName = "Debug", propertyName = "_DebugMode", type = ShaderPropertyType.CustomFun },

            new ShaderFeature () { name = "BaseTex", shaderGroupName = "Base", propertyName = "_BaseTex", type = ShaderPropertyType.Tex },
            new ShaderFeature () { name = "BaseFromColor", shaderGroupName = "Base", propertyName = "_BASE_FROM_COLOR", type = ShaderPropertyType.Keyword },
            new ShaderFeature () { name = "AlphaFromColor", shaderGroupName = "Base", propertyName = "_ALPHA_FROM_COLOR", type = ShaderPropertyType.Keyword },
            new ShaderFeature ()
            {
            name = "Color", shaderGroupName = "Base", propertyName = "_Color", type = ShaderPropertyType.Color,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "BaseFromColor", "AlphaFromColor" } }
            },
            new ShaderFeature () { name = "ColorBlend", shaderGroupName = "Base", propertyName = "_ColorR", type = ShaderPropertyType.Color },

            new ShaderFeature () { name = "PbsFromParam", shaderGroupName = "Pbs", propertyName = "_PBS_FROM_PARAM", type = ShaderPropertyType.Keyword },
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
            new ShaderFeature () { name = "MagicParam", shaderGroupName = "Pbs", propertyName = "_MagicParam", type = ShaderPropertyType.Custom },
            new ShaderFeature () { name = "PbsNoEnv", shaderGroupName = "Pbs", propertyName = "_PBS_NO_IBL", type = ShaderPropertyType.Keyword },

            new ShaderFeature () { hide = true, name = "OneLayer", shaderGroupName = "Scene", propertyName = "_SPLAT1", type = ShaderPropertyType.Keyword },
            new ShaderFeature () { hide = true, name = "TwoLayer", shaderGroupName = "Scene", propertyName = "_SPLAT2", type = ShaderPropertyType.Keyword },
            new ShaderFeature () { hide = true, name = "ThreeLayer", shaderGroupName = "Scene", propertyName = "_SPLAT3", type = ShaderPropertyType.Keyword },
            new ShaderFeature () { hide = true, name = "FourLayer", shaderGroupName = "Scene", propertyName = "_SPLAT4", type = ShaderPropertyType.Keyword },
            new ShaderFeature ()
            {
            name = "BlendTex", shaderGroupName = "Scene", propertyName = "_BlendTex", type = ShaderPropertyType.Tex,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "_SPLAT2", "_SPLAT3", "_SPLAT4" } }
            },
            new ShaderFeature ()
            {
            name = "SplatTex0", shaderGroupName = "Scene", propertyName = "_BaseTex0", type = ShaderPropertyType.Tex,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "_SPLAT1", "_SPLAT2", "_SPLAT3", "_SPLAT4" } }
            },
            new ShaderFeature ()
            {
            name = "SplatTex1", shaderGroupName = "Scene", propertyName = "_BaseTex1", type = ShaderPropertyType.Tex,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "_SPLAT2", "_SPLAT3", "_SPLAT4" } }
            },
            new ShaderFeature ()
            {
            name = "SplatTex2", shaderGroupName = "Scene", propertyName = "_BaseTex2", type = ShaderPropertyType.Tex,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "_SPLAT3", "_SPLAT4" } }
            },
            new ShaderFeature ()
            {
            name = "SplatTex3", shaderGroupName = "Scene", propertyName = "_BaseTex3", type = ShaderPropertyType.Tex,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "_SPLAT4" } }
            },
            new ShaderFeature () { name = "UVST", shaderGroupName = "Scene", propertyName = "_uvST", type = ShaderPropertyType.Vector },
            new ShaderFeature () { readOnly = true, name = "AtlasUVST", shaderGroupName = "Scene", propertyName = "_AtlasUVST", type = ShaderPropertyType.Vector },
            new ShaderFeature () { name = "MainColor", shaderGroupName = "Scene", propertyName = "_MainColor", type = ShaderPropertyType.Color },

            new ShaderFeature () { name = "Emission", shaderGroupName = "Custom", propertyName = "_EMISSION", type = ShaderPropertyType.Keyword },
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
            name = "OverlayTex", shaderGroupName = "Custom", propertyName = "_EffectTex", type = ShaderPropertyType.Tex,
            dependencyPropertys = new ShaderPropertyDependency () { dependencyType = DependencyType.Or, dependencyShaderProperty = new List<string> () { "Overlay" } }
            },
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
            // new ShaderFeature () { name = "Rim", shaderGroupName = "Custom", propertyName = "_Rim", type = ShaderPropertyType.Custom },
            new ShaderFeature () { name = "Skin", shaderGroupName = "Custom", propertyName = "_SkinSpecularScatter", type = ShaderPropertyType.Custom },

        };
        [HideInInspector]
        public bool shaderInfoFolder = false;

        public List<ShaderInfo> ShaderInfos = new List<ShaderInfo> ();
        [HideInInspector]
        public bool dummyMaterialsFolder = false;
        [HideInInspector]
        public bool sceneDummyMaterialsFolder = false;
        [HideInInspector]
        public List<DummyMaterialInfo> sceneDummyMaterials = new List<DummyMaterialInfo> (64);

        [HideInInspector]
        public bool effectMaterialsFolder = false;
        [HideInInspector]
        public bool roleMatFolder = false;
        [HideInInspector]
        public bool specialMatFolder = false;
        public List<DummyMaterialInfo> roleMaterials = new List<DummyMaterialInfo> ();

        [HideInInspector]
        public List<DummyMaterialInfo> effectMaterials = new List<DummyMaterialInfo> ();
        [HideInInspector]
        public bool sceneMatInfoFolder;

        [HideInInspector]
        public List<DummyMaterialInfo> customMaterials = new List<DummyMaterialInfo> ();
        [HideInInspector]
        public bool customMatInfoFolder;

        [HideInInspector]
        public List<ShaderProperty> commonShaderProperty = new List<ShaderProperty> ()
        {
            //tex
            new ShaderProperty () { shaderProperty = "_MainTex", isTex = true, shaderID = (int) EShaderKeyID.BaseTex },
            new ShaderProperty () { shaderProperty = "_MainColor", shaderID = (int) EShaderKeyID.MainColor },
            new ShaderProperty () { shaderProperty = "_MagicParam", shaderID = (int) EShaderKeyID.MagicParam },
            new ShaderProperty () { shaderProperty = "_uvST", shaderID = (int) EShaderKeyID.uvST },
        };

        [HideInInspector]
        public List<ShaderProperty> shaderPropertyKey = new List<ShaderProperty> ()
        {
            //tex
            new ShaderProperty () { shaderProperty = "_BaseTex", isTex = true, shaderID = (int) EShaderKeyID.BaseTex },
            new ShaderProperty () { shaderProperty = "_PBSTex", isTex = true, shaderID = (int) EShaderKeyID.PBSTex },
        };

        [HideInInspector]
        public List<MatShaderType> matShaderType = new List<MatShaderType> ()
        {
            new MatShaderType ()
            {
            name = "Overlay", macro = "_OVERLAY", matOffset = ESceneMaterial.SceneOverlay, hasPbs = true, hasCutout = true, findPropertyType = FindPropertyType.CommonProperty,
            shaderPropertys = new List<ShaderProperty> ()
            {
            new ShaderProperty () { shaderProperty = "_OverLayParam", shaderID = (int) EShaderKeyID.OverLayParam },
            new ShaderProperty () { shaderProperty = "_OverlayColor", shaderID = (int) EShaderKeyID.OverlayColor },
            new ShaderProperty () { shaderProperty = "_EffectTex", isTex = true, shaderID = (int) EShaderKeyID.EffectTex },
            }
            },
            new MatShaderType ()
            {
            name = "Emission", macro = "_ETX_EFFECT", matOffset = ESceneMaterial.SceneEmission, hasPbs = true, hasCutout = true, findPropertyType = FindPropertyType.CommonProperty,
            shaderPropertys = new List<ShaderProperty> ()
            {
            new ShaderProperty () { shaderProperty = "_EmissionAOColor", shaderID = (int) EShaderKeyID.EmissionAOColor },
            new ShaderProperty () { shaderProperty = "_EffectTex", isTex = true, shaderID = (int) EShaderKeyID.EffectTex },
            }
            },
            new MatShaderType ()
            {
            name = "BoxReflect", macro = "_NEED_BOX_PROJECT_REFLECT", matOffset = ESceneMaterial.SceneBoxReflect, hasPbs = true, findPropertyType = FindPropertyType.CommonProperty,
            shaderPropertys = new List<ShaderProperty> ()
            {
            new ShaderProperty () { shaderProperty = "_BoxCenter", shaderID = (int) EShaderKeyID.ReflBoxCenter },
            new ShaderProperty () { shaderProperty = "_BoxSize", shaderID = (int) EShaderKeyID.ReflBoxSize },
            new ShaderProperty () { shaderProperty = "_EnvReflectTex", isTex = true, shaderID = (int) EShaderKeyID.EnvReflectTex },
            }
            },
            new MatShaderType ()
            {
            name = "Parallax", macro = "_PARALLAX_EFFECT", matOffset = ESceneMaterial.ScenePOM, findPropertyType = FindPropertyType.CommonProperty,
            shaderPropertys = new List<ShaderProperty> ()
            {
            new ShaderProperty () { shaderProperty = "_Param0", shaderID = (int) EShaderKeyID.Param0 },
            new ShaderProperty () { shaderProperty = "_EffectTex", isTex = true, shaderID = (int) EShaderKeyID.EffectTex },
            }
            },
            new MatShaderType () { name = "Grass", matOffset = ESceneMaterial.SceneMassive, hasTransparentCout = true },
            new MatShaderType () { name = "LavaSat", matOffset = ESceneMaterial.LavaSat },
            new MatShaderType () { name = "SceneWater", matOffset = ESceneMaterial.SceneWater },
            new MatShaderType () { name = "IceCave", matOffset = ESceneMaterial.IceCave },
            // new MatShaderType () { name = "Lava", matOffset = ESceneMaterial.Lava},
        };

        private static AssetsConfig g_AssetsConfig;
        public static AssetsConfig GlobalAssetsConfig
        {
            get
            {
                if (g_AssetsConfig == null)
                {
                    g_AssetsConfig = AssetDatabase.LoadAssetAtPath<AssetsConfig> ("Assets/Engine/Editor/EditorResources/AssetsConfig.asset");
                }
                if (g_AssetsConfig == null)
                {
                    g_AssetsConfig = ScriptableObject.CreateInstance<AssetsConfig> ();
                }
                return g_AssetsConfig;
            }
        }

        public static Dictionary<string, ShaderFeature> GetShaderFeatureList ()
        {
            Dictionary<string, ShaderFeature> shaderFeatures = new Dictionary<string, ShaderFeature> ();

            for (int i = 0; i < GlobalAssetsConfig.ShaderFeatures.Count; ++i)
            {
                ShaderFeature sf = GlobalAssetsConfig.ShaderFeatures[i];
                shaderFeatures[sf.name] = sf;
            }
            return shaderFeatures;
        }

        public static void GetGroupedShaderFeatureList (Dictionary<string, List<ShaderFeature>> groupedShaderFeatures)
        {
            groupedShaderFeatures.Clear ();
            for (int i = 0; i < GlobalAssetsConfig.ShaderFeatures.Count; ++i)
            {
                ShaderFeature sf = GlobalAssetsConfig.ShaderFeatures[i];
                List<ShaderFeature> sfList;
                if (!groupedShaderFeatures.TryGetValue (sf.shaderGroupName, out sfList))
                {
                    sfList = new List<ShaderFeature> ();
                    groupedShaderFeatures[sf.shaderGroupName] = sfList;
                }
                sfList.Add (sf);
            }
        }
        public TextAsset debugFile;
        public static void RefreshShaderDebugNames(bool force = false)
        {
            if (shaderDebugNames == null || force)
            {
                try
                {
                    if (AssetsConfig.GlobalAssetsConfig.debugFile != null)
                    {
                        string path = AssetDatabase.GetAssetPath(AssetsConfig.GlobalAssetsConfig.debugFile);
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
                catch (Exception)
                {

                }
            }

        }
    }
}
#endif