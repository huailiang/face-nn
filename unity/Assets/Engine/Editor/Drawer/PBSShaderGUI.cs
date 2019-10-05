using System;
using System.Collections.Generic;
using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;

namespace XEngine.Editor
{
    internal class PBSShaderGUI : ShaderGUI
    {
        public class ShaderPropertyInstance
        {
            public AssetsConfig.ShaderFeature shaderFeature;
            public MaterialProperty property;
            public AssetsConfig.ShaderCustomProperty scp = null;
            public int customIndex = 0;
        }

        public class ShaderGroupInstance
        {
            public string name;
            public List<ShaderPropertyInstance> spiList = new List<ShaderPropertyInstance>();
            public bool hasCustomGroup = false;
        }

        public struct DrawPropertyContext
        {
            public DrawFun fun;
            public Material material;
            public MaterialEditor materialEditor;
            public HashSet<string> hasDepency;
            public ShaderGroupInstance sgi;
            public ShaderPropertyInstance spi;
            public DebugData dd;
        }

        private static class Styles
        {
            public static string renderingMode = "Rendering Mode";
            public static string rimText = "Rim";
            public static string skinText = "Skin";
            public static string debugText = "Debug";
            public static string debugMode = "Debug Mode";
            public static readonly string[] blendNames = Enum.GetNames(typeof(BlendMode));
        }

        static Color[] mipMapColors = new Color[]
        {
            Color.black, //mipmap 0 black
            Color.red, //mipmap 1 red
            new Color (1.0f, 0.5f, 0.0f, 1.0f), //mipmap 2 orange
            Color.yellow, //mipmap 3 yellow
            Color.green, //mipmap 4 green
            Color.cyan, //mipmap 5 cyan
            Color.blue, //mipmap 6 blue
            Color.magenta, //mipmap 7 magenta
            Color.gray, //mipmap 8 gray
            Color.white, //mipmap 9 white
        };

        public delegate void DrawFun(ref DrawPropertyContext context);
        Dictionary<string, DrawFun> customFunc = new Dictionary<string, DrawFun>();
        static DrawFun[] drawPropertyFun = new DrawFun[]
        {
            null,
            DrawTexFun,
            DrawColorFun,
            DrawVectorFun,
            DrawKeyWordFun,
            DrawCustomFun,
            DrawCustomGroupFun,
            DrawRenderQueueFun
        };
        static DrawFun drawGroupBegin = DrawGroupBegin;
        static DrawFun drawGroupEnd = DrawGroupEnd;
        List<ShaderGroupInstance> shaderPropertyGroups = null;
        HashSet<string> hasDepency = new HashSet<string>();
        protected MaterialEditor m_MaterialEditor;
        MaterialProperty baseColorMp;
        protected Material m_Material;
        DebugData dd;
        bool m_FirstTimeApply = true;
        List<DrawPropertyContext> drawList = new List<DrawPropertyContext>();
        protected virtual void FindProperties(MaterialProperty[] props)
        {
            baseColorMp = null;
            shaderPropertyGroups = new List<ShaderGroupInstance>();
            hasDepency.Clear();
            for (int i = 0; i < AssetsConfig.GlobalAssetsConfig.ShaderGroupInfo.Count; ++i)
            {
                string groupName = AssetsConfig.GlobalAssetsConfig.ShaderGroupInfo[i];
                shaderPropertyGroups.Add(new ShaderGroupInstance() { name = groupName });
            }

            Shader shader = m_Material.shader;
            var shaderFeatureMap = AssetsConfig.GetShaderFeatureList();

            for (int i = 0; i < AssetsConfig.GlobalAssetsConfig.ShaderInfos.Count; ++i)
            {
                var si = AssetsConfig.GlobalAssetsConfig.ShaderInfos[i];
                if (si.shader != null && si.shader.name == shader.name)
                {
                    for (int j = 0; j < si.shaderFeatures.Count; ++j)
                    {
                        string featurename = si.shaderFeatures[j];
                        AssetsConfig.ShaderFeature sf;
                        if (shaderFeatureMap.TryGetValue(featurename, out sf))
                        {
                            MaterialProperty mp = null;
                            if (!string.IsNullOrEmpty(sf.propertyName))
                            {
                                mp = FindProperty(sf.propertyName, props, false);
                            }
                            if (sf.type == AssetsConfig.ShaderPropertyType.CustomGroup)
                            {
                                for (int k = 0; k < sf.customProperty.Length; ++k)
                                {
                                    var cp = sf.customProperty[k];
                                    if (cp.valid && !string.IsNullOrEmpty(cp.desc))
                                    {
                                        ShaderGroupInstance sgi = null;
                                        if (string.IsNullOrEmpty(cp.subGroup))
                                        {
                                            sgi = shaderPropertyGroups[shaderPropertyGroups.Count - 1];
                                        }
                                        else
                                        {
                                            sgi = shaderPropertyGroups.Find((param) => { return param.name == cp.subGroup; });
                                        }

                                        if (sgi == null)
                                        {
                                            sgi = new ShaderGroupInstance() { name = cp.subGroup };
                                            sgi.hasCustomGroup = true;
                                            shaderPropertyGroups.Add(sgi);
                                        }
                                        ShaderPropertyInstance spi = new ShaderPropertyInstance();
                                        spi.shaderFeature = sf;
                                        spi.property = mp;
                                        spi.scp = cp;
                                        spi.customIndex = k;
                                        sgi.spiList.Add(spi);
                                    }
                                }
                            }
                            else
                            {
                                ShaderPropertyInstance spi = new ShaderPropertyInstance();
                                spi.shaderFeature = sf;
                                spi.property = mp;
                                ShaderGroupInstance sgi = shaderPropertyGroups.Find((param) => { return param.name == sf.shaderGroupName; });
                                if (sgi == null)
                                {
                                    sgi = shaderPropertyGroups[shaderPropertyGroups.Count - 1];
                                }
                                sgi.spiList.Add(spi);
                                if (sf.name == "Color")
                                {
                                    baseColorMp = spi.property;
                                }
                            }

                            if (sf.type == AssetsConfig.ShaderPropertyType.Keyword &&
                                !hasDepency.Contains(sf.name) &&
                                m_Material.IsKeywordEnabled(sf.propertyName))
                            {
                                hasDepency.Add(sf.name);
                            }
                        }
                    }
                    break;
                }

            }
            customFunc.Clear();
            customFunc.Add("BlendMode", BlendModePopup);
            customFunc.Add("Debug", DoDebugArea);
            AssetsConfig.RefreshShaderDebugNames();
            GameObject go = Selection.activeGameObject;
            if (go != null)
                dd = go.GetComponent<DebugData>();
            Update();
        }

        public override void AssignNewShaderToMaterial(Material material, Shader oldShader, Shader newShader)
        {
            if (m_Material != null)
            {
                if (material.HasProperty("_Emission") && material.HasProperty("_EmissionColor"))
                {
                    m_Material.SetColor("_EmissionColor", material.GetColor("_Emission"));
                }
                if (material.HasProperty("_MainTex"))
                {
                    m_Material.SetTexture("_BaseTex", material.GetTexture("_MainTex"));
                }
                if (material.HasProperty("_BaseTex"))
                {
                    m_Material.SetTexture("_MainTex", material.GetTexture("_BaseTex"));
                }
                if (material.HasProperty("_BumpMap"))
                {
                    m_Material.SetTexture("_PBSTex", material.GetTexture("_BumpMap"));
                }
            }

            base.AssignNewShaderToMaterial(material, oldShader, newShader);

            if (oldShader == null || !oldShader.name.Contains("Legacy Shaders/"))
            {
                ShaderAssets.SetupMaterialWithBlendMode(material, ShaderAssets.GetBlendMode(material));
                return;
            }

            MaterialChanged(material);
        }

        public override void OnGUI(MaterialEditor materialEditor, MaterialProperty[] props)
        {
            m_MaterialEditor = materialEditor;
            m_Material = materialEditor.target as Material;

            if (m_FirstTimeApply)
            {
                FindProperties(props);
                MaterialChanged(m_Material);
                m_FirstTimeApply = false;
            }
            ShaderPropertiesGUI();
        }

        private void Update()
        {
            if (shaderPropertyGroups != null)
            {
                drawList.Clear();
                for (int i = 0; i < shaderPropertyGroups.Count; ++i)
                {
                    ShaderGroupInstance sgi = shaderPropertyGroups[i];
                    if (sgi.hasCustomGroup)
                    {
                        sgi.spiList.Sort((x, y) =>
                        {
                            int idx = x.scp != null ? x.scp.indexInGroup : 0;
                            int idy = y.scp != null ? y.scp.indexInGroup : 0;
                            return idx - idy;
                        });
                    }
                    bool hasBeginFun = false;
                    for (int j = 0; j < sgi.spiList.Count; ++j)
                    {
                        ShaderPropertyInstance spi = sgi.spiList[j];
                        var sf = spi.shaderFeature;
                        if (j == 0 && (sf.indexInGroup >= 0 || sf.type == AssetsConfig.ShaderPropertyType.CustomGroup))
                        {
                            DrawPropertyContext context = new DrawPropertyContext();
                            context.fun = drawGroupBegin;
                            context.sgi = sgi;
                            drawList.Add(context);
                            hasBeginFun = true;
                        }
                        if (sf.type == AssetsConfig.ShaderPropertyType.CustomFun)
                        {
                            DrawFun fun = null;
                            if (customFunc.TryGetValue(sf.name, out fun))
                            {
                                DrawPropertyContext context = new DrawPropertyContext();
                                context.fun = fun;
                                context.material = m_Material;
                                context.materialEditor = m_MaterialEditor;
                                context.hasDepency = hasDepency;
                                context.spi = spi;
                                context.dd = dd;
                                drawList.Add(context);
                            }
                        }
                        else
                        {
                            bool show = true;
                            if (sf.dependencyPropertys != null && sf.dependencyPropertys.dependencyShaderProperty.Count > 0)
                            {
                                bool hasFeature = false;
                                for (int k = 0; k < sf.dependencyPropertys.dependencyShaderProperty.Count; ++k)
                                {
                                    string featureName = sf.dependencyPropertys.dependencyShaderProperty[k];
                                    if (sf.dependencyPropertys.dependencyType == AssetsConfig.DependencyType.Or)
                                    {
                                        hasFeature |= hasDepency.Contains(featureName);
                                        if (hasFeature)
                                            break;
                                    }
                                    else if (sf.dependencyPropertys.dependencyType == AssetsConfig.DependencyType.And)
                                    {
                                        hasFeature &= hasDepency.Contains(featureName);
                                        if (!hasFeature)
                                            break;
                                    }
                                }
                                if (sf.dependencyPropertys.isNor)
                                {
                                    hasFeature = !hasFeature;
                                }
                                show = hasFeature;
                            }
                            if (show && !sf.hide)
                            {
                                var drawFun = drawPropertyFun[(int)sf.type];
                                if (drawFun != null)
                                {
                                    DrawPropertyContext context = new DrawPropertyContext();
                                    context.fun = drawFun;
                                    context.material = m_Material;
                                    context.materialEditor = m_MaterialEditor;
                                    context.hasDepency = hasDepency;
                                    context.spi = spi;
                                    context.dd = dd;
                                    context.sgi = sgi;
                                    drawList.Add(context);
                                }
                            }
                            if (j == sgi.spiList.Count - 1 && hasBeginFun)
                            {
                                DrawPropertyContext context = new DrawPropertyContext();
                                context.fun = drawGroupEnd;
                                context.sgi = sgi;
                                drawList.Add(context);
                            }
                        }
                    }
                }
            }
        }
        private static void DrawTexFun(ref DrawPropertyContext context)
        {
            if (context.spi.property != null)
            {
                if (context.spi.shaderFeature.readOnly)
                {
                    EditorGUILayout.ObjectField(context.spi.property.displayName, context.spi.property.textureValue, typeof(Texture), false);
                }
                else
                {
                    EditorGUI.BeginChangeCheck();
                    context.materialEditor.TextureProperty(context.spi.property, context.spi.property.displayName, false);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(context.materialEditor.target, context.spi.property.displayName);
                        if (context.dd != null)
                        {
                            context.dd.SetTexture(context.spi.property.name, context.spi.property.textureValue);
                        }
                    }
                }
            }

        }
        private static void DrawColorFun(ref DrawPropertyContext context)
        {
            if (context.spi.property != null)
            {
                if (context.spi.shaderFeature.readOnly)
                {
                    EditorGUILayout.ColorField(context.spi.property.displayName, context.spi.property.colorValue);
                }
                else
                {
                    EditorGUI.BeginChangeCheck();
                    context.materialEditor.ColorProperty(context.spi.property, context.spi.property.displayName);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(context.materialEditor.target, context.spi.property.displayName);
                        if (context.dd != null)
                        {
                            context.dd.SetColor(context.spi.property.name, context.spi.property.colorValue);
                        }
                    }
                }
            }
        }

        private static void DrawVectorFun(ref DrawPropertyContext context)
        {
            if (context.spi.property != null)
            {
                if (context.spi.shaderFeature.readOnly)
                {
                    EditorGUILayout.Vector4Field(context.spi.property.displayName, context.spi.property.vectorValue);
                }
                else
                {
                    EditorGUI.BeginChangeCheck();
                    context.materialEditor.VectorProperty(context.spi.property, context.spi.property.displayName);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(context.materialEditor.target, context.spi.property.displayName);
                        if (context.dd != null)
                        {
                            context.dd.SetVector(context.spi.property.name, context.spi.property.vectorValue);
                        }
                    }
                }
            }
        }

        private static void DrawKeyWordFun(ref DrawPropertyContext context)
        {
            bool isEnable = context.material.IsKeywordEnabled(context.spi.shaderFeature.propertyName);
            if (context.spi.shaderFeature.readOnly)
            {
                EditorGUILayout.Toggle(context.spi.shaderFeature.name, isEnable);
            }
            else
            {
                EditorGUI.BeginChangeCheck();
                bool enable = EditorGUILayout.Toggle(context.spi.shaderFeature.name, isEnable);
                if (EditorGUI.EndChangeCheck())
                {
                    if (isEnable != enable)
                    {
                        SetKeyword(context.material, context.spi.shaderFeature.propertyName, enable);
                    }
                    if (enable && !context.hasDepency.Contains(context.spi.shaderFeature.name))
                        context.hasDepency.Add(context.spi.shaderFeature.name);
                    else if (!enable && context.hasDepency.Contains(context.spi.shaderFeature.name))
                        context.hasDepency.Remove(context.spi.shaderFeature.name);

                    context.materialEditor.RegisterPropertyChangeUndo(context.spi.shaderFeature.name);
                }
            }
        }

        private static void DrawCustomFun(ref DrawPropertyContext context)
        {
            if (context.spi.property != null)
            {
                if (context.spi.shaderFeature.readOnly)
                {
                    EditorGUILayout.Vector4Field(context.spi.property.displayName, context.spi.property.vectorValue);
                }
                else
                {
                    EditorGUI.BeginChangeCheck();
                    PropertyDrawer.OnGUI(context.spi.property, context.materialEditor, context.spi.shaderFeature);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(context.materialEditor.target, context.spi.property.displayName);
                    }
                }
            }
        }
        private static void DrawCustomGroupFun(ref DrawPropertyContext context)
        {
            if (context.spi.property != null && context.spi.scp != null)
            {
                var scp = context.spi.scp;
                if (context.spi.shaderFeature.readOnly)
                {
                    EditorGUILayout.FloatField(scp.desc, context.spi.property.vectorValue[context.spi.customIndex]);
                }
                else
                {
                    EditorGUI.BeginChangeCheck();
                    PropertyDrawer.OnGUI(context.spi.property, context.materialEditor, scp, context.spi.customIndex);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(context.materialEditor.target, context.spi.property.displayName);
                    }
                }
            }
        }
        private static void DrawRenderQueueFun(ref DrawPropertyContext context)
        {
            if (context.spi.shaderFeature != null)
            {
                if (context.spi.shaderFeature.readOnly)
                {
                    EditorGUILayout.IntField(context.spi.shaderFeature.name, context.material.renderQueue);
                }
                else
                {
                    EditorGUI.BeginChangeCheck();
                    int renderQueue = EditorGUILayout.IntField(context.spi.shaderFeature.name, context.material.renderQueue);
                    if (EditorGUI.EndChangeCheck())
                    {
                        renderQueue = Mathf.Clamp(renderQueue, -1, 5000);
                        context.material.renderQueue = renderQueue;
                        Undo.RecordObject(context.materialEditor.target, context.spi.shaderFeature.name);
                    }
                }
            }
        }

        private static void DrawGroupBegin(ref DrawPropertyContext context)
        {
            EditorGUILayout.LabelField(context.sgi.name, EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
        }

        private static void DrawGroupEnd(ref DrawPropertyContext context)
        {
            EditorGUI.indentLevel--;
        }

        protected virtual void ShaderPropertiesGUI()
        {
            EditorGUIUtility.labelWidth = 0f;

            EditorGUI.BeginChangeCheck();
            {
                EditorGUILayout.Toggle("LightMapOn", m_Material.IsKeywordEnabled("LIGHTMAP_ON"));
                EditorGUILayout.Toggle("CustomLightMapOn", m_Material.IsKeywordEnabled("_CUSTOM_LIGHTMAP_ON"));
                for (int i = 0; i < drawList.Count; ++i)
                {
                    var dc = drawList[i];
                    if (dc.fun != null)
                    {
                        dc.fun(ref dc);
                    }
                }
            }
            if (EditorGUI.EndChangeCheck())
            {
                if (m_MaterialEditor != null)
                {
                    Update();
                }
            }
        }

        protected void BlendModePopup(ref DrawPropertyContext context)
        {
            BlendMode blendMode = ShaderAssets.GetBlendMode(m_Material);

            EditorGUI.BeginChangeCheck();
            var mode = (BlendMode)EditorGUILayout.Popup(Styles.renderingMode, (int)blendMode, Styles.blendNames);
            if (EditorGUI.EndChangeCheck())
            {
                if (mode != blendMode)
                {
                    ShaderAssets.SetupMaterialWithBlendMode(m_Material, mode);
                }
            }
        }

        protected void DoDebugArea(ref DrawPropertyContext context)
        {
            if (context.spi.property != null)
            {

                if (AssetsConfig.shaderDebugNames != null)
                {
                    GUILayout.Label(Styles.debugText, EditorStyles.boldLabel);
                    EditorGUI.BeginChangeCheck();
                    var mode = context.spi.property.floatValue;
                    mode = EditorGUILayout.Popup(Styles.debugMode, (int)mode, AssetsConfig.shaderDebugNames);
                    if (EditorGUI.EndChangeCheck())
                    {
                        m_MaterialEditor.RegisterPropertyChangeUndo("Debug Mode");
                        context.spi.property.floatValue = mode;
                    }
                    string debugName = AssetsConfig.shaderDebugNames[(int)mode];
                    if (debugName == "CubeMipmap")
                    {
                        GUILayout.BeginHorizontal();
                        int swatchSize = 22;
                        int viewWidth = (int)EditorGUIUtility.currentViewWidth - 12;
                        int swatchesPerRow = viewWidth / (swatchSize + 4);
                        swatchSize += (viewWidth % (swatchSize + 4)) / swatchesPerRow;
                        for (int i = 0; i < mipMapColors.Length; ++i)
                        {
                            GUI.backgroundColor = mipMapColors[i].linear;
                            GUILayout.Button(i.ToString(),
                                GUILayout.MinWidth(swatchSize),
                                GUILayout.MaxWidth(swatchSize),
                                GUILayout.MinHeight(swatchSize),
                                GUILayout.MaxHeight(swatchSize));
                        }
                        GUI.backgroundColor = Color.white;
                        GUILayout.EndHorizontal();
                    }
                }
            }
        }

        public static void SetupMaterialWithDebugMode(Material material, float debugMode)
        {
            material.SetFloat("_DebugMode", (int)debugMode);
        }

        protected void MaterialChanged(Material material)
        {
            ShaderAssets.SetupMaterialWithBlendMode(material, ShaderAssets.GetBlendMode(material), false);
            if (material.HasProperty("_DebugMode"))
                SetupMaterialWithDebugMode(material, material.GetFloat("_DebugMode"));
        }

        static void SetKeyword(Material m, string keyword, bool state)
        {
            if (state)
                m.EnableKeyword(keyword);
            else
                m.DisableKeyword(keyword);
        }
    }
}