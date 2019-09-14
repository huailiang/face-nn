using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using CFEngine;
using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;

namespace CFEngine.Editor
{
    [CustomEditor(typeof(AssetsConfig))]
    public class AssetsConfigEdit : BaseEditor<AssetsConfigEdit>
    {
        public enum OpType
        {
            None,
            OpGenAllSceneMat,
            OpRefreshAllSceneMat,
            OpSaveAllSceneMat,
            OpGenMat,
            OpGenEffectMat,
            OpRefreshMat,
        }

        private List<SerializedProperty> constPropertyList;
        private List<SerializedProperty> constListPropertyList;
        private List<SerializedProperty> commonPropertyList;
        private string groupName = "";
        private Dictionary<string, List<AssetsConfig.ShaderFeature>> groupedShaderFeatures = new Dictionary<string, List<AssetsConfig.ShaderFeature>>();
        private string[] shaderGroupNames = null;
        private bool shaderGroupDirty = true;
        private List<string> shaderFeatures = new List<string>();
        private AssetsConfig.DummyMaterialInfo genMat = null;
        private bool multiBlend;

        private int dragType = -1;
        private string dragName = "";
        private int dragIndex = -1;

        private Rect groupRect;
        private Rect shaderFeatureRect;
        private Rect dummyMaterialsRect;
        private OpType opType = OpType.None;
        private int shaderFeatureCopyIndex = -1;
        private int matInfoCopyIndex = -1;
        private void OnEnable()
        {
            constPropertyList = new List<SerializedProperty>();
            constListPropertyList = new List<SerializedProperty>();
            commonPropertyList = new List<SerializedProperty>();
            Type t = typeof(AssetsConfig);
            FieldInfo[] fields = t.GetFields();
            for (int i = 0; i < fields.Length; ++i)
            {
                FieldInfo fi = fields[i];
                var atttrs = fi.GetCustomAttributes(typeof(HideInInspector), false);
                if (atttrs != null && atttrs.Length > 0)
                    continue;
                if (fi.FieldType == typeof(string))
                {
                    SerializedProperty sp = serializedObject.FindProperty(fi.Name);
                    if (sp != null)
                    {
                        constPropertyList.Add(sp);
                    }
                }
                else if (fi.FieldType == typeof(List<AssetsConfig.ShaderInfo>) ||
                    fi.FieldType == typeof(List<AssetsConfig.ShaderFeature>) ||
                    fi.FieldType == typeof(List<AssetsConfig.DummyMaterialInfo>) ||
                    fi.Name == "ShaderGroupInfo" ||
                    fi.Name == "TextureTypes" ||
                    fi.Name == "TextureFolders")
                {
                    //custom editor
                }
                else if (fi.FieldType == typeof(System.Array))
                {
                    SerializedProperty sp = serializedObject.FindProperty(fi.Name);
                    if (sp != null)
                    {
                        constListPropertyList.Add(sp);
                    }
                }
                else
                {
                    SerializedProperty sp = serializedObject.FindProperty(fi.Name);
                    if (sp != null)
                    {
                        commonPropertyList.Add(sp);
                    }
                }
            }
        }
        private void BeginRect(ref Rect rect)
        {
            rect = GUILayoutUtility.GetLastRect();
        }
        private void BeginRect(ref Rect rect, IRectSelect rs, int index, int width)
        {
            Rect r = GUILayoutUtility.GetLastRect();
            r.size = new Vector2(width, 20);
            rs.SelectRect = r;
            if (index == 0)
            {
                BeginRect(ref rect);
            }
        }

        private void EndRect(ref Rect rect, int width)
        {
            Rect r = GUILayoutUtility.GetLastRect();
            rect.size = new Vector2(width, r.y + r.height - rect.yMin);
        }

        private bool DragEffect(ref Rect rect, Event e, int groupType, int count)
        {
            if (rect.Contains(e.mousePosition) && count > 0)
            {
                float y = e.mousePosition.y - rect.yMin;
                dragType = groupType;
                dragIndex = (int)(y / 20);
                if (dragIndex >= count)
                {
                    dragIndex = count - 1;
                }
                return true;
            }
            return false;
        }
        private bool DragEffect(ref Rect rect, Event e, int groupType, IList list)
        {
            if (rect.Contains(e.mousePosition) && list.Count > 0)
            {
                for (int i = 0; i < list.Count; ++i)
                {
                    var obj = list[i] as IRectSelect;
                    if (obj.SelectRect.Contains(e.mousePosition))
                    {
                        dragType = 1;
                        dragIndex = i;
                        dragName = obj.Name;
                        return true;
                    }
                }
            }
            return false;
        }
        private void DragEffectEnd(ref Rect rect, Event e, IList list, int lastDragIndex)
        {
            if (rect.Contains(e.mousePosition) && list.Count > 0)
            {
                var dragObj = list[dragIndex] as IRectSelect;
                for (int i = 0; i < list.Count; ++i)
                {
                    var obj = list[i] as IRectSelect;
                    if (obj.SelectRect.Contains(e.mousePosition) && lastDragIndex != i)
                    {
                        float offsetY = e.mousePosition.y - obj.SelectRect.yMin;
                        bool insertUp = offsetY < obj.SelectRect.height * 0.5f;
                        ReRange(list, insertUp, dragObj, i, lastDragIndex);
                        break;
                    }
                }
            }
        }

        private bool CalcDragIndex(ref Rect rect, Event e, int count)
        {
            int upRectCount = dragIndex > 0 ? dragIndex - 1 : 0;
            float height = rect.height / count;
            float offsetY = e.mousePosition.y - rect.yMin - upRectCount * height;
            return offsetY < height * 0.5f;
        }

        private void ReRange(IList list, bool insertUp, object obj, int insertIndex, int lastDragIndex)
        {
            if (!insertUp)
            {
                insertIndex = insertIndex + 1;
            }
            if (insertIndex >= list.Count)
            {
                list.RemoveAt(lastDragIndex);
                list.Add(obj);
            }
            else
            {
                if (insertIndex < lastDragIndex)
                {
                    list.RemoveAt(lastDragIndex);
                    list.Insert(insertIndex, obj);
                }
                else
                {
                    list.Insert(insertIndex, obj);
                    list.RemoveAt(lastDragIndex);
                }
            }
        }

        private void OnEventProcessGUI(AssetsConfig ac)
        {
            var e = Event.current;
            if (e.type == EventType.MouseDown)
            {
                if (DragEffect(ref groupRect, e, 0, ac.ShaderGroupInfo.Count))
                {
                    dragName = ac.ShaderGroupInfo[dragIndex];
                }
                else if (DragEffect(ref shaderFeatureRect, e, 1, ac.ShaderFeatures)) { }
                else if (DragEffect(ref dummyMaterialsRect, e, 2, ac.roleMaterials)) { }
                Repaint();
            }
            else if (e.type == EventType.MouseDrag)
            {
                if (dragType >= 0)
                {
                    Repaint();
                }
            }
            else if (e.type == EventType.MouseUp)
            {
                int lastDragIndex = dragIndex;
                if (dragType == 0)
                {
                    if (DragEffect(ref groupRect, e, 0, ac.ShaderGroupInfo.Count) && lastDragIndex != dragIndex)
                    {
                        bool insertUp = CalcDragIndex(ref groupRect, e, ac.ShaderGroupInfo.Count);
                        ReRange(ac.ShaderGroupInfo, insertUp, dragName, dragIndex, lastDragIndex);
                    }
                }
                else if (dragType == 1)
                {
                    DragEffectEnd(ref shaderFeatureRect, e, ac.ShaderFeatures, lastDragIndex);
                }
                else if (dragType == 2)
                {
                    DragEffectEnd(ref dummyMaterialsRect, e, ac.roleMaterials, lastDragIndex);
                }
                dragType = -1;
                dragName = "";
                dragIndex = -1;
                Repaint();
            }
            if (e.type == EventType.Repaint)
            {
                if (dragType >= 0)
                {
                    Handles.Label(e.mousePosition, dragName);
                }
            }
        }
        private void ConstValuesGUI(AssetsConfig ac)
        {
            ac.commonFolder = EditorGUILayout.Foldout(ac.commonFolder, "Const");
            if (ac.commonFolder)
            {
                if (constPropertyList != null)
                {
                    for (int i = 0; i < constPropertyList.Count; ++i)
                    {
                        SerializedProperty sp = constPropertyList[i];
                        EditorGUI.BeginChangeCheck();
                        var str = EditorGUILayout.TextField(sp.displayName, sp.stringValue);
                        if (EditorGUI.EndChangeCheck())
                        {
                            Undo.RecordObject(target, sp.name);
                            sp.stringValue = str;
                        }
                    }
                }
                if (constListPropertyList != null)
                {
                    for (int i = 0; i < constListPropertyList.Count; ++i)
                    {
                        SerializedProperty sp = constListPropertyList[i];
                        EditorGUILayout.PropertyField(sp, true);
                    }
                }
                if (commonPropertyList != null)
                {
                    for (int i = 0; i < commonPropertyList.Count; ++i)
                    {
                        SerializedProperty sp = commonPropertyList[i];
                        EditorGUILayout.PropertyField(sp, true);
                    }
                }
            }

        }

        private void TexturePlatformConfigGUI(TexImportSetting tis, string name)
        {
            tis.folder = EditorGUILayout.Foldout(tis.folder, name);
            if (tis.folder)
            {
                tis.maxTextureSize = (SpriteSize)EditorGUILayout.EnumPopup("Size", tis.maxTextureSize, GUILayout.MaxWidth(300));
                tis.format = (TextureImporterFormat)EditorGUILayout.EnumPopup("Format", tis.format, GUILayout.MaxWidth(300));
                tis.alphaFormat = (TextureImporterFormat)EditorGUILayout.EnumPopup("Format_A", tis.alphaFormat, GUILayout.MaxWidth(300));
            }
        }
        private void TextureProcessGUI(AssetsConfig ac)
        {
            ac.texCompressConfigFolder = EditorGUILayout.Foldout(ac.texCompressConfigFolder, "Texture Compress");
            if (ac.texCompressConfigFolder)
            {
                if (GUILayout.Button("Add", GUILayout.MaxWidth(80)))
                {
                    ac.texCompressConfig.Add(new TexCompressConfig());
                }
                int removeIndex = -1;
                for (int i = 0; i < ac.texCompressConfig.Count; ++i)
                {
                    var tcc = ac.texCompressConfig[i];
                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.LabelField(string.IsNullOrEmpty(tcc.name) ? "empty" : tcc.name, GUILayout.MaxWidth(150));
                    if (GUILayout.Button("Edit", GUILayout.MaxWidth(80)))
                    {
                        tcc.folder = !tcc.folder;
                    }

                    if (GUILayout.Button("Delete", GUILayout.MaxWidth(80)))
                    {
                        removeIndex = i;
                    }
                    EditorGUILayout.EndHorizontal();
                    if (tcc.folder)
                    {
                        tcc.vaild = EditorGUILayout.Toggle("Valid", tcc.vaild);
                        tcc.name = EditorGUILayout.TextField("Name", tcc.name);
                        tcc.priority = EditorGUILayout.IntField("Priority", tcc.priority);
                        tcc.importType = (TextureImporterType)EditorGUILayout.EnumPopup("Type", tcc.importType);
                        tcc.importShape = (TextureImporterShape)EditorGUILayout.EnumPopup("Shape", tcc.importShape);
                        tcc.sRGB = EditorGUILayout.Toggle("sRGB", tcc.sRGB);
                        tcc.mipMap = EditorGUILayout.Toggle("Mipmap", tcc.mipMap);
                        tcc.filterMode = (FilterMode)EditorGUILayout.EnumPopup("Filter", tcc.filterMode);
                        tcc.wrapMode = (TextureWrapMode)EditorGUILayout.EnumPopup("Wrap", tcc.wrapMode);
                        tcc.anisoLevel = EditorGUILayout.IntSlider("AnisoLevel", tcc.anisoLevel, 0, 3);
                        EditorGUI.indentLevel++;
                        TexturePlatformConfigGUI(tcc.iosSetting, "iOS");
                        TexturePlatformConfigGUI(tcc.androidSetting, "Android");
                        TexturePlatformConfigGUI(tcc.standaloneSetting, "Standalone");
                        EditorGUI.indentLevel--;
                        EditorGUI.indentLevel++;
                        if (GUILayout.Button("AddFilter", GUILayout.MaxWidth(80)))
                        {
                            tcc.compressFilters.Add(new TexCompressFilter());
                        }
                        int subremoveIndex = -1;
                        for (int j = 0; j < tcc.compressFilters.Count; ++j)
                        {
                            var cf = tcc.compressFilters[j];
                            EditorGUILayout.BeginHorizontal();
                            cf.type = (TexFilterType)EditorGUILayout.EnumPopup("", cf.type, GUILayout.MaxWidth(100));
                            cf.str = EditorGUILayout.TextField("", cf.str, GUILayout.MaxWidth(300));
                            if (GUILayout.Button("Delete", GUILayout.MaxWidth(80)))
                            {
                                subremoveIndex = j;
                            }
                            EditorGUILayout.EndHorizontal();
                        }
                        EditorGUI.indentLevel--;
                        if (subremoveIndex >= 0)
                        {
                            tcc.compressFilters.RemoveAt(subremoveIndex);
                        }
                    }

                }
                if (removeIndex >= 0)
                {
                    ac.texCompressConfig.RemoveAt(removeIndex);
                }
            }
        }

        private void SceneDummyMatGUI(AssetsConfig.DummyMaterialInfo dummyMaterialInfo, int i, string name)
        {
            ToolsUtility.BeginGroup("");

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(string.Format("{0}.{1}", i.ToString(), name), GUILayout.MaxWidth(250));
            if (GUILayout.Button("Reset", GUILayout.MaxWidth(80)))
            {
                dummyMaterialInfo.enumIndex = -1;
                dummyMaterialInfo.mat = null;
                dummyMaterialInfo.mat1 = null;
                dummyMaterialInfo.mat2 = null;
            }
            if (GUILayout.Button("Gen", GUILayout.MaxWidth(80)))
            {
                opType = OpType.OpGenMat;
                genMat = dummyMaterialInfo;
            }
            if (GUILayout.Button("Refresh", GUILayout.MaxWidth(80)))
            {
                opType = OpType.OpRefreshMat;
                genMat = dummyMaterialInfo;
            }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.Space();
            EditorGUI.indentLevel++;
            EditorGUILayout.BeginHorizontal();
            dummyMaterialInfo.name = EditorGUILayout.TextField(dummyMaterialInfo.name, GUILayout.MaxWidth(300));
            dummyMaterialInfo.shader = EditorGUILayout.ObjectField(dummyMaterialInfo.shader, typeof(Shader), false, GUILayout.MaxWidth(300)) as Shader;
            EditorGUILayout.EndHorizontal();

            EditorGUI.indentLevel--;
            ToolsUtility.EndGroup();
        }

        private void RoleDummyMatGUI(AssetsConfig.DummyMaterialInfo dummyMaterialInfo, int i, string name, bool multiBlendType)
        {
            ToolsUtility.BeginGroup("");

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(string.Format("{0}.{1}", i.ToString(), name), GUILayout.MaxWidth(200));
            if (GUILayout.Button("Reset", GUILayout.MaxWidth(80)))
            {
                dummyMaterialInfo.enumIndex = -1;
            }
            if (GUILayout.Button("Gen", GUILayout.MaxWidth(80)))
            {
                opType = OpType.OpGenEffectMat;
                genMat = dummyMaterialInfo;
                multiBlend = multiBlendType;
            }

            EditorGUILayout.EndHorizontal();
            EditorGUILayout.Space();
            EditorGUI.indentLevel++;
            EditorGUILayout.BeginHorizontal();
            dummyMaterialInfo.name = EditorGUILayout.TextField(dummyMaterialInfo.name, GUILayout.MaxWidth(300));
            dummyMaterialInfo.shader = EditorGUILayout.ObjectField(dummyMaterialInfo.shader, typeof(Shader), false, GUILayout.MaxWidth(300)) as Shader;
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (multiBlendType)
                dummyMaterialInfo.blendType = (EBlendType)EditorGUILayout.EnumFlagsField("", dummyMaterialInfo.blendType, GUILayout.MaxWidth(100));
            else
                dummyMaterialInfo.blendType = (EBlendType)EditorGUILayout.EnumPopup("", dummyMaterialInfo.blendType, GUILayout.MaxWidth(100));
            KeywordFlags e = (KeywordFlags)EditorGUILayout.EnumFlagsField((KeywordFlags)dummyMaterialInfo.flag, GUILayout.MaxWidth(200));
            dummyMaterialInfo.flag = (uint)e;
            EditorGUILayout.LabelField(MaterialShaderAssets.GetKeyWords(e));
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (multiBlendType)
            {
                if ((dummyMaterialInfo.blendType & EBlendType.Opaque) != 0)
                    EditorGUILayout.ObjectField(dummyMaterialInfo.mat, typeof(Material), false, GUILayout.MaxWidth(300));
                if ((dummyMaterialInfo.blendType & EBlendType.Cutout) != 0)
                {
                    if (dummyMaterialInfo.ext1 != "Cutout")
                        dummyMaterialInfo.ext1 = "Cutout";
                    EditorGUILayout.ObjectField(dummyMaterialInfo.mat1, typeof(Material), false, GUILayout.MaxWidth(300));
                }
                else
                {
                    if (dummyMaterialInfo.ext1 != "")
                        dummyMaterialInfo.ext1 = "";
                }
                if ((dummyMaterialInfo.blendType & EBlendType.CutoutTransparent) != 0)
                {
                    if (dummyMaterialInfo.ext2 != "CutoutBlend")
                        dummyMaterialInfo.ext2 = "CutoutBlend";
                    EditorGUILayout.ObjectField(dummyMaterialInfo.mat2, typeof(Material), false, GUILayout.MaxWidth(300));
                }
                else
                {
                    if (dummyMaterialInfo.ext2 != "")
                        dummyMaterialInfo.ext2 = "";
                }
            }
            else
            {
                EditorGUILayout.ObjectField(dummyMaterialInfo.mat, typeof(Material), false, GUILayout.MaxWidth(300));
            }

            EditorGUILayout.EndHorizontal();
            EditorGUI.indentLevel--;
            ToolsUtility.EndGroup();
        }
        private void SpecialDummyMatGUI(AssetsConfig.DummyMaterialInfo dummyMaterialInfo, int i, string name)
        {
            ToolsUtility.BeginGroup("");

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(string.Format("{0}.{1}", i.ToString(), name), GUILayout.MaxWidth(200));
            if (GUILayout.Button("Reset", GUILayout.MaxWidth(80)))
            {
                dummyMaterialInfo.enumIndex = -1;
            }
            if (GUILayout.Button("Gen", GUILayout.MaxWidth(80)))
            {
                MaterialShaderAssets.DefaultEffectMat(dummyMaterialInfo, false);
            }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.Space();
            EditorGUI.indentLevel++;
            EditorGUILayout.BeginHorizontal();
            dummyMaterialInfo.name = EditorGUILayout.TextField(dummyMaterialInfo.name, GUILayout.MaxWidth(300));
            dummyMaterialInfo.shader = EditorGUILayout.ObjectField(dummyMaterialInfo.shader, typeof(Shader), false, GUILayout.MaxWidth(300)) as Shader;
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            dummyMaterialInfo.blendType = (EBlendType)EditorGUILayout.EnumPopup("", dummyMaterialInfo.blendType, GUILayout.MaxWidth(100));
            KeywordFlags e = (KeywordFlags)EditorGUILayout.EnumFlagsField((KeywordFlags)dummyMaterialInfo.flag, GUILayout.MaxWidth(200));
            dummyMaterialInfo.flag = (uint)e;
            EditorGUILayout.LabelField(MaterialShaderAssets.GetKeyWords(e));
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.ObjectField(dummyMaterialInfo.mat, typeof(Material), false, GUILayout.MaxWidth(300));
            EditorGUILayout.EndHorizontal();
            EditorGUI.indentLevel--;
            ToolsUtility.EndGroup();
        }

        private static void InitMatNames(List<AssetsConfig.DummyMaterialInfo> materials, int matCount, ref List<string> matNames, Type enumType)
        {
            if (materials.Count < matCount)
            {
                for (int i = 0; i < matCount - materials.Count; ++i)
                {
                    materials.Add(new AssetsConfig.DummyMaterialInfo());
                }
            }
            if (matNames == null)
            {
                string[] names = Enum.GetNames(enumType);
                matNames = new List<string>(names);
            }

            for (int i = 0; i < matCount; ++i)
            {
                var dummyMaterialInfo = materials[i];
                if (dummyMaterialInfo.enumIndex == -1 && !string.IsNullOrEmpty(dummyMaterialInfo.name))
                {
                    dummyMaterialInfo.enumIndex = matNames.FindIndex((x) => { return x == dummyMaterialInfo.name; });
                }
                if (i != dummyMaterialInfo.enumIndex && dummyMaterialInfo.enumIndex >= 0)
                {
                    var tmp = materials[dummyMaterialInfo.enumIndex];
                    materials[dummyMaterialInfo.enumIndex] = dummyMaterialInfo;
                    materials[i] = tmp;
                }
                if (dummyMaterialInfo.shader != null && dummyMaterialInfo.enumIndex >= 0)
                {
                    if (dummyMaterialInfo.mat == null)
                    {
                        dummyMaterialInfo.mat = MaterialShaderAssets.GetDummyMat(dummyMaterialInfo.name);
                    }
                    if (dummyMaterialInfo.mat1 == null && !string.IsNullOrEmpty(dummyMaterialInfo.ext1))
                    {
                        dummyMaterialInfo.mat1 = MaterialShaderAssets.GetDummyMat(dummyMaterialInfo.name + dummyMaterialInfo.ext1);
                    }
                    if (dummyMaterialInfo.mat2 == null && !string.IsNullOrEmpty(dummyMaterialInfo.ext2))
                    {
                        dummyMaterialInfo.mat2 = MaterialShaderAssets.GetDummyMat(dummyMaterialInfo.name + dummyMaterialInfo.ext2);
                    }
                    if (dummyMaterialInfo.mat3 == null && !string.IsNullOrEmpty(dummyMaterialInfo.ext3))
                    {
                        dummyMaterialInfo.mat3 = MaterialShaderAssets.GetDummyMat(dummyMaterialInfo.name + dummyMaterialInfo.ext3);
                    }
                }
            }
        }
        private void DummyMatGUI(AssetsConfig ac)
        {
            ac.dummyMaterialsFolder = EditorGUILayout.Foldout(ac.dummyMaterialsFolder, "Dummy Materials");
            if (ac.dummyMaterialsFolder)
            {
                if (GUILayout.Button("SaveAll", GUILayout.MaxWidth(80)))
                {
                    opType = OpType.OpSaveAllSceneMat;
                }
                CustomMatGUI(ac);
            }
        }
        private void CustomMatGUI(AssetsConfig ac)
        {
            EditorGUI.indentLevel++;
            ac.customMatInfoFolder = EditorGUILayout.Foldout(ac.customMatInfoFolder, "Custom Materials");
            if (ac.customMatInfoFolder)
            {
                EditorGUILayout.LabelField("Custom");
                for (int i = 0; i < ac.customMaterials.Count; ++i)
                {
                    var dummyMaterialInfo = ac.customMaterials[i];
                    RoleDummyMatGUI(dummyMaterialInfo, i, ((ECharMaterial)dummyMaterialInfo.enumIndex).ToString(), true);
                }
            }
            EditorGUI.indentLevel--;
        }

        private void ShaderGroupGUI(AssetsConfig ac)
        {
            ac.groupFolder = EditorGUILayout.Foldout(ac.groupFolder, "Shader Group");
            if (ac.groupFolder)
            {
                groupName = EditorGUILayout.TextField(groupName);
                if (GUILayout.Button("Add Shader Gropu", GUILayout.MaxWidth(180)))
                {
                    groupName = groupName.Trim();
                    if (!string.IsNullOrEmpty(groupName))
                    {
                        if (!ac.ShaderGroupInfo.Exists((param => { return param == groupName; })))
                        {
                            ac.ShaderGroupInfo.Add(groupName);
                            shaderGroupDirty = true;
                        }
                    }
                }

                for (int i = 0; i < ac.ShaderGroupInfo.Count; ++i)
                {
                    string name = ac.ShaderGroupInfo[i];
                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.LabelField(name, GUILayout.MaxWidth(100));
                    if (GUILayout.Button("Delete", GUILayout.MaxWidth(80)))
                    {
                        ac.ShaderGroupInfo.RemoveAt(i);
                        shaderGroupDirty = true;
                    }
                    EditorGUILayout.EndHorizontal();
                    if (i == 0)
                    {
                        BeginRect(ref groupRect);
                    }
                }
                EndRect(ref groupRect, 100);

            }
            else
            {
                groupRect = Rect.zero;
            }
            if (shaderGroupDirty)
            {
                shaderGroupNames = ac.ShaderGroupInfo.ToArray();
                shaderGroupDirty = false;
                for (int i = 0; i < ac.ShaderFeatures.Count; ++i)
                {
                    var sf = ac.ShaderFeatures[i];
                    if (sf.type != AssetsConfig.ShaderPropertyType.CustomGroup)
                    {
                        sf.shaderGroupIndex = ac.ShaderGroupInfo.IndexOf(sf.shaderGroupName);
                    }
                    else
                    {
                        for (int j = 0; j < sf.customProperty.Length; ++j)
                        {
                            var scp = sf.customProperty[j];
                            scp.shaderGroupIndex = ac.ShaderGroupInfo.IndexOf(scp.subGroup);
                        }
                    }
                }
            }
        }

        private void GroupNameGUI(ref int groupNameIndex, ref string groupName)
        {
            if (groupNameIndex >= 0 && shaderGroupNames != null)
            {
                int newIndex = EditorGUILayout.Popup("GroupName", groupNameIndex, shaderGroupNames);
                if (newIndex != groupNameIndex)
                {
                    groupNameIndex = newIndex;
                    groupName = shaderGroupNames[newIndex];
                }
            }
            else
            {
                groupName = EditorGUILayout.TextField("GroupName", groupName);
            }
        }
        private void ShaderFeatureGUI(AssetsConfig ac)
        {
            ac.shaderFeatureFolder = EditorGUILayout.Foldout(ac.shaderFeatureFolder, "Shader Feature");
            if (ac.shaderFeatureFolder)
            {
                int removeIndex = -1;
                for (int i = 0; i < ac.ShaderFeatures.Count; ++i)
                {
                    var sf = ac.ShaderFeatures[i];

                    EditorGUILayout.BeginHorizontal(GUILayout.MaxWidth(400));
                    EditorGUILayout.LabelField(sf.name, GUILayout.MaxWidth(150));
                    BeginRect(ref shaderFeatureRect, sf, i, 150);
                    if (GUILayout.Button("Edit", GUILayout.MaxWidth(80)))
                    {
                        sf.folder = !sf.folder;
                    }

                    if (GUILayout.Button("Delete", GUILayout.MaxWidth(80)))
                    {
                        removeIndex = i;
                        shaderFeatureCopyIndex = -1;
                    }
                    if (shaderFeatureCopyIndex == -1 || shaderFeatureCopyIndex == i)
                    {
                        if (GUILayout.Button("Copy", GUILayout.MaxWidth(80)))
                        {
                            if (shaderFeatureCopyIndex == -1)
                                shaderFeatureCopyIndex = i;
                            else
                                shaderFeatureCopyIndex = -1;
                        }
                    }
                    else
                    {
                        if (GUILayout.Button("Paste", GUILayout.MaxWidth(80)))
                        {
                            var src = ac.ShaderFeatures[shaderFeatureCopyIndex];
                            sf.Clone(src);
                        }
                    }
                    EditorGUILayout.EndHorizontal();
                    if (sf.folder)
                    {
                        sf.name = EditorGUILayout.TextField("Name", sf.name);
                        sf.hide = EditorGUILayout.Toggle("hide", sf.hide);
                        sf.readOnly = EditorGUILayout.Toggle("readOnly", sf.readOnly);
                        sf.type = (AssetsConfig.ShaderPropertyType)EditorGUILayout.EnumPopup("type", sf.type);

                        if (sf.type != AssetsConfig.ShaderPropertyType.CustomGroup)
                        {
                            GroupNameGUI(ref sf.shaderGroupIndex, ref sf.shaderGroupName);
                            sf.indexInGroup = EditorGUILayout.IntField("GroupIndex", sf.indexInGroup);
                        }
                        if (sf.type != AssetsConfig.ShaderPropertyType.RenderQueue)
                        {
                            GUILayout.Space(10);
                            sf.propertyName = EditorGUILayout.TextField("PropertyName", sf.propertyName);
                            EditorGUI.indentLevel++;
                            if (sf.type == AssetsConfig.ShaderPropertyType.Custom ||
                                sf.type == AssetsConfig.ShaderPropertyType.CustomGroup)
                            {
                                for (int j = 0; j < sf.customProperty.Length; ++j)
                                {
                                    var scp = sf.customProperty[j];
                                    string desc = string.IsNullOrEmpty(scp.desc) || !scp.valid ? "value" + j.ToString() : scp.desc;
                                    scp.folder = EditorGUILayout.Foldout(scp.folder, desc);
                                    if (scp.folder)
                                    {
                                        scp.valid = EditorGUILayout.Toggle("valid", scp.valid);
                                        scp.desc = EditorGUILayout.TextField("desc", scp.desc);
                                        scp.defaultValue = EditorGUILayout.Slider("default", scp.defaultValue, scp.min, scp.max);
                                        scp.min = EditorGUILayout.FloatField("min", scp.min);
                                        scp.max = EditorGUILayout.FloatField("max", scp.max);
                                        if (scp.max < scp.min)
                                        {
                                            scp.max = scp.min;
                                        }
                                        if (sf.type == AssetsConfig.ShaderPropertyType.CustomGroup)
                                        {
                                            GroupNameGUI(ref scp.shaderGroupIndex, ref scp.subGroup);
                                            scp.indexInGroup = EditorGUILayout.IntField("GroupIndex", scp.indexInGroup);
                                        }
                                    }
                                }
                            }
                            EditorGUI.indentLevel--;
                            GUILayout.Space(10);
                            EditorGUILayout.LabelField("Shader Dependency", GUILayout.MaxWidth(100));
                            sf.dependencyPropertys.isNor = EditorGUILayout.Toggle("isNor", sf.dependencyPropertys.isNor);
                            sf.dependencyPropertys.dependencyType = (AssetsConfig.DependencyType)EditorGUILayout.EnumPopup("depType", sf.dependencyPropertys.dependencyType);
                            for (int j = 0; j < sf.dependencyPropertys.dependencyShaderProperty.Count; ++j)
                            {
                                EditorGUILayout.BeginHorizontal();
                                sf.dependencyPropertys.dependencyShaderProperty[j] = EditorGUILayout.TextField("Dep", sf.dependencyPropertys.dependencyShaderProperty[j]);
                                if (GUILayout.Button("Remove", GUILayout.MaxWidth(100)))
                                {
                                    sf.dependencyPropertys.dependencyShaderProperty.RemoveAt(j);
                                }
                                EditorGUILayout.EndHorizontal();
                            }
                            if (GUILayout.Button("Add Dep", GUILayout.MaxWidth(100)))
                            {
                                sf.dependencyPropertys.dependencyShaderProperty.Add("");
                            }
                        }

                    }

                }
                EndRect(ref shaderFeatureRect, 150);
                if (removeIndex != -1)
                {
                    ac.ShaderFeatures.RemoveAt(removeIndex);
                }
                if (GUILayout.Button("Add Shader Feature", GUILayout.MaxWidth(180)))
                {
                    ac.ShaderFeatures.Add(new AssetsConfig.ShaderFeature());
                }
            }
            else
            {
                shaderFeatureRect = Rect.zero;
            }
        }
        private void ShaderFeatureInfoGUI(AssetsConfig.ShaderInfo si, AssetsConfig.ShaderFeature sf, ref bool featureDirty)
        {
            bool hasFeature = si.shaderFeatures.Exists((x) => { return sf.name == x; });
            bool newHasFeature = GUILayout.Toggle(hasFeature, sf.name, GUILayout.MaxWidth(160));
            if (newHasFeature)
            {
                shaderFeatures.Add(sf.name);
            }
            if (hasFeature != newHasFeature)
            {
                featureDirty = true;
            }
        }

        private void ShaderInfoGUI(AssetsConfig ac)
        {
            ac.shaderInfoFolder = EditorGUILayout.Foldout(ac.shaderInfoFolder, "Shader Info");
            if (ac.shaderInfoFolder)
            {
                EditorGUI.indentLevel++;

                for (int i = 0; i < ac.ShaderInfos.Count; ++i)
                {
                    AssetsConfig.ShaderInfo si = ac.ShaderInfos[i];
                    si.folder = EditorGUILayout.Foldout(si.folder, si.shader != null ? si.shader.name : "empty");
                    if (si.folder)
                    {
                        EditorGUILayout.BeginHorizontal();
                        if (GUILayout.Button("Delete", GUILayout.MaxWidth(80)))
                        {
                            ac.ShaderInfos.RemoveAt(i);
                        }
                        if (GUILayout.Button("Clear", GUILayout.MaxWidth(80)))
                        {
                            si.shaderFeatures.Clear();
                        }
                        EditorGUILayout.EndHorizontal();
                        si.shader = EditorGUILayout.ObjectField(si.shader, typeof(Shader), false, GUILayout.MaxWidth(160)) as Shader;
                        bool featureDirty = false;
                        shaderFeatures.Clear();

                        var it = groupedShaderFeatures.GetEnumerator();
                        while (it.MoveNext())
                        {
                            var kvp = it.Current;
                            EditorGUILayout.LabelField(kvp.Key);
                            EditorGUI.indentLevel++;
                            var sfList = kvp.Value;
                            for (int j = 0; j < sfList.Count; j += 3)
                            {
                                EditorGUILayout.BeginHorizontal();
                                AssetsConfig.ShaderFeature sf = sfList[j];
                                ShaderFeatureInfoGUI(si, sf, ref featureDirty);
                                if (j + 1 < sfList.Count)
                                {
                                    sf = sfList[j + 1];
                                    ShaderFeatureInfoGUI(si, sf, ref featureDirty);
                                    if (j + 2 < sfList.Count)
                                    {
                                        sf = sfList[j + 2];
                                        ShaderFeatureInfoGUI(si, sf, ref featureDirty);
                                    }
                                }
                                EditorGUILayout.EndHorizontal();
                            }
                            EditorGUI.indentLevel--;
                            EditorGUILayout.Space();
                        }
                        if (featureDirty)
                        {
                            si.shaderFeatures.Clear();
                            si.shaderFeatures.AddRange(shaderFeatures);
                        }
                    }
                }
                EditorGUI.indentLevel--;
                if (GUILayout.Button("Add Shader Config", GUILayout.MaxWidth(180)))
                {
                    ac.ShaderInfos.Add(new AssetsConfig.ShaderInfo());
                }
                if (GUILayout.Button("Sort Shader Features", GUILayout.MaxWidth(180)))
                {
                    List<string> shaderFeatures = new List<string>();
                    for (int i = 0; i < ac.ShaderInfos.Count; ++i)
                    {
                        AssetsConfig.ShaderInfo si = ac.ShaderInfos[i];
                        if (si.shader != null)
                        {
                            shaderFeatures.Clear();
                            for (int j = 0; j < ac.ShaderFeatures.Count; ++j)
                            {
                                AssetsConfig.ShaderFeature sf = ac.ShaderFeatures[j];
                                bool hasFeature = si.shaderFeatures.Exists((x) => { return sf.name == x; });
                                if (hasFeature)
                                    shaderFeatures.Add(sf.name);
                            }
                            si.shaderFeatures.Clear();
                            si.shaderFeatures.AddRange(shaderFeatures);
                        }
                    }
                }
            }
        }

        private void ShaderPropertyFeature(ShaderProperty sp, ref bool remove)
        {
            EditorGUILayout.BeginHorizontal(GUILayout.MaxWidth(400));
            EditorGUILayout.LabelField(sp.shaderProperty, GUILayout.MaxWidth(150));
            if (GUILayout.Button("Edit", GUILayout.MaxWidth(80)))
            {
                sp.folder = !sp.folder;
            }

            if (GUILayout.Button("Delete", GUILayout.MaxWidth(80)))
            {
                remove = true;
            }
            EditorGUILayout.EndHorizontal();
            if (sp.folder)
            {
                sp.shaderProperty = EditorGUILayout.TextField("PropertyName", sp.shaderProperty);
                sp.isTex = EditorGUILayout.Toggle("IsTex", sp.isTex);
                sp.shaderID = (int)(EShaderKeyID)EditorGUILayout.EnumPopup("", (EShaderKeyID)sp.shaderID, GUILayout.MaxWidth(200));
            }
        }
        private void SceneMatShaderTypeGUI(AssetsConfig ac)
        {
            ac.sceneMatInfoFolder = EditorGUILayout.Foldout(ac.sceneMatInfoFolder, "Mat Shader Type");
            if (ac.sceneMatInfoFolder)
            {
                EditorGUI.indentLevel++;
                GUILayout.Label("MatShaderMap", EditorStyles.boldLabel);
                if (GUILayout.Button("Add", GUILayout.MaxWidth(80)))
                {
                    ac.matShaderType.Add(new MatShaderType());
                }
                int removeIndex = -1;
                for (int i = 0; i < ac.matShaderType.Count; ++i)
                {
                    var mst = ac.matShaderType[i];
                    EditorGUILayout.BeginHorizontal(GUILayout.MaxWidth(400));
                    EditorGUILayout.LabelField(mst.name, GUILayout.MaxWidth(150));
                    if (GUILayout.Button("Edit", GUILayout.MaxWidth(80)))
                    {
                        mst.folder = !mst.folder;
                    }

                    if (GUILayout.Button("Delete", GUILayout.MaxWidth(80)))
                    {
                        removeIndex = i;
                        matInfoCopyIndex = -1;
                    }
                    if (matInfoCopyIndex == -1 || matInfoCopyIndex == i)
                    {
                        if (GUILayout.Button("Copy", GUILayout.MaxWidth(80)))
                        {
                            if (matInfoCopyIndex == -1)
                                matInfoCopyIndex = i;
                            else
                                matInfoCopyIndex = -1;
                        }
                    }
                    else
                    {
                        if (GUILayout.Button("Paste", GUILayout.MaxWidth(80)))
                        {
                            var src = ac.matShaderType[matInfoCopyIndex];
                            mst.Clone(src);
                        }
                    }

                    EditorGUILayout.EndHorizontal();
                    if (mst.folder)
                    {
                        mst.name = EditorGUILayout.TextField("Feature", mst.name);
                        mst.shader = EditorGUILayout.ObjectField("Shader", mst.shader, typeof(Shader), false) as Shader;
                        mst.macro = EditorGUILayout.TextField("Macro", mst.macro);
                        mst.matOffset = (ESceneMaterial)EditorGUILayout.EnumPopup("", (ESceneMaterial)mst.matOffset, GUILayout.MaxWidth(260));
                        mst.findPropertyType = (FindPropertyType)EditorGUILayout.EnumPopup("", mst.findPropertyType, GUILayout.MaxWidth(260));
                        mst.hasPbs = EditorGUILayout.Toggle("HasPbs", mst.hasPbs);
                        if (mst.hasPbs)
                            mst.pbsOffset = (uint)EditorGUILayout.IntSlider("PbsOffset", (int)mst.pbsOffset, 0, 1);
                        mst.hasCutout = EditorGUILayout.Toggle("HasCutout", mst.hasCutout);
                        mst.hasTransparent = EditorGUILayout.Toggle("HasTransparent", mst.hasTransparent);
                        mst.hasTransparentCout = EditorGUILayout.Toggle("HasTransparentCout", mst.hasTransparentCout);
                        if (mst.hasCutout || mst.hasTransparent || mst.hasTransparentCout)
                            mst.renderTypeOffset = (uint)EditorGUILayout.IntSlider("RenderTypeOffset", (int)mst.renderTypeOffset, 0, 2);
                        bool isPostprocess = mst.HasFlag(EMatFlag.IsPostProcess);
                        bool postProcess = EditorGUILayout.Toggle("IsPostProcess", isPostprocess);
                        if (postProcess != isPostprocess)
                        {
                            mst.SetFlag(EMatFlag.IsPostProcess, postProcess);
                        }

                        EditorGUI.indentLevel++;
                        if (GUILayout.Button("Add", GUILayout.MaxWidth(80)))
                        {
                            mst.shaderPropertys.Add(new ShaderProperty());
                        }
                        int subRemoveIndex = -1;
                        for (int j = 0; j < mst.shaderPropertys.Count; ++j)
                        {
                            var sp = mst.shaderPropertys[j];
                            bool remove = false;
                            ShaderPropertyFeature(sp, ref remove);
                            if (remove)
                            {
                                subRemoveIndex = j;
                            }
                        }
                        if (subRemoveIndex >= 0)
                        {
                            mst.shaderPropertys.RemoveAt(subRemoveIndex);
                        }
                        EditorGUI.indentLevel--;
                    }
                }
                if (removeIndex >= 0)
                {
                    ac.matShaderType.RemoveAt(removeIndex);
                }

                GUILayout.Label("CommonProperty", EditorStyles.boldLabel);

                if (GUILayout.Button("Add", GUILayout.MaxWidth(80)))
                {
                    ac.commonShaderProperty.Add(new ShaderProperty());
                }

                removeIndex = -1;
                for (int i = 0; i < ac.commonShaderProperty.Count; ++i)
                {
                    var sp = ac.commonShaderProperty[i];
                    bool remove = false;
                    ShaderPropertyFeature(sp, ref remove);
                    if (remove)
                    {
                        removeIndex = i;
                    }
                }
                if (removeIndex >= 0)
                {
                    ac.commonShaderProperty.RemoveAt(removeIndex);
                }

                GUILayout.Label("PropertyMap", EditorStyles.boldLabel);
                if (GUILayout.Button("Add", GUILayout.MaxWidth(80)))
                {
                    ac.shaderPropertyKey.Add(new ShaderProperty());
                }

                removeIndex = -1;
                for (int i = 0; i < ac.shaderPropertyKey.Count; ++i)
                {
                    var sp = ac.shaderPropertyKey[i];
                    bool remove = false;
                    ShaderPropertyFeature(sp, ref remove);
                    if (remove)
                    {
                        removeIndex = i;
                    }
                }
                if (removeIndex >= 0)
                {
                    ac.shaderPropertyKey.RemoveAt(removeIndex);
                }

                EditorGUI.indentLevel--;
            }
        }

        public override void OnInspectorGUI()
        {

            AssetsConfig ac = target as AssetsConfig;
            if (ac != null)
            {
                ConstValuesGUI(ac);
                TextureProcessGUI(ac);
                DummyMatGUI(ac);
                ShaderGroupGUI(ac);
                AssetsConfig.GetGroupedShaderFeatureList(groupedShaderFeatures);
                ShaderFeatureGUI(ac);
                ShaderInfoGUI(ac);
                SceneMatShaderTypeGUI(ac);
                if (GUILayout.Button("Save", GUILayout.MaxWidth(100)))
                {
                    CommonAssets.SaveAsset(ac);
                }
                OnEventProcessGUI(ac);
            }

            serializedObject.ApplyModifiedProperties();
            switch (opType)
            {
                case OpType.OpGenAllSceneMat:
                    GenAllSceneMat();
                    break;
                case OpType.OpRefreshAllSceneMat:
                    RefreshAllSceneMat();
                    break;
                case OpType.OpSaveAllSceneMat:
                    SaveAllSceneMat();
                    break;
                case OpType.OpGenMat:
                    GenMat();
                    break;
                case OpType.OpGenEffectMat:
                    GenEffectMat();
                    break;
                case OpType.OpRefreshMat:
                    RefreshMat();
                    break;
            }
            opType = OpType.None;
        }
        private void GenMat()
        {
            if (genMat != null)
            {
                MaterialShaderAssets.DefaultMat(genMat);
                genMat = null;
            }
        }
        private void RefreshMat()
        {
            if (genMat != null)
            {
                MaterialShaderAssets.DefaultRefeshMat(genMat);
                genMat = null;
            }
        }

        private void GenEffectMat()
        {
            if (genMat != null)
            {
                MaterialShaderAssets.DefaultEffectMat(genMat, multiBlend);
                genMat = null;
            }
        }
        private void GenAllSceneMat()
        {
            AssetsConfig ac = target as AssetsConfig;
            if (ac != null)
            {
                int matCount = (int)ESceneMaterial.Num;
                for (int i = 0; i < matCount; ++i)
                {
                    var dummyMaterialInfo = ac.sceneDummyMaterials[i];
                    EditorUtility.DisplayProgressBar(string.Format("{0}-{1}/{2}", "GenMat", i, matCount), dummyMaterialInfo.name, (float)i / matCount);

                    MaterialShaderAssets.DefaultMat(dummyMaterialInfo);
                }
                EditorUtility.ClearProgressBar();
            }
        }

        private void RefreshAllSceneMat()
        {
            AssetsConfig ac = target as AssetsConfig;
            if (ac != null)
            {
                int matCount = (int)ESceneMaterial.Num;
                for (int i = 0; i < matCount; ++i)
                {
                    var dummyMaterialInfo = ac.sceneDummyMaterials[i];
                    EditorUtility.DisplayProgressBar(string.Format("{0}-{1}/{2}", "RefreshMat", i, matCount), dummyMaterialInfo.name, (float)i / matCount);

                    MaterialShaderAssets.DefaultRefeshMat(dummyMaterialInfo);
                }
                EditorUtility.ClearProgressBar();
            }
        }
        private void SaveAllSceneMat()
        {
            AssetsConfig ac = target as AssetsConfig;
            if (ac != null)
            {
                string path = "Assets/BundleRes/Config/EffectData.asset";
                EffectData ed = AssetDatabase.LoadAssetAtPath<EffectData>(path);
                if (ed != null)
                {
                    ed.terrainOffset = (int)ESceneMaterial.TerrainChunk0;
                    List<Material> materials = new List<Material>();
                    int matCount = (int)ESceneMaterial.Num;
                    ed.matCount = matCount;
                    for (int i = 0; i < matCount; ++i)
                    {
                        var dummyMaterialInfo = ac.sceneDummyMaterials[i];
                        materials.Add(dummyMaterialInfo.mat);
                    }
                    for (int i = 0; i < matCount; ++i)
                    {
                        var dummyMaterialInfo = ac.sceneDummyMaterials[i];
                        materials.Add(dummyMaterialInfo.mat1);
                    }
                    ed.sceneMats = materials.ToArray();

                    materials.Clear();
                    matCount = (int)ECharMaterial.Num;
                    for (int i = 0; i < matCount; ++i)
                    {
                        var dummyMaterialInfo = ac.roleMaterials[i];
                        if ((dummyMaterialInfo.blendType & EBlendType.Opaque) != 0)
                            materials.Add(dummyMaterialInfo.mat);
                        if ((dummyMaterialInfo.blendType & EBlendType.Cutout) != 0)
                            materials.Add(dummyMaterialInfo.mat1);
                        if ((dummyMaterialInfo.blendType & EBlendType.CutoutTransparent) != 0)
                            materials.Add(dummyMaterialInfo.mat2);
                    }
                    ed.charMats = materials.ToArray();

                    materials.Clear();
                    matCount = (int)EEffectMaterial.Num;
                    for (int i = 0; i < matCount; ++i)
                    {
                        var dummyMaterialInfo = ac.effectMaterials[i];
                        materials.Add(dummyMaterialInfo.mat);
                    }
                    ed.effectMats = materials.ToArray();

                    CommonAssets.SaveAsset(ed);
                }
            }

        }
    }
}
