using UnityEditor;
using UnityEngine;

namespace XEngine.Editor
{
    public class PropertyDrawer
    {
        static void DrawSlider(AssetsConfig.ShaderCustomProperty scp, ref Vector4 value, int index)
        {
            if (scp.valid)
            {
                if (!string.IsNullOrEmpty(scp.desc))
                {
                    if (scp.min < scp.max)
                    {
                        EditorGUILayout.BeginHorizontal();
                        value[index] = EditorGUILayout.Slider(string.Format("{0}({1}-{2})", scp.desc, scp.min, scp.max), value[index], scp.min, scp.max);
                        if (GUILayout.Button("R", GUILayout.MaxWidth(20)))
                        {
                            value[index] = scp.defaultValue;
                        }
                        EditorGUILayout.EndHorizontal();
                    }
                }
                else
                {
                    value[index] = scp.defaultValue;
                }
            }
        }

        public static void OnGUI(MaterialProperty prop, MaterialEditor editor, AssetsConfig.ShaderFeature sf)
        {
            EditorGUILayout.LabelField(sf.name, EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            Vector4 value = prop.vectorValue;
            EditorGUI.BeginChangeCheck();
            for (int i = 0; i < sf.customProperty.Length; ++i)
            {
                var scp = sf.customProperty[i];
                DrawSlider(scp, ref value, i);
            }
            if (EditorGUI.EndChangeCheck())
            {
                prop.vectorValue = value;
            }
            EditorGUI.indentLevel--;
        }

        public static void OnGUI(MaterialProperty prop, MaterialEditor editor, AssetsConfig.ShaderCustomProperty scp, int customIndex)
        {
            Vector4 value = prop.vectorValue;
            EditorGUI.BeginChangeCheck();
            DrawSlider(scp, ref value, customIndex);
            if (EditorGUI.EndChangeCheck())
            {
                prop.vectorValue = value;
            }
        }
    }
}