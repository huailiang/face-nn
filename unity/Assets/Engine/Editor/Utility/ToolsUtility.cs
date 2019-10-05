using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;

namespace XEngine.Editor
{

    public class ToolsUtility
    {

        public static void BeginGroup(string name, bool beginHorizontal = true)
        {
            BeginGroup(name, new Vector4Int(0, 0, 1000, 100), beginHorizontal);
        }

        public static void BeginGroup(string name, Vector4Int minMax, bool beginHorizontal)
        {
            if (beginHorizontal)
                GUILayout.BeginHorizontal();
            EditorGUILayout.BeginVertical(GUI.skin.box,
                GUILayout.MinWidth(minMax.x),
                GUILayout.MinHeight(minMax.y),
                GUILayout.MaxWidth(minMax.z),
                GUILayout.MaxHeight(minMax.w));
            if (!string.IsNullOrEmpty(name))
                EditorGUILayout.LabelField(name, EditorStyles.boldLabel);
        }

        public static void EndGroup(bool endHorizontal = true)
        {
            EditorGUILayout.EndVertical();
            if (endHorizontal)
                GUILayout.EndHorizontal();
        }

        public static bool BeginFolderGroup(string name, ref bool folder)
        {
            folder = EditorGUILayout.Foldout(folder, name);
            if (folder)
            {
                ToolsUtility.BeginGroup("");

            }
            return folder;
        }

        public static void EndFolderGroup()
        {
            ToolsUtility.EndGroup();
        }
    }
}