using UnityEngine;
using UnityEditor;
using System;
using System.IO;
using System.Collections.Generic;
namespace CFEngine.Editor
{
    internal class AssetExportConfig : EditorWindow
    {
        private Action endCb = null;
        public static void ShowConfig(Action endCb)
        {
            FBXAssets.export = false;
            AssetExportConfig window = EditorWindow.GetWindow<AssetExportConfig>(false);
            window.endCb = endCb;
            window.Show();
        }

        void OnGUI()
        {
            GUILayout.BeginHorizontal();
            FBXAssets.removeUV2 = GUILayout.Toggle(FBXAssets.removeUV2, "Remove UV2", GUILayout.MaxWidth(200));
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            FBXAssets.removeColor = GUILayout.Toggle(FBXAssets.removeColor, "Remove Color", GUILayout.MaxWidth(200));
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            FBXAssets.isNotReadable = GUILayout.Toggle(FBXAssets.isNotReadable, "Is Not Readable", GUILayout.MaxWidth(200));
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Ok", GUILayout.MaxWidth(160)))
            {
                FBXAssets.export = true;
                this.Close();
                if (endCb != null)
                {
                    endCb();
                }
            }
            if (GUILayout.Button("Cancel", GUILayout.MaxWidth(160)))
            {
                FBXAssets.export = false;
                this.Close();
            }
            GUILayout.EndHorizontal();
        }
    }
}