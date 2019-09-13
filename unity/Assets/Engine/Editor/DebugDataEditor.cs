using System;
using System.Collections.Generic;
using System.IO;
using System.Linq.Expressions;
using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;
using XEditor;

namespace CFEngine.Editor
{
    [CustomEditor (typeof (DebugData))]
    public class DebugDataEditor : BaseEditor<DebugData>
    {
        private static Dictionary<string, string> pathMap = null;
        public void OnEnable ()
        {
            LoadPathMap ();
        }

        public override void OnInspectorGUI ()
        {
            serializedObject.Update ();
            EditorGUI.BeginChangeCheck ();
            // if (GUILayout.Button("RefreshFashion", GUILayout.MaxWidth(160)))
            // {
            //      var fashionInfo = XFashionLibrary.FashionsInfo;
            //     if (fashionInfo != null)
            //     {
            //         pathMap.Clear();
            //         for (int i = 0; i < fashionInfo.Length; i++)
            //         {
            //             FashionExportWindow.ExportSuit(fashionInfo[i], pathMap);
            //         }
            //         SavePathMap();
            //     }
            // }
            if (GUILayout.Button ("Save", GUILayout.MaxWidth (160)))
            {
                Save((target as DebugData).mat);
            }

            if (EditorGUI.EndChangeCheck ())
            {

            }
            serializedObject.ApplyModifiedProperties ();
        }

        public static void Save(Material material)
        {
            if (material == null)
                return;

            var fashionInfo = XFashionLibrary.FashionsInfo;
            if (fashionInfo != null)
            {
                pathMap.Clear();
                for (int i = 0; i < fashionInfo.Length; i++)
                {
                    FashionExportWindow.ExportSuit(fashionInfo[i], pathMap);
                }
                SavePathMap();
            }

            CommonAssets.SaveAsset(material);
            string path = AssetDatabase.GetAssetPath(material);
            string srcPath;
            if (pathMap.TryGetValue(path, out srcPath))
            {
                CommonAssets.CreateAsset<Material>(srcPath, ".mat", material);
            }
            else
            {
                FileInfo fi = new FileInfo(path);
                string dir = fi.DirectoryName;
                srcPath = string.Format(AssetsConfig.GlobalAssetsConfig.Creature_Material_Format_Path,
                    AssetsConfig.GlobalAssetsConfig.Creature_Path,
                    dir,
                    dir + "_bandpose" + fi.Name);
                if (File.Exists(srcPath))
                {
                    CommonAssets.CreateAsset<Material>(srcPath, ".mat", material);
                }

            }
        }

        private static void SavePathMap ()
        {
            string path = "Assets/BundleRes/EditorSceneRes/partMatPathMap.partbytes";
            using (FileStream fs = new FileStream (path, FileMode.Create))
            {
                BinaryWriter bw = new BinaryWriter (fs);
                bw.Write (pathMap.Count);
                var it = pathMap.GetEnumerator ();
                while (it.MoveNext ())
                {
                    var kvp = it.Current;
                    bw.Write (kvp.Key);
                    bw.Write (kvp.Value);
                }

            }
        }

        public static void LoadPathMap ()
        {
            if (pathMap == null)
                pathMap = new Dictionary<string, string> ();
            pathMap.Clear ();
            string path = "Assets/BundleRes/EditorSceneRes/partMatPathMap.partbytes";
            if (File.Exists (path))
            {
                using (FileStream fs = new FileStream (path, FileMode.Open))
                {
                    BinaryReader br = new BinaryReader (fs);
                    int count = br.ReadInt32 ();
                    for (int i = 0; i < count; ++i)
                    {
                        string key = br.ReadString ();
                        string value = br.ReadString ();
                        pathMap[key] = value;
                    }
                }
            }
        }
    }
}
