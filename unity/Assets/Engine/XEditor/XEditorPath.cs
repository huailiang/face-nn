#if UNITY_EDITOR
using System;
using UnityEditor;

namespace XEditor
{
	public class XEditorPath
	{
        public static readonly string Sce = "Assets/XScene/";
        public static readonly string Skp = "Assets/BundleRes/SkillPackage/";
        public static readonly string Crv = "Assets/Editor/EditorResources/Curve/";
        public static readonly string San = "Assets/Editor/EditorResources/Server/Animation/";
        public static readonly string Scb = "Assets/BundleRes/Table/SceneBlock/";

        private static readonly string _root = "Assets/BundleRes";
        private static readonly string _editor_root = "Assets/Editor";
        private static readonly string _editor_res_root = "Assets/Editor/EditorResources";

        public static string GetCfgFromSkp(string skp, string suffix = ".config")
        {
            skp = skp.Replace("/BundleRes/", "/Editor/EditorResources/");
            int m = skp.LastIndexOf('.');

            return skp.Substring(0, m) + suffix;
        }

        private static void RootPath()
        {
            if (!System.IO.Directory.Exists(_root))
            {
                AssetDatabase.CreateFolder("Assets", "BundleRes");
            }
        }

        private static void EditorRootPath()
        {
            if (!System.IO.Directory.Exists(_editor_root))
            {
                AssetDatabase.CreateFolder("Assets", "Editor");
            }

            if (!System.IO.Directory.Exists(_editor_res_root))
            {
                AssetDatabase.CreateFolder("Assets/Editor", "EditorResources");
            }
        }

        public static string BuildPath(string dictionary, string root)
        {
            string[] splits = dictionary.Split('/');
            string _base = root;

            foreach (string s in splits)
            {
                string path = _base + "/" + s + "/";

                if (!System.IO.Directory.Exists(path))
                {
                    AssetDatabase.CreateFolder(_base, s);
                }

                _base = path.Substring(0, path.Length - 1);
            }

            return _base + "/";
        }

        public static string GetEditorBasedPath(string dictionary)
        {
            EditorRootPath();

            return BuildPath(dictionary, _editor_res_root);
        }

        public static string GetPath(string dictionary)
        {
            RootPath();

            return BuildPath(dictionary, _root);
        }
	}
}
#endif