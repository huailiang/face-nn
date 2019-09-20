#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using UnityEngine;
using CFUtilPoolLib;
using UnityEditor;
using System.IO;

class TypeSpecializer<T>
{
    static string suffix = string.Empty;
    public static string Suffix { get { return suffix; } }

    static TypeSpecializer()
    {
        TypeSpecializer<AnimationClip>.suffix = ".anim";
        TypeSpecializer<GameObject>.suffix = ".prefab";
    }
}

public class XResources : XSingleton<XResources>, IResourceHelp
{
    public void CheckResource(UnityEngine.Object o, string path) { }

    public UnityEngine.Object LoadEditorResource(string path, string suffix, Type t)
    {
        return AssetDatabase.LoadAssetAtPath(path + suffix, t);
    }

    public static T LoadEditorResourceAtBundleRes<T>(string path, bool bInstantiate = false) where T : UnityEngine.Object
    {
        T o = AssetDatabase.LoadAssetAtPath<T>("Assets/BundleRes/" + path + GetSuffix<T>());
        if (bInstantiate)
            return UnityEngine.GameObject.Instantiate<T>(o, null);
        return o;
    }

    public bool Deprecated { get; set; }

    static string GetSuffix<T>() where T : UnityEngine.Object
    {
        return TypeSpecializer<T>.Suffix;
    }

    public static void LoadAllAssets(string folder, List<UnityEngine.Object> outputFiles)
    {
        DirectoryInfo direction = new DirectoryInfo(folder);
        FileSystemInfo[] fs = direction.GetFileSystemInfos();

        for (int i = 0; i < fs.Length; i++)
        {
            if (fs[i] is DirectoryInfo)
            {
                LoadAllAssets(fs[i].FullName, outputFiles);
            }
            else if (fs[i] is FileInfo)
            {
                if (fs[i].FullName.EndsWith(".meta")) continue;
                int index = fs[i].FullName.IndexOf("Assets\\");
                string path = fs[i].FullName.Substring(index).Replace('\\', '/');

                var obj = AssetDatabase.LoadMainAssetAtPath(path);
                if (obj != null)
                    outputFiles.Add(obj);
            }
        }
    }
}
#endif