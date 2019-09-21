using UnityEngine;
using UnityEditor;
using System.IO;

public class HelperEditor : MonoBehaviour
{

    public static string basepath
    {
        get
        {
            string path = Application.dataPath;
            path = path.Remove(path.IndexOf("/Assets"));
            return path;
        }
    }

    [MenuItem("Help/IO/OpenCacheDiectory")]
    public static void OpenCacheDiectory()
    {
        Open(Application.temporaryCachePath);
    }

    [MenuItem("Help/IO/OpenPersistDirectory")]
    public static void OpenPersistDirectory()
    {
        Open(Application.persistentDataPath);
    }


    [MenuItem("Help/IO/OpenShellDirectory")]
    public static void OpenAssetbundle()
    {
        Open(basepath + "/Shell");
    }

    [MenuItem("Help/IO/OpenUnityInstallDirectory")]
    public static void OpenUnityDir()
    {
        Open(EditorApplication.applicationContentsPath);
    }


    [MenuItem("Help/RestartUnity")]
    private static void RestartUnity()
    {
#if UNITY_EDITOR_WIN
        string install = Path.GetDirectoryName(EditorApplication.applicationContentsPath);
        string path = Path.Combine(install, "Unity.exe");
        string[] args = path.Split('\\');
        System.Diagnostics.Process po = new System.Diagnostics.Process();
        Debug.Log("install: " + install + " path: " + path);
        po.StartInfo.FileName = path;
        po.Start();

        System.Diagnostics.Process[] pro = System.Diagnostics.Process.GetProcessesByName(args[args.Length - 1].Split('.')[0]);//Unity
        foreach (var item in pro)
        {
            item.Kill();
        }
#endif
    }

    public static void Open(string path)
    {
        if (File.Exists(path))
        {
            path = Path.GetDirectoryName(path);
        }
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
#if UNITY_EDITOR_OSX
        string shell = basepath + "/Shell/open.sh";
        string arg = path;
        string ex = shell + " " + arg;
        System.Diagnostics.Process.Start("/bin/bash", ex);
#elif UNITY_EDITOR_WIN
        path = path.Replace("/", "\\");
        System.Diagnostics.Process.Start("explorer.exe", path);
#endif
    }
}
