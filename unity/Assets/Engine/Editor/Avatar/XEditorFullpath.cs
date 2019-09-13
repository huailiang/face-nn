using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;


public class XEditorFullpath : Editor
{

    [MenuItem("GameObject/AbsolutelyPath", priority = 2)]
    static void GetFullPath()
    {
        var go = Selection.activeGameObject;
        XDebug.singleton.AddGreenLog(go.name);
        string path = go.name;
        var transf = go.transform;
        while (transf.parent != null)
        {
            transf = transf.parent;
            path = transf.name + "/" + path;
        }
        XDebug.singleton.AddLog("AbsolutelyPath: " + path);
    }


    public static string GetFullPath(Transform transf)
    {
        string path = transf.name;
        while (transf.parent != null)
        {
            transf = transf.parent;
            path = transf.name + "/" + path;
        }
        return path;
    }

    public static string GetRootFullPath(Transform tf)
    {
        string path = tf.name;
        while (tf.parent != null && tf.name != "root")
        {
            tf = tf.parent;
            path = tf.name + "/" + path;
        }
        return path;
    }


    public static Transform GetTopTransform(GameObject go)
    {
        XDebug.singleton.AddGreenLog(go.name);
        string path = go.name;
        var transf = go.transform;
        while (transf.parent != null)
        {
            transf = transf.parent;
        }
        return transf;
    }

    //[MenuItem("Assets/AssetsPath")]
    static void OutputPath()
    {
        var selects = Selection.GetFiltered(typeof(object), SelectionMode.Unfiltered | SelectionMode.DeepAssets);
        for (int i = 0; i < selects.Length; i++)
        {
            var path = AssetDatabase.GetAssetPath(selects[i]);
            Debug.Log(path + "  type: " + selects[i].GetType());
        }
    }

}
