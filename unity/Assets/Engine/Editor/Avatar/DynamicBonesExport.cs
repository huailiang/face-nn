using UnityEditor;
using UnityEngine;

/// <summary>
///  动态骨骼导出工具
/// </summary>

public class DynamicBonesExport : EditorWindow
{

    [MenuItem("GameObject/DynamicBonesExportTool %1")]
    static void DynamicBonesExportTool()
    {
        Object obj = Selection.activeObject;
        if (obj != null)
        {
            GameObject go = obj as GameObject;
            Transform top = XEditorFullpath.GetTopTransform(go);
            GameObject nobj = new GameObject(top.name + "_" + go.name);
            nobj.transform.localPosition = go.transform.localPosition;
            nobj.transform.localRotation = go.transform.localRotation;
            nobj.transform.localScale = go.transform.localScale;

            GameObject ngo = GameObject.Instantiate(go);
            ngo.transform.SetParent(nobj.transform);
            ngo.transform.localPosition = Vector3.zero;
            ngo.transform.localRotation = Quaternion.identity;
            ngo.transform.localScale = Vector3.one;
            DynamicBone dynamicBone= ngo.AddComponent<DynamicBone>();
            dynamicBone.m_Root=ngo.transform;
            dynamicBone.m_UpdateRate=30;
            ngo.name = obj.name;

            GameObject output = PrefabUtility.CreatePrefab("Assets/BundleRes/Prefabs/DynamicBones/" + nobj.name + ".prefab", nobj);
            Selection.activeGameObject = output;
            GameObject.DestroyImmediate(nobj);
            EditorUtility.DisplayDialog("tip", " job done", "ok");
            AssetDatabase.Refresh();
        }
        else
        {
            EditorUtility.DisplayDialog("error", " you did not select anything", "ok");
        }
    }



    [MenuItem("Assets/Fashion/DynamicBonesPath")]
    static void DynamicBonesPath()
    {
        Object obj = Selection.activeObject;
        if (obj != null)
        {
            GameObject go = obj as GameObject;
            string[] arr = go.name.Split('_');
            bool iscommon = go.name.Contains("_common_");
            int index = 0;

            if (arr.Length > 3)
            {
                if (!iscommon)
                {
                    index += arr[0].Length + arr[1].Length + arr[2].Length + 2;
                    string dir = arr[0] + "_" + arr[1];
                    string fbx = go.name.Substring(0, index);
                    string post = go.name.Substring(index + 1);
                    HandleDBones(dir, fbx, post, false);
                }
                else
                {
                    index += arr[0].Length + arr[1].Length + arr[2].Length + arr[3].Length + arr[4].Length + 4;
                    string dir = arr[0] + "_" + arr[1] + "/Common";
                    string fbx = go.name.Substring(0, index);
                    string post = go.name.Substring(index + 1);
                    HandleDBones(dir, fbx, post, true);
                }
            }
            else
            {
                EditorUtility.DisplayDialog("error", " you select error object!", "ok");
            }
        }
        else
        {
            EditorUtility.DisplayDialog("error", " you did not select anything", "ok");
        }
    }



    static void HandleDBones(string dir,string fbx,string post,bool iscommon)
    {
        string path = "Assets/Creatures/" + dir + "/" + fbx + "/" + fbx + ".FBX";
        if(iscommon)
        {
            path = "Assets/Creatures/" + dir + "/" + fbx + ".FBX";
        }
        GameObject fbxObj = AssetDatabase.LoadAssetAtPath<GameObject>(path);
        Transform tf = SearchChild(fbxObj.transform, post);
        if (tf != null)
        {
            string outp = string.Empty;
            while (tf.parent != null && tf.parent.name != "root")
            {
                tf = tf.parent;
                outp = tf.name + "/" + outp;
            }
            if (!string.IsNullOrEmpty(outp))
            {
                outp = "root/" + outp.Remove(outp.Length - 1);
                Debug.Log(outp);
                EditorUtility.DisplayDialog("tip", outp, "ok");
            }
        }
    }



    static Transform SearchChild(Transform tf, string name)
    {
        if (tf.name == name) return tf;
        int cnt = tf.childCount;
        for (int i = 0; i < cnt; i++)
        {
            var child = tf.GetChild(i);
            var childtf = SearchChild(child, name);
            if (childtf != null) return childtf;
        }
        return null;
    }

}
