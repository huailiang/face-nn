using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;
using XEditor;

[CustomEditor(typeof(XRoleParts))]
public class XPartsEditor : Editor
{

    private XRoleParts part;

    private void OnEnable()
    {
        part = target as XRoleParts;
    }


    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Debug"))
        {
            SkinnedMeshRenderer[] sks = part.gameObject.GetComponentsInChildren<SkinnedMeshRenderer>();
            for (int i = 0; i < sks.Length; i++)
            {
                var item = sks[i];
                var bs = item.bones;
                Debug.Log(string.Format("************** {0} ***************", item.name));
                for (int j = 0; j < bs.Length; j++)
                {
                    Debug.Log(j + ": " + bs[j].name);
                }
                if (item.name.Contains("hair"))
                {
                    var weights = sks[i].sharedMesh.boneWeights;
                    foreach (var it in weights)
                    {
                        Debug.Log(it);
                    }
                }
            }
        }
        if (GUILayout.Button("Face Weights"))
        {
            XInterfaceMgr.singleton.AttachInterface<IResourceHelp>(XInterfaceMgr.ResourceHelperID, XResources.singleton);
            SkinnedMeshRenderer[] sks = part.gameObject.GetComponentsInChildren<SkinnedMeshRenderer>();
            for (int i = 0; i < sks.Length; i++)
            {
                var item = sks[i];
                if (item.name.Contains("face"))
                {
                    if (sks[i].sharedMesh != null)
                    {
                        if (sks[i].sharedMesh != null)
                        {
                            var weights = sks[i].sharedMesh.boneWeights;
                            var matrix = sks[i].sharedMesh.bindposes;
                            Debug.Log("weights len：" + weights.Length + " bind: " + matrix.Length);
                        }
                    }
                }
                string name = part.gameObject.name;
                if (name.Contains("_male"))
                    FashionExportWindow.MakeKnead(RoleShape.MALE, part);
                else if (name.Contains("_female"))
                    FashionExportWindow.MakeKnead(RoleShape.FEMALE, part);
                else if (name.Contains("_tall"))
                    FashionExportWindow.MakeKnead(RoleShape.TALL, part);
                else if (name.Contains("_giant"))
                    FashionExportWindow.MakeKnead(RoleShape.GIANT, part);
            }
        }
        if (GUILayout.Button("Bind Face"))
        {
            SkinnedMeshRenderer[] sks = part.gameObject.GetComponentsInChildren<SkinnedMeshRenderer>();
            for (int i = 0; i < sks.Length; i++)
            {
                var item = sks[i];
                if (item.name.Contains("face"))
                {
                    Debug.Log(item.bones.Length);
                    part.face = item.bones;
                }
            }
        }
        GUILayout.EndHorizontal();
    }
}
