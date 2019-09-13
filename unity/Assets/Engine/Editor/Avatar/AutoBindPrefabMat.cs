using CFEngine.Editor;
using UnityEditor;
using UnityEngine;

public class AutoBindPrefabMat
{

    /// <summary>
    /// tools for meiyikai
    /// </summary>

    [MenuItem("Assets/Fashion/BindPrefabMat")]
    static void Start()
    {
        CommonAssets.enumPrefab.cb = (GameObject go, string path) =>
              {
                  SkinnedMeshRenderer[] renders = go.GetComponentsInChildren<SkinnedMeshRenderer>();
                  if (renders != null)
                  {
                      foreach (var item in renders)
                      {
                          if (item.sharedMesh != null)
                          {
                              string npath = AssetDatabase.GetAssetPath(item.sharedMesh);

                              npath = npath.Replace(".asset", ".mat");
                              var mat = AssetDatabase.LoadAssetAtPath<Material>(npath);
                              if (mat != null)
                              {
                                  item.sharedMaterial = mat;
                              }
                              else
                              {
                                  Debug.Log("mat not found, " + npath);
                              }
                          }
                      }
                      AssetDatabase.Refresh();
                  }
              };
        CommonAssets.EnumAsset<GameObject>(CommonAssets.enumPrefab, "BindMaterial");
    }


}
