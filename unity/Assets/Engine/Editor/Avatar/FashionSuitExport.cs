using CFUtilPoolLib;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using CFEngine.Editor;
/// <summary>
/// 美术工具 
/// 导出套装材质和mesh,但不会计算保存骨骼
/// </summary>


namespace XEditor
{
    public class FashionSuitExport : EditorWindow
    {
        private int suit_select = 0;
        private FashionSuit.RowData[] fashionInfo;
        private string[] fashionDesInfo;
        private GameObject go;

        [MenuItem("Tools/FashionSuitExport")]
        static void AnimExportTool()
        {
            if (EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
            {
                FashionSuitExport window = (FashionSuitExport)EditorWindow.GetWindow(typeof(FashionSuitExport));
                window.Show();
            }
        }

        void OnGUI()
        {
            GUILayout.Label(XEditorUtil.Config.suit_shape, XEditorUtil.titleLableStyle);

            if (fashionDesInfo == null || fashionInfo == null)
            {
                fashionInfo = XFashionLibrary.FashionsInfo;
                fashionDesInfo = new string[fashionInfo.Length];
                for (int i = 0; i < fashionInfo.Length; i++)
                {
                    fashionDesInfo[i] = fashionInfo[i].name;
                }
            }

            GUILayout.Space(8);
            GUILayout.Label("Select Suit");
            suit_select = EditorGUILayout.Popup(suit_select, fashionDesInfo);

            FashionSuit.RowData row = fashionInfo[suit_select];
            GUILayout.Label("Player_" + ((RoleShape)row.shape).ToString().ToLower() + "_" + row.dir);
            GUILayout.Space(8);

            if (GUILayout.Button("Export"))
            {
                Export();
            }
            if (GUILayout.Button("Preview"))
            {
                Preview();
            }

            GUILayout.Space(16);
            if (GUILayout.Button("Open Main Scene"))
            {
                XEditorUtil.ClearCreatures();
            }

        }


        void Export()
        {
            FashionSuit.RowData row = fashionInfo[suit_select];
            XDebug.singleton.AddLog("export: " + row.dir + " shape: " + row.shape);
            FashionExportWindow.ExportSuit(row);
            AssetDatabase.Refresh();
            XEditorUtil.ClearCreatures();
            EditorUtility.DisplayDialog("tip", "job done", "ok");
        }


        void Preview()
        {
            FashionSuit.RowData row = fashionInfo[suit_select];
            if (go != null) GameObject.DestroyImmediate(go);
            string path = "Assets/BundleRes/Prefabs/Player_" + ((RoleShape)row.shape).ToString().ToLower() + ".prefab";
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (prefab != null)
            {
                go = Instantiate(prefab);
                go.name = prefab.name;
                go.transform.localScale = Vector3.one;
                go.transform.rotation = Quaternion.Euler(0, 180, 0);
                go.transform.position = new Vector3(106, 5f, 146);
                Selection.activeGameObject = go;
                uint presentid = uint.Parse(row.weapon[0, 0]);
                FashionUtil.DrawSuit(go, row, presentid, 1);
            }
        }

    }


    /// <summary>
    /// export by part 
    /// required by art(wushuang)
    /// </summary>
    public class SuitExport
    {
        static string dest_dir { get { return "Assets/BundleRes/FBXRawData/"; } }

        [MenuItem("Assets/Fashion/Fbx_Export")]
        static void Export()
        {
            object[] objs = Selection.GetFiltered<object>(SelectionMode.Unfiltered | SelectionMode.DeepAssets);
            for (int i = 0; i < objs.Length; i++)
            {
                if (objs[i] is GameObject)
                {
                    GameObject go = objs[i] as GameObject;
                    if (!go.name.Contains("_bandpose"))
                    {
                        Export(go);
                    }
                }
            }

            AssetDatabase.Refresh();
            EditorUtility.DisplayDialog("TIP", "JOB DONE", "ok");
        }


        static void Export(GameObject obj)
        {
            string path = AssetDatabase.GetAssetPath(obj);
            Debug.Log(path);
            if (path.EndsWith("FBX"))
            {
                if (obj.name.Contains("_face_"))
                {
                    ExportFace(obj);
                }
                else if (obj.name.Contains("_weapon_"))
                {
                    ExportWeapon(obj);
                }
                else if (obj.name.Contains("_hair_"))
                {
                    ExportHair(obj);
                }
                else if (obj.name.Contains("_wing"))
                {
                    ExportWing(obj);
                }
                else if (obj.name.StartsWith("Player_"))
                {
                    ExportMainBody(obj);
                }
            }
        }

        static void CulCommon(GameObject obj, out string cname, out string origin, out string matdir, out RoleShape shape)
        {
            shape = ParseShape(obj.name);
            origin = string.Empty;
            cname = obj.name;
            matdir = GetMatDir(obj, out origin);
        }

        public static void ExportFace(GameObject obj)
        {
            string cname, origin, matdir;
            RoleShape shape;
            CulCommon(obj, out cname, out origin, out matdir, out shape);
            string target = dest_dir + "Player_" + shape.ToString().ToLower() + "/Common/";
            uint presentid = 1;
            string dir = cname.Substring(cname.LastIndexOf('_') + 1);
            var table = XFashionLibrary._profession.Table;
            for (int i = 0; i < table.Length; i++)
            {
                if (!string.IsNullOrEmpty(table[i].face) && table[i].face == dir)
                {
                    presentid = table[i].PresentID;
                    break;
                }
            }
            string cname2 = "Player_" + shape.ToString().ToLower() + "_common_face_" + presentid;
            Transform[] bones;
            FashionExportWindow.HandleRolePart(origin, target, matdir, cname, cname2, out bones, null);
            ClearObjs();
        }


        static void ExportWeapon(GameObject obj)
        {
            string cname, origin, matdir;
            RoleShape shape;
            CulCommon(obj, out cname, out origin, out matdir, out shape);

            string weapon_name = cname.Substring(cname.LastIndexOf('_') + 1);
            string dir = matdir.Substring(matdir.LastIndexOf('_') + 1);
            dir = dir.Remove(dir.Length - 1);
            string sshape = shape.ToString().ToLower();
            string target = dest_dir + "Player_" + sshape + "/Player_" + sshape + "_" + dir + "/";
            var suit = XFashionLibrary.FashionsInfo;
            int weaponindex = 1;
            uint presentid = 1;
            for (int i = 0; i < suit.Length; i++)
            {
                if (suit[i].dir == dir)
                {
                    var weapon = suit[i].weapon;
                    for (int j = 0; j < weapon.Count; j++)
                    {
                        if (weapon[j, 1] == weapon_name)
                        {
                            weaponindex = 1;
                            uint.TryParse(weapon[j, 0], out presentid);
                            break;
                        }
                        if (weapon[j, 2] == weapon_name)
                        {
                            weaponindex = 2;
                            uint.TryParse(weapon[j, 0], out presentid);
                            break;
                        }
                    }
                    break;
                }
            }
            Transform[] bones;
            string cname2 = "Player_" + shape.ToString().ToLower() + "_" + dir + "_weapon_" + presentid + "_" + weaponindex;
            FashionExportWindow.HandleRolePart(origin, target, matdir, cname, cname2, out bones, null);
            ClearObjs();
        }


        static void ExportHair(GameObject obj)
        {
            string cname, origin, matdir;
            RoleShape shape;
            CulCommon(obj, out cname, out origin, out matdir, out shape);

            if (cname.Contains("_common_"))
            {
                string target = dest_dir + "Player_" + shape.ToString().ToLower() + "/Common/";
                string cname2 = cname;
                Transform[] bones;
                FashionExportWindow.HandleRolePart(origin, target, matdir, cname, cname2, out bones, null);
            }
            else
            {
                Debug.LogError("Process is not handle nocommon hair");
            }
            ClearObjs();
        }

        static void ExportWing(GameObject obj)
        {
            string cname, origin, matdir;
            RoleShape shape;
            CulCommon(obj, out cname, out origin, out matdir, out shape);

            string target = dest_dir + "Player_common/";
            Transform[] bones;
            FashionExportWindow.HandleRolePart(origin, target, matdir, cname, cname, out bones, null);
            ClearObjs();
        }


        static void ExportMainBody(GameObject obj)
        {
            string cname, origin, matdir;
            RoleShape shape;
            CulCommon(obj, out cname, out origin, out matdir, out shape);

            string[] sp = cname.Split('_');
            if (sp.Length == 3) //valid
            {
                string dir = sp[2];
                string sshape = shape.ToString().ToLower();
                string target_dir = dest_dir + "Player_" + sshape + "/Player_" + sshape + "_" + dir + "/";
                Material newMat = null;
                var skins = obj.GetComponentsInChildren<SkinnedMeshRenderer>();
                for (int i = 0; i < skins.Length; i++)
                {
                    string name = skins[i].gameObject.name;
                    if (name.Contains("_helmet"))
                    {
                        Mesh newMesh = null;
                        SkinnedMeshRenderer skin = skins[i].gameObject.GetComponent<SkinnedMeshRenderer>();
                        FBXAssets.GetMeshMat(skin, out newMesh, out newMat);
                        if (newMesh != null)
                        {
                            XEditorUtil.ClearMesh(newMesh);
                            AssetDatabase.CreateAsset(newMesh, target_dir + name + ".asset");
                            AssetDatabase.CopyAsset(origin + matdir + name + ".mat", target_dir + name + ".mat");
                        }
                    }
                    else if (name.Contains("_body"))
                    {
                        Mesh newMesh = null;
                        SkinnedMeshRenderer skin = skins[i].gameObject.GetComponent<SkinnedMeshRenderer>();
                        FBXAssets.GetMeshMat(skin, out newMesh, out newMat);
                        if (newMesh != null)
                        {
                            XEditorUtil.ClearMesh(newMesh);
                            AssetDatabase.CreateAsset(newMesh, target_dir + name + ".asset");
                            AssetDatabase.CopyAsset(origin + matdir + name + ".mat", target_dir + name + ".mat");
                        }
                    }
                }
            }
            else //invalid
            {
                Debug.Log("not realize part with name:  " + obj.name);
            }
        }


        static RoleShape ParseShape(string name)
        {
            RoleShape shape = RoleShape.MALE;
            name = name.ToLower();
            if (name.Contains("female"))
            {
                shape = RoleShape.FEMALE;
            }
            else if (name.Contains("male"))
            {
                shape = RoleShape.MALE;
            }
            else if (name.Contains("giant"))
            {
                shape = RoleShape.GIANT;
            }
            else if (name.Contains("tall"))
            {
                shape = RoleShape.TALL;
            }
            return shape;
        }


        static string GetMatDir(GameObject obj, out string origindir)
        {
            string asset = AssetDatabase.GetAssetPath(obj);
            int index = asset.LastIndexOf('/');
            string parent = asset.Substring(0, index);
            origindir = parent + "/";
            string[] folders = AssetDatabase.GetSubFolders(parent);
            for (int i = 0; i < folders.Length; i++)
            {
                index = folders[i].LastIndexOf('/');
                string name = folders[i].Substring(index + 1);
                if (name.StartsWith("Materials_"))
                {
                    return name + "/";
                }
            }
            return asset;
        }

        static void ClearObjs()
        {
            var objs = GameObject.FindGameObjectsWithTag("EditorOnly");
            foreach (var item in objs)
            {
                GameObject.DestroyImmediate(item);
            }
        }

    }
}