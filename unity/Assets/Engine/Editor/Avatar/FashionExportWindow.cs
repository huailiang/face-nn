using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;
using System.IO;
using System.Collections.Generic;
using CFEngine.Editor;

/// <summary>
/// 时装导出工具
/// 按照体型导出
/// 1， 导出套装的mat&mesh
/// 2,  导出翅膀prefab
/// 3， 计算体型相关的基础骨骼和动态骨骼
/// 4， 导出基础体型prefab
/// 5,  预处理mesh & skinedmeshrender
/// ....
/// </summary>

namespace XEditor
{
    public class XStrWeapon
    {
        public uint presentid;
        public string[] weapon1;
        public string[] weapon2;
    }


    public class XStrRolePart
    {
        public uint suitid;
        public string[] helmet;
        public string[] hair;
        public string[] body;
        public List<XStrWeapon> weapons;
    }

    public class FashionExportWindow : EditorWindow
    {
        private FashionSuit.RowData[] fashionInfo;

        [MenuItem("Tools/FashionExportWindow")]
        static void AnimExportTool()
        {
            if (XEditorUtil.MakeNewScene())
            {
                EditorWindow.GetWindowWithRect(typeof(FashionExportWindow), new Rect(0, 0, 600, 400), true, "Fashion Export Tool");
            }
        }


        void OnGUI()
        {
            GUILayout.BeginVertical();
            GUILayout.Label(XEditorUtil.Config.suit_shape, XEditorUtil.titleLableStyle);
            GUILayout.Space(8);

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("TALL")) Export(RoleShape.TALL);
            GUILayout.EndHorizontal();
            GUILayout.Space(4);

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("GIANT")) Export(RoleShape.GIANT);
            GUILayout.EndHorizontal();
            GUILayout.Space(4);

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("MALE")) Export(RoleShape.MALE);
            GUILayout.EndHorizontal();
            GUILayout.Space(4);

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("FEMALE")) Export(RoleShape.FEMALE);
            GUILayout.EndHorizontal();
            GUILayout.Space(10);

            GUILayout.BeginHorizontal();
            if (GUILayout.Button("EXPORT ALL", XEditorUtil.boldButtonStyle))
            {
                if (EditorUtility.DisplayDialog("warn", "Do you really wanted to export all?", "ok", "cancel"))
                {
                    XDebug.singleton.AddLog("start export, wait patient");
                    Export(RoleShape.FEMALE);
                    Export(RoleShape.MALE);
                    Export(RoleShape.GIANT);
                    Export(RoleShape.TALL);
                    XDebug.singleton.AddGreenLog("all shape export finished");
                }
                else
                {
                    XDebug.singleton.AddLog("export canceled");
                }
            }
            GUILayout.EndHorizontal();
            GUILayout.Space(4);

            GUILayout.EndVertical();
        }


        public static string orig_dir { get { return Application.dataPath + "/Creatures/"; } }

        public static string dest_dir { get { return Application.dataPath + "/BundleRes/FBXRawData/"; } }

        static RoleShape role_shape;

        static string[] strFaces;

        static Dictionary<uint, XStrRolePart> bone_map = new Dictionary<uint, XStrRolePart>();

        static List<DirectoryInfo> list = new List<DirectoryInfo>();


        static void Export(RoleShape shape)
        {
            role_shape = shape;
            bone_map.Clear();
            list.Clear();

            string shape_dir = dest_dir + "Player_" + shape.ToString().ToLower();
            if (Directory.Exists(shape_dir)) Directory.Delete(shape_dir, true);
            var ddir = Directory.CreateDirectory(shape_dir);
            list.Add(ddir);
            string shape_common = shape_dir + "/Common";
            if (Directory.Exists(shape_common)) Directory.Delete(shape_common);
            ddir = Directory.CreateDirectory(shape_common);
            list.Add(ddir);

            XEditorUtil.ClearCreatures();
            var suits = XFashionLibrary.GetFashionsInfo(shape);
            for (int i = 0; i < suits.Length; i++)
            {
                ExportSuit(suits[i]);
            }
            ExportFace(shape);
            MakeRolePrefab(shape);
            AssetDatabase.Refresh();
            XEditorUtil.ClearCreatures();
            CleanEmptyDir();
            EditorUtility.DisplayDialog("TIP", "JOB DONE", "ok");
        }


        public static void ExportSuit(FashionSuit.RowData suit, Dictionary<string, string> pathMap = null)
        {
            string dir = suit.dir;
            RoleShape shape = (RoleShape)suit.shape;
            string shape_dir = dest_dir + "Player_" + shape.ToString().ToLower();
            string target = shape_dir + "/" + "Player_" + shape.ToString().ToLower() + "_" + dir;
            if (pathMap == null)
            {
                var ddir = Directory.CreateDirectory(target);
                list.Add(ddir);
            }

            XStrRolePart rolepart = pathMap != null ? null : new XStrRolePart();

            role_shape = (RoleShape)suit.shape;
            HandleMainRolePart(dir, rolepart, pathMap);
            Transform[] trans = null;
            HandleRolePart(suit.hair, (RoleShape)suit.shape, dir, TCConst.HAIR, out trans, pathMap);
            if (pathMap == null) rolepart.hair = XEditorUtil.Transf2Str(trans);
            HandleRolePart(suit.wing, (RoleShape)suit.shape, dir, TCConst.WING, out trans, pathMap);
            if (pathMap == null) rolepart.weapons = new List<XStrWeapon>();
            for (int j = 0; j < suit.weapon.Count; j++)
            {
                string prof = suit.weapon[j, 0];
                if (prof != "E")
                {
                    string weapon1 = suit.weapon[j, 1];
                    string weapon2 = suit.weapon[j, 2];
                    Transform[] tran1 = null, tran2 = null;

                    HandleRolePart(suit.weapon[j, 1], (RoleShape)suit.shape, dir, TCConst.WEAPON, out tran1, pathMap, new object[2] { prof, 1 });
                    HandleRolePart(suit.weapon[j, 2], (RoleShape)suit.shape, dir, TCConst.WEAPON, out tran2, pathMap, new object[2] { prof, 2 });
                    if (pathMap == null)
                    {
                        XStrWeapon weapon = new XStrWeapon();
                        weapon.presentid = uint.Parse(prof);
                        weapon.weapon1 = XEditorUtil.Transf2Str(tran1);
                        weapon.weapon2 = XEditorUtil.Transf2Str(tran2);
                        rolepart.weapons.Add(weapon);
                    }
                }
            }
            if (pathMap == null)
            {
                bone_map.Add(suit.id, rolepart);
                MakeWingPrefab(shape, suit);
            }
        }

        public static void ExportFace(RoleShape shape)
        {
            strFaces = null;
            var table = XFashionLibrary._profession.Table;
            Transform[] trans = null;
            for (int i = 0; i < table.Length; i++)
            {
                string prof = table[i].ID.ToString();
                string name = table[i].face;
                if (!string.IsNullOrEmpty(name) && table[i].Shape == (int)shape)
                {
                    HandleRolePart(name, (RoleShape)table[i].Shape, string.Empty, TCConst.FACE, out trans, null, new object[1] { prof });
                }
            }
            strFaces = XEditorUtil.Transf2Str(trans);
        }

        private static void MakeRolePrefab(RoleShape shape)
        {
            string path = orig_dir + "Player_" + shape.ToString().ToLower() + "/" + "Player_" + shape.ToString().ToLower() + "_bandpose";
            path = path.Substring(path.IndexOf("Assets/"));
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path + ".FBX");
            GameObject go = GameObject.Instantiate(prefab);
            go.name = prefab.name;

            //1. clean mesh & mat
            var renders = go.GetComponentsInChildren<SkinnedMeshRenderer>();
            foreach (var item in renders)
            {
                item.sharedMaterials = new Material[] { };
                item.sharedMesh = null;
            }

            //2. Animator
            var ator = go.GetComponent<Animator>();
            if (ator != null) ator.runtimeAnimatorController = AssetDatabase.LoadAssetAtPath<RuntimeAnimatorController>("Assets/BundleRes/Controller/XAnimator.controller");

            //3. bones
            XRoleParts xparts = go.AddComponent<XRoleParts>();
            xparts.parts = new List<XRolePart>();
            foreach (var item in bone_map)
            {
                XRolePart part = new XRolePart();
                part.suitid = item.Key;
                part.hair = XEditorUtil.Str2Transf(item.Value.hair, go);
                part.shair = item.Value.hair;
                XEditorUtil.CheckDBNil(part.shair, part.hair);
                part.helmet = XEditorUtil.Str2Transf(item.Value.helmet, go);
                part.shelmet = item.Value.helmet;
                XEditorUtil.CheckDBNil(part.shelmet, part.helmet);
                var strWeapons = item.Value.weapons;
                int cnt = strWeapons.Count;
                part.weapon = new XRoleWeapon[cnt];// XEditorUtil.Str2Transf(item.Value.weapon, go);
                for (int i = 0; i < cnt; i++)
                {
                    part.weapon[i] = new XRoleWeapon();
                    part.weapon[i].presentid = item.Value.weapons[i].presentid;
                    part.weapon[i].weapon1 = XEditorUtil.Str2Transf(strWeapons[i].weapon1, go);
                    part.weapon[i].sweapon1 = strWeapons[i].weapon1;
                    part.weapon[i].weapon2 = XEditorUtil.Str2Transf(strWeapons[i].weapon2, go);
                    part.weapon[i].sweapon2 = strWeapons[i].weapon2;
                    XEditorUtil.CheckDBNil(part.weapon[i].sweapon1, part.weapon[i].weapon1);
                    XEditorUtil.CheckDBNil(part.weapon[i].sweapon2, part.weapon[i].weapon2);
                }
                part.body = XEditorUtil.Str2Transf(item.Value.body, go);
                part.sbody = item.Value.body;
                XEditorUtil.CheckDBNil(part.sbody, part.body);
                xparts.face = XEditorUtil.Str2Transf(strFaces, go);
                xparts.parts.Add(part);
            }

            //4. wing root & face root
            Transform[] childs = go.transform.GetComponentsInChildren<Transform>();
            for (int i = 0; i < childs.Length; i++)
            {
                if (childs[i].name.Contains("_wing"))
                {
                    xparts.wingRoot = childs[i];
                }
            }

            //5. cc
            CharacterController cc = go.AddComponent<CharacterController>();
            cc.stepOffset = 0f;
            cc.skinWidth = 0.0001f;
            cc.radius = 0.25f;
            cc.height = 2f;
            cc.minMoveDistance = 0f;
            cc.center = new Vector3(0, 1, 0);

            //6.layer & occluded
            var skms = go.GetComponentsInChildren<SkinnedMeshRenderer>();
            for (int i = 0; i < skms.Length; i++)
            {
                skms[i].gameObject.layer = LayerMask.NameToLayer("Role");
                skms[i].allowOcclusionWhenDynamic = false;
                skms[i].quality = SkinQuality.Bone4;
                skms[i].lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;

                if (skms[i].gameObject.name.ToLower().Contains("weapon")) // required by pangchihai 2018.11.14
                {
                    skms[i].gameObject.tag = "Weapon";
                }
            }

            //7. knead face
            MakeKnead(shape, xparts);

            //8. save to disk
            PrefabUtility.CreatePrefab("Assets/BundleRes/Prefabs/" + go.name.Replace("_bandpose", ".prefab"), go);
        }

        public static void MakeKnead(RoleShape shape, XRoleParts root)
        {
            string path = "Config/" + shape.ToString().ToLower();
            TextAsset ta = XResourceLoaderMgr.singleton.GetSharedResource<TextAsset>(path, ".bytes");
            if (ta != null)
            {
                MemoryStream ms = new MemoryStream(ta.bytes);
                FaceBoneDatas datas = new FaceBoneDatas(ms);
                var list = datas.BoneDatas;
                var childs = root.gameObject.GetComponentsInChildren<Transform>();
                List<Transform> temp = new List<Transform>();
                for (int i = 0; i < list.Length; i++)
                {
                    bool find = false;
                    for (int j = 0; j < childs.Length; j++)
                    {
                        if (childs[j].name == list[i].name)
                        {
                            find = true;
                            if (!temp.Contains(childs[j])) temp.Add(childs[j]);
                            break;
                        }
                    }
                    if (!find)
                    {
                        Debug.LogError("knead face data error, find yuanzongyang");
                    }
                }
                root.knead = temp.ToArray();
                ms.Close();
            }
        }


        private static void MakeWingPrefab(RoleShape shape, FashionSuit.RowData row)
        {
            string path = orig_dir + "Player_common/Player_common_" + row.dir + "_wing";
            path = path.Substring(path.IndexOf("Assets/"));

            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path + ".FBX");
            if (prefab != null)
            {
                GameObject go = GameObject.Instantiate(prefab);
                go.name = prefab.name;

                //1.clean mesh & mat
                var renders = go.GetComponentsInChildren<SkinnedMeshRenderer>();
                foreach (var item in renders)
                {
                    item.sharedMaterials = new Material[] { };
                    item.sharedMesh = null;
                }

                //2. Animator
                var ator = go.GetComponent<Animator>();
                if (ator != null) ator.runtimeAnimatorController = AssetDatabase.LoadAssetAtPath<RuntimeAnimatorController>("Assets/BundleRes/Controller/XAnimator.controller");

                //3. basic bones
                WingPart part = go.AddComponent<WingPart>();
                part.tbones = go.GetComponentInChildren<SkinnedMeshRenderer>().bones;

                //4. dynamic bones
                string dest = "Assets/BundleRes/Prefabs/Wing/" + go.name + ".prefab";
                prefab = AssetDatabase.LoadAssetAtPath<GameObject>(dest);
                DynamicBone[] dbones;
                if (prefab != null)
                {
                    dbones = prefab.GetComponents<DynamicBone>();
                    if (dbones != null)
                    {
                        string[] names = new string[dbones.Length];
                        for (int i = 0; i < dbones.Length; i++)
                        {
                            names[i] = dbones[i].m_Root.name;
                            UnityEditorInternal.ComponentUtility.CopyComponent(dbones[i]);
                            UnityEditorInternal.ComponentUtility.PasteComponentAsNew(go);
                        }
                        dbones = go.GetComponents<DynamicBone>();
                        Transform[] tranfs = go.GetComponentsInChildren<Transform>();
                        for (int i = 0; i < dbones.Length; i++)
                        {
                            for (int j = 0; j < tranfs.Length; j++)
                            {
                                if (tranfs[j].name == names[i])
                                {
                                    dbones[i].m_Root = tranfs[j];
                                    break;
                                }
                            }
                        }
                    }
                }

                //5. reset transform  required by artist(yuanzongyang)
                int cnt = go.transform.childCount;
                for (int i = 0; i < cnt; i++)
                {
                    go.transform.GetChild(i).localRotation = Quaternion.identity;
                }

                //6. save to disk
                PrefabUtility.CreatePrefab(dest, go);
            }
        }

        private static void HandleMainRolePart(string dir, XStrRolePart rolepart, Dictionary<string, string> pathMap = null)
        {
            string shape = role_shape.ToString().ToLower();
            string fbx_dir = orig_dir + "Player_" + shape + "/Player_" + shape + "_" + dir + "/";
            fbx_dir = fbx_dir.Substring(fbx_dir.IndexOf("Assets/"));
            string fbx_name = "Player_" + shape + "_" + dir;
            string target_dir = dest_dir + "Player_" + shape + "/Player_" + shape + "_" + dir;
            target_dir = target_dir.Substring(target_dir.IndexOf("Assets/"));

            string suffx = "_" + TCConst.BODY;
            string name = fbx_name + suffx;
            string bodyMat0 = fbx_dir + "Materials_" + fbx_name + "/" + name + ".mat";
            string bodyMat1 = target_dir + "/" + name + ".mat";


            string suffx1 = "_" + TCConst.HELMET;
            string helmetname = fbx_name + suffx1;
            string helmetMat0 = fbx_dir + "Materials_" + fbx_name + "/" + helmetname + ".mat";
            string helmetMat1 = target_dir + "/" + helmetname + ".mat";

            if (pathMap == null)
            {
                GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(fbx_dir + fbx_name + ".FBX");
                if (prefab != null)
                {
                    GameObject go = GameObject.Instantiate(prefab);
                    go.name = prefab.name;

                    Transform body = go.transform.Find(fbx_name + suffx);
                    if (body != null)
                    {
                        SkinnedMeshRenderer body_skin = body.gameObject.GetComponent<SkinnedMeshRenderer>();
                        rolepart.body = XEditorUtil.Transf2Str(body_skin.bones);
                        Mesh newMesh = null;
                        Material newMat = null;
                        FBXAssets.GetMeshMat(body_skin, out newMesh, out newMat);
                        if (newMesh != null)
                        {
                            XEditorUtil.ClearMesh(newMesh);
                            AssetDatabase.CreateAsset(newMesh, target_dir + "/" + name + ".asset");
                            AssetDatabase.CopyAsset(bodyMat0, bodyMat1);
                        }
                    }

                    Transform helmet = go.transform.Find(fbx_name + suffx1);
                    if (helmet != null)
                    {
                        SkinnedMeshRenderer helmet_skin = helmet.gameObject.GetComponent<SkinnedMeshRenderer>();
                        rolepart.helmet = XEditorUtil.Transf2Str(helmet_skin.bones);
                        Mesh newMesh = null;
                        Material newMat = null;
                        FBXAssets.GetMeshMat(helmet_skin, out newMesh, out newMat);
                        if (newMesh != null)
                        {
                            XEditorUtil.ClearMesh(newMesh);
                            AssetDatabase.CreateAsset(newMesh, target_dir + "/" + helmetname + ".asset");
                            AssetDatabase.CopyAsset(helmetMat0, helmetMat1);
                        }
                    }
                }
            }
            else
            {
                pathMap[bodyMat1] = bodyMat0;
                pathMap[helmetMat1] = helmetMat0;
            }
        }

        private static void HandleRolePart(string part, RoleShape shape, string dir, string suffix, out Transform[] bones, Dictionary<string, string> pathMap, object[] args = null)
        {
            string sshape = shape.ToString().ToLower();
            string shape_dir = "Player_" + sshape;
            string suit_dir = shape_dir + "_" + dir;
            string offset_dir = shape_dir + "/" + suit_dir + "/";
            string mat_dir = "Materials_Player_" + sshape + "_" + dir + "/";
            if (suffix == TCConst.WEAPON) //handle special because weapon is bind with profession
            {
                string presentid = args[0].ToString();
                int index = (int)args[1];
                string cname = "Player_" + sshape + "_" + dir + "_" + suffix + "_" + part;
                string cname2 = "Player_" + sshape + "_" + dir + "_" + suffix + "_" + presentid + "_" + index;
                HandleRolePart(orig_dir + offset_dir, dest_dir + offset_dir, mat_dir, cname, cname2, out bones, pathMap);
            }
            else if (suffix == TCConst.FACE)
            {
                string prof = args[0].ToString();
                offset_dir = shape_dir + "/Common/";
                mat_dir = "Materials_Player_" + sshape + "_common/";
                string cname = "Player_" + sshape + "_common_" + suffix + "_" + part;
                string cname2 = "Player_" + sshape + "_common_" + suffix + "_" + prof;
                XDebug.singleton.AddGreenLog(cname + " : " + cname2);
                HandleRolePart(orig_dir + offset_dir, dest_dir + shape_dir + "/Common/", mat_dir, cname, cname2, out bones, pathMap);
            }
            else if (suffix == TCConst.WING)
            {
                string cname = "Player_common_" + dir + "_" + suffix;
                mat_dir = "Materials_Player_common/";

                HandleRolePart(orig_dir + "Player_common/", dest_dir + "Player_common/", mat_dir, cname, cname, out bones, pathMap);
            }
            else if (string.IsNullOrEmpty(part))
            {
                string cname = "Player_" + sshape + "_" + dir + "_" + suffix;
                HandleRolePart(orig_dir + offset_dir, dest_dir + offset_dir + "/", mat_dir, cname, cname, out bones, pathMap);
            }
            else if (!part.Trim().Equals("E"))
            {
                string cname = "Player_" + sshape + "_common_" + suffix + "_" + part;
                offset_dir = shape_dir + "/Common/";
                mat_dir = "Materials_Player_" + sshape + "_common/";
                HandleRolePart(orig_dir + offset_dir, dest_dir + shape_dir + "/Common/", mat_dir, cname, cname, out bones, pathMap);
            }
            else
            {
                bones = null;
            }
        }

        public static void HandleRolePart(string origin, string target, string matdir, string cname, string cname2, out Transform[] bones, Dictionary<string, string> pathMap)
        {
            var arr = SearchRefenced(origin, cname);
            DoRolePart(origin, target, matdir, cname, cname2, out bones, pathMap);
            for (int i = 0; i < arr.Length; i++)
            {
                string suffx = arr[i];
                DoRolePart(origin, target, matdir, cname + suffx, cname2 + suffx, out bones, pathMap);
            }
        }


        private static void DoRolePart(string origin, string target, string matdir, string cname, string cname2, out Transform[] bones, Dictionary<string, string> pathMap)
        {
            origin = origin.Substring(origin.IndexOf("Assets/"));
            target = target.Substring(target.IndexOf("Assets/"));

            string srcMat = origin + matdir + cname + ".mat";
            string desMat = target + cname2 + ".mat";
            bones = null;
            if (pathMap == null)
            {
                GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(origin + cname + ".FBX");
                if (prefab != null)
                {
                    AssetDatabase.DeleteAsset(target + cname2 + ".mat");
                    GameObject go = GameObject.Instantiate(prefab);
                    go.name = prefab.name;
                    go.tag = "EditorOnly";
                    SkinnedMeshRenderer smr = go.GetComponentInChildren<SkinnedMeshRenderer>();
                    bones = smr.bones;
                    Mesh newMesh = null;
                    Material newMat = null;
                    FBXAssets.GetMeshMat(smr, out newMesh, out newMat);
                    if (newMesh != null)
                    {
                        XEditorUtil.ClearMesh(newMesh);
                        AssetDatabase.CopyAsset(srcMat, desMat);
                        AssetDatabase.CreateAsset(newMesh, target + cname2 + ".asset");
                    }
                }
            }
            else
            {
                pathMap[desMat] = srcMat;
            }

        }

        /// <summary>
        /// 根据文件命令查找AB套相关的变种资源后缀
        /// </summary>
        private static string[] SearchRefenced(string dir, string name)
        {
            List<string> list = new List<string>();
            DirectoryInfo dirinfo = new DirectoryInfo(dir);
            var files = dirinfo.GetFiles(name + "*.FBX", SearchOption.TopDirectoryOnly); //根据名字正则匹配
            for (int i = 0; i < files.Length; i++)
            {
                string newname = files[i].Name.Replace(".FBX", string.Empty).Replace(name, string.Empty);
                list.Add(newname);
            }
            return list.ToArray();
        }

        /// <summary>
        /// 清除空目录
        /// </summary>
        private static void CleanEmptyDir()
        {
            for (int i = list.Count - 1; i >= 0; i--)
            {
                var dir = list[i];
                var files = dir.GetFiles();
                if (files == null || files.Length == 0)
                {
                    dir.Delete(true);
                }
            }
        }
    }

}