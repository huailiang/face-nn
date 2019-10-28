#if UNITY_EDITOR

using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;

namespace XEngine.Editor
{

    public static class FashionUtility
    {

        public static void UnloadSuit(GameObject go)
        {
            SkinnedMeshRenderer[] skms = go.GetComponentsInChildren<SkinnedMeshRenderer>();
            foreach (var item in skms)
            {
                item.sharedMaterials = new Material[] { null };
                item.sharedMesh = null;
            }
            XRoleParts part = go.GetComponent<XRoleParts>();
            if (part != null)
            {
                Transform wing = part.wingRoot;
                int cnt = wing.childCount;
                for (int i = 0; i < cnt; i++)
                {
                    GameObject.DestroyImmediate(wing.GetChild(i).gameObject);
                }
            }
            Resources.UnloadUnusedAssets();
        }


        public static FashionList.RowData SearchPart(uint suitid, uint pos)
        {
            var table = XFashionLibrary.FashionList;
            for (int i = 0; i < table.Length; i++)
            {
                if (table[i].SuitID == suitid && table[i].EquipPos == pos)
                {
                    return table[i];
                }
            }
            return null;
        }

        public static void DrawSuit(GameObject go, FashionSuit.RowData rowData, uint presentid, int weapon_index, bool complete)
        {
            if (rowData != null)
            {
                XRoleParts xpart = go.GetComponent<XRoleParts>();
                XRolePart part = UtilPart(go, rowData.id);
                if (part != null)
                {
                    UnloadSuit(go);
                    DrawFace(go, (RoleShape)rowData.shape, TCConst.FACE, xpart.face, presentid);
                    if (complete)
                    {
                        DrawPart(go, rowData.body, (RoleShape)rowData.shape, rowData.dir, TCConst.BODY, part.body, part.sbody, SearchPart(rowData.id, (uint)FPart.BODY));
                        DrawPart(go, rowData.hair, (RoleShape)rowData.shape, rowData.dir, TCConst.HAIR, part.hair, part.shair, SearchPart(rowData.id, (uint)FPart.HAIR));
                        DrawPart(go, rowData.helmet, (RoleShape)rowData.shape, rowData.dir, TCConst.HELMET, part.helmet, part.shelmet, SearchPart(rowData.id, (uint)FPart.HELMET));
                        uint pos = weapon_index == 1 ? (uint)FPart.WEAPON : 5;
                        DrawWeapon(go, (RoleShape)rowData.shape, rowData.dir, part, presentid, weapon_index, SearchPart(rowData.id, pos));
                        DrawWing(go, rowData.dir);
                    }
                }
            }
            else
            {
                Debug.LogError("rowdata is null " + go.name + "  presentid: " + presentid);
            }
        }

        public static void DrawWing(GameObject go, string dir)
        {
            XRoleParts parts = go.GetComponent<XRoleParts>();
            Transform root = parts.wingRoot;
            if (root.childCount > 0) return;
            string path = "Assets/Resource/Prefabs/Wing/Player_common_" + dir + "_wing.prefab";
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (prefab != null)
            {
                go = GameObject.Instantiate(prefab);
                go.transform.SetParent(root);
                go.transform.localScale = Vector3.one;
                go.transform.rotation = Quaternion.identity;
                go.transform.position = Vector3.zero;

                SkinnedMeshRenderer skm = go.GetComponentInChildren<SkinnedMeshRenderer>();
                path = "Assets/Resource/RawData/Player_common/Player_common_" + dir + "_wing";
                skm.sharedMaterial = AssetDatabase.LoadAssetAtPath<Material>(path + ".mat");
                skm.sharedMesh = AssetDatabase.LoadAssetAtPath<Mesh>(path + ".asset");
            }
        }


        public static void DrawWeapon(GameObject go, RoleShape shape, string dir, XRolePart part, uint presentid, int weapon_index, FashionList.RowData row)
        {
            string sshape = shape.ToString().ToLower();
            Transform transf = go.transform.Find("Player_" + sshape + "_" + TCConst.WEAPON);
            string shape_dir = "Player_" + sshape;
            string cname = "Player_" + sshape + "_" + dir + "_" + TCConst.WEAPON + "_" + presentid + "_" + weapon_index;
            string asset = "Assets/Resource/RawData/" + shape_dir + "/" + shape_dir + "_" + dir + "/" + cname;
            for (int i = 0; i < part.weapon.Length; i++)
            {
                if (part.weapon[i].presentid == presentid)
                {
                    if (weapon_index == 1)
                    {
                        FashionUtility.DrawPart(transf, asset, part.weapon[i].weapon1, part.weapon[i].sweapon1, row);
                    }
                    else
                    {
                        DrawPart(transf, asset, part.weapon[i].weapon2, part.weapon[i].sweapon2, row);
                    }
                    break;
                }
            }
        }

        public static void DrawFace(GameObject go, RoleShape shape, string suffix, Transform[] bones, uint presentid)
        {
            string sshape = shape.ToString().ToLower();
            Transform transf = go.transform.Find("Player_" + sshape + "_" + TCConst.FACE);
            string shape_dir = "Player_" + sshape;
            var row = XFashionLibrary.FindRole(presentid);
            if (row != null)
            {
                string cname = "Player_" + sshape + "_common_" + TCConst.FACE + "_" + row.ID;
                string asset = "Assets/Resource/RawData/" + shape_dir + "/Common/" + cname;
                DrawPart(transf, asset, bones, null, null);
            }
            else
            {
                Debug.LogError("not find present id in profession table, " + presentid);
            }
        }

        public static void DrawPart(GameObject go, string part, RoleShape shape, string dir, string suffix, Transform[] bones, string[] dbones, FashionList.RowData row)
        {
            string sshape = shape.ToString().ToLower();
            Transform transf = go.transform.Find("Player_" + sshape + "_" + suffix);
            string asset = string.Empty;
            string shape_dir = "Player_" + sshape;
            if (string.IsNullOrEmpty(part))
            {
                string cname = "Player_" + sshape + "_" + dir + "_" + suffix;
                asset = "Assets/Resource/RawData/" + shape_dir + "/" + shape_dir + "_" + dir + "/" + cname;
                DrawPart(transf, asset, bones, dbones, row);
            }
            else if (!part.Equals("E"))
            {
                string cname = "Player_" + sshape + "_common_" + suffix + "_" + part;
                asset = "Assets/Resource/RawData/" + shape_dir + "/Common/" + cname;
                DrawPart(transf, asset, bones, dbones, row);
            }
        }

        public static void DrawPart(Transform tranf, string asset, Transform[] bones, string[] dbones, FashionList.RowData row)
        {
            if (tranf != null)
            {
                SkinnedMeshRenderer smr = tranf.gameObject.GetComponent<SkinnedMeshRenderer>();
                var mesh = AssetDatabase.LoadAssetAtPath<Mesh>(asset + ".asset");
                var mat = AssetDatabase.LoadAssetAtPath<Material>(asset + ".mat");
                if (mesh != null) smr.sharedMesh = mesh;
                if (mat != null) smr.sharedMaterial = mat;

                if (row != null && row.dbparent != null)
                {
                    for (int k = 0; k < row.dbparent.Length; k++)
                    {
                        GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Resource/Prefabs/DynamicBone/" + row.dbprefab[k] + ".prefab");
                        if (prefab == null)
                        {
                            Debug.LogError("DynamicBones error: fashionlist id: " + row.ID + " index: " + k + "  prefab config: " + row.dbprefab[k]);
                        }
                        else
                        {
                            GameObject go = GameObject.Instantiate(prefab);
                            Vector3 pos = go.transform.localPosition;
                            Quaternion rot = go.transform.localRotation;
                            Transform par = tranf.parent.Find(row.dbparent[k]);
                            go.transform.SetParent(par);
                            go.transform.localPosition = pos;
                            go.transform.localRotation = rot;

                            Transform[] childs = go.GetComponentsInChildren<Transform>();
                            for (int i = 0; i < bones.Length; i++)
                            {
                                if (dbones != null && !string.IsNullOrEmpty(dbones[i]))
                                {
                                    var tf = SearchChild(childs, tranf, dbones[i]);
                                    if (tf != null) bones[i] = tf;
                                }
                            }
                        }
                    }
                }
                smr.bones = bones;
            }
        }

        private static Transform SearchChild(Transform[] childs, Transform parent, string name)
        {
            if (parent.name == name) return parent;
            for (int i = 0; i < childs.Length; i++)
            {
                if (childs[i].name == name)
                {
                    return childs[i];
                }
            }
            return null;
        }


        public static XRolePart UtilPart(GameObject go, uint suitid)
        {
            XRoleParts xpart = go.GetComponent<XRoleParts>();
            if (xpart != null)
            {
                for (int i = 0; i < xpart.parts.Count; i++)
                {
                    if (xpart.parts[i].suitid == suitid)
                    {
                        return xpart.parts[i];
                    }
                }
            }
            return null;
        }

    }
}

#endif