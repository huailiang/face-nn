#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

using CFUtilPoolLib;

using System.IO;
using UnityEditor;

namespace XEditor
{
    public class XParse
    {
        public static readonly char[] ListSeparator = new char[] { '|' };
        private static bool inited = false;
        public static float Parse(string str)
        {
            if(!inited)
            {
                System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
                inited = true;
            }
            float value = 0.0f;
            float.TryParse(str, out value);
            return value;
        }

        public static string[] ParseStrs(string str)
        {
            if (!inited)
            {
                System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
                inited = true;
            }
            return str.Split(ListSeparator);
        }
    }


    public class XFashionLibrary
    {
        private static FashionSuit _suit = new FashionSuit();
        private static FashionList _list = new FashionList();
        public static ProfessionTable _profession = new ProfessionTable();

        static XFashionLibrary()
        {
            XTableReader.ReadFile(@"Table/FashionSuit", _suit);
            XTableReader.ReadFile(@"Table/FashionList", _list);
            XTableReader.ReadFile(@"Table/Profession", _profession);
        }


        public static FashionSuit.RowData[] FashionsInfo
        {
            get { return _suit.Table; }
        }

        public static FashionList.RowData[] FashionList
        {
            get { return _list.Table; }
        }

        public static FashionSuit.RowData[] GetFashionsInfo(RoleShape shape)
        {
            List<FashionSuit.RowData> list = new List<FashionSuit.RowData>();
            foreach (var item in _suit.Table)
            {
                if (item.shape == (int)shape)
                {
                    list.Add(item);
                }
            }
            return list.ToArray();
        }

        public static FashionSuit.RowData GetFashionsInfo(uint suitID)
        {
            foreach(var item in _suit.Table)
            {
                if (item.id == suitID)
                    return item;
            }
            return null;
        }

        public static FashionSuit.RowData GetFashionsInfo(string suitName)
        {
            foreach (var item in _suit.Table)
            {
                if (item.dir == suitName)
                    return item;
            }
            return null;
        }

        public static ProfessionTable.RowData FindRole(uint presentid)
        {
            var ptable = _profession.Table;
            for (int i = 0; i < ptable.Length; i++)
            {
                if (ptable[i].PresentID == presentid || ptable[i].SecondaryPresentID == presentid)
                {
                    return ptable[i];
                }
            }
            return null;
        }


        public static void DrawRoleWithPresentID(uint presentid, GameObject go)
        {
            var role = FindRole(presentid);
            if (role != null)
            {
                for (int i = 0; i < _suit.Table.Length; i++)
                {
                    if (_suit.Table[i].id == role.SuitID)
                    {
                        FashionUtil.DrawSuit(go, _suit.Table[i], presentid, 1);
                        break;
                    }
                }
            }
        }

    }


    public class XItemLibrary
    {
        private static ItemList _item = new ItemList();

        static XItemLibrary()
        {
            XTableReader.ReadFile(@"Table/ItemList", _item);
        }

        public static ItemList.RowData GetItemInfo(uint itemID)
        {
            return _item.GetByID(itemID);
        }
    }

    public class XDestructionLibrary
    {
        private static DestructionPart _part = new DestructionPart();
        public static DestructionPart PartTable { get { return _part; } }
        static XDestructionLibrary()
        {
            XTableReader.ReadFile(@"Table/DestructionPart", _part);
        }

        public static DestructionPart.RowData[] GetPartsInfo(uint presentid)
        {
            List<DestructionPart.RowData> list = new List<DestructionPart.RowData>();
            foreach (var item in PartTable.Table)
            {
                if (item.PresentID == presentid)
                {
                    list.Add(item);
                }
            }
            return list.ToArray();
        }

        public static void InitWithPerfectPart(DestructionPart.RowData[] dData, string suff, SkinnedMeshRenderer[] renders, XParts xpart)
        {
            for (int i = 0; i < dData.Length; i++)
            {
                var item = dData[i];
                if (!string.IsNullOrEmpty(item.PerfectPart))
                {
                    if (!renders[i].gameObject.activeSelf)
                        renders[i].gameObject.SetActive(true);
                    var mesh = AssetDatabase.LoadAssetAtPath<Mesh>(suff + item.PerfectPart + ".asset");
                    var mat = AssetDatabase.LoadAssetAtPath<Material>(suff + item.PerfectPart + ".mat");
                    if (mesh != null) renders[i].sharedMesh = mesh;
                    if (mat != null) renders[i].sharedMaterial = mat;
                    if (xpart != null)
                    {
                        for (int k = 0; k < xpart.parts.Count; k++)
                        {
                            if (xpart.parts[k].part == item.PerfectPart)
                            {
                                renders[i].bones = xpart.parts[k].perfect;
                                break;
                            }
                        }
                    }
                }
            }
        }


        public static void AttachDress(uint presentid, GameObject go)
        {
            var dData = XDestructionLibrary.GetPartsInfo(presentid);
            var present = XAnimationLibrary.AssociatedAnimations(presentid);
            XParts xpart = go.GetComponent<XParts>();
            if (dData != null && dData.Length > 0)
            {
                SkinnedMeshRenderer[] renders = new SkinnedMeshRenderer[dData.Length];
                for (int i = 0; i < dData.Length; i++)
                {
                    var t = go.transform.Find(dData[i].PerfectPart);
                    if (t == null) { XDebug.singleton.AddErrorLog("DestructionPart config error: " + presentid + " perfectpart: " + dData[i].PerfectPart); continue; }
                    renders[i] = t.GetComponent<SkinnedMeshRenderer>();
                }
                InitWithPerfectPart(dData, "Assets/BundleRes/FBXRawData/" + present.Prefab + "/", renders, xpart);
            }
            else
            {
                XFashionLibrary.DrawRoleWithPresentID(presentid, go);
            }
        }
    }

    public class XAnimationLibrary
    {
        private static XEntityPresentation _presentations = new XEntityPresentation();
        public static XEntityPresentation Presentations { get { return _presentations; } }
        static XAnimationLibrary()
        {
            XTableReader.ReadFile(@"Table/XEntityPresentation", _presentations);
        }

        public static XEntityPresentation.RowData AssociatedAnimations(uint presentid)
        {
            return _presentations.GetByPresentID(presentid);
        }

        public static XEntityPresentation.RowData FindByColliderID(int colliderid)
        {
            int cnt = _presentations.Table.Length;
            foreach (var item in _presentations.Table)
            {
                if (item.ColliderID != null)
                {
                    foreach (var id in item.ColliderID)
                    {
                        if (id == colliderid) return item;
                    }
                }
            }
            return null;
        }



        public static GameObject GetDummy(uint presentid)
        {
            XEntityPresentation.RowData raw_data = AssociatedAnimations(presentid);
            if (raw_data == null) return null;
            
            return GetDummy(raw_data.Prefab);
        }

        public static GameObject GetDummy(string path)
        {

            int n = path.LastIndexOf("_SkinnedMesh");
            int m = path.LastIndexOf("Loading");
            return n < 0 || m > 0 ?
                AssetDatabase.LoadAssetAtPath("Assets/BundleRes/Prefabs/" + path + ".prefab", typeof(GameObject)) as GameObject :
                AssetDatabase.LoadAssetAtPath("Assets/Editor/EditorResources/Prefabs/" + path.Substring(0, n) + ".prefab", typeof(GameObject)) as GameObject;
        }

        public static XEntityPresentation GetPresentation { get { return _presentations; } }
    }

    public class XStatisticsLibrary
    {
        private static XEntityStatistics _statistics = new XEntityStatistics();

        static XStatisticsLibrary()
        {
            XTableReader.ReadFile(@"Table/XEntityStatistics", _statistics);
        }

        public static XEntityStatistics.RowData AssociatedData(uint id)
        {
            return _statistics.GetByID(id);
        }

        public static GameObject GetDummy(uint id)
        {
            XEntityStatistics.RowData data = AssociatedData(id);
            if (data == null) return null;
            return XAnimationLibrary.GetDummy(data.PresentID);
        }
    }

    public class XSceneLibrary
    {
        private static SceneTable _table = new SceneTable();

        static XSceneLibrary()
        {
            XTableReader.ReadFile(@"Table/SceneList", _table);
        }

        public static SceneTable.RowData AssociatedData(uint id)
        {
            return _table.GetBySceneID((int)id);
        }

        public static string GetDynamicString(string levelConfig)
        {
            for (int i = 0; i < _table.Table.Length; i++)
            {
                if (_table.Table[i].configFile == levelConfig)
                    return _table.Table[i].DynamicScene;
            }

            return "";
        }
    }

    public class XColliderLibrary
    {
        private static ColliderTable _table = new ColliderTable();

        static XColliderLibrary()
        {
            XTableReader.ReadFile(@"Table/ColliderTable", _table);
        }

        public static ColliderTable.RowData AssociatedData(uint id)
        {
            return _table.GetByColliderID(id);
        }
    }



}
#endif