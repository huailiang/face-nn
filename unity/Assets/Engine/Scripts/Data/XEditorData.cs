#if UNITY_EDITOR
using CFUtilPoolLib;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace XEngine.Editor
{

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
            return _suit.Table.Where(x => x.shape == (int)shape).ToArray();
        }

        public static FashionSuit.RowData GetFashionsInfo(uint suitID)
        {
            return _suit.Table.Where(x => x.id == suitID).First();
        }

        public static FashionSuit.RowData GetFashionsInfo(string suitName)
        {
            return _suit.Table.Where(x => x.dir == suitName).First();
        }

        public static ProfessionTable.RowData FindRole(uint presentid)
        {
            var ptable = _profession.Table;
            return ptable.Where(x => x.PresentID == presentid).First();
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
            return PartTable.Table.Where(x => x.PresentID == presentid).ToArray();
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
                    if (t == null) { Debug.LogError("DestructionPart config error: " + presentid + " perfectpart: " + dData[i].PerfectPart); continue; }
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

}

#endif