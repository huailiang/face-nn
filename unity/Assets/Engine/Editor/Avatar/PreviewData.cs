using System.Collections.Generic;
using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;

namespace XEditor
{

    public class PreviewData
    {
        public class PlayerFaceData
        {
            public string shape;
            public uint pid;
            public PlayerFaceData(string _shape, uint _pid)
            {
                shape = _shape;
                pid = _pid;
            }
        }

        public class WeaponData
        {
            public string shape;
            public uint pid;
            public bool one;
            public string name;
            public string suitName;
            public WeaponData(string _shape, uint _pid, bool _one, string _name)
            {
                shape = _shape;
                pid = _pid;
                one = _one;
                name = _name;
                suitName = name.Substring(0, name.IndexOf('_'));
            }
        }

        public class ShapeProfData
        {
            public string name;
            public uint pid;
            public ShapeProfData(string _name, uint _pid)
            {
                name = _name;
                pid = _pid;
            }
        }

        public static List<string> GetAllBossName()
        {
            List<string> res = new List<string>();
            string[] ids = AssetDatabase.FindAssets("t:Prefab", new string[] { "Assets/BundleRes/Prefabs" });
            HashSet<uint> pids = new HashSet<uint>();
            HashSet<string> prefabNames = new HashSet<string>();
            for (int i = 0; i < XDestructionLibrary.PartTable.Table.Length; i++)
                if (!pids.Contains(XDestructionLibrary.PartTable.Table[i].PresentID))
                    pids.Add(XDestructionLibrary.PartTable.Table[i].PresentID);
            foreach (uint pid in pids)
            {
                var data = XAnimationLibrary.AssociatedAnimations(pid);
                if (!prefabNames.Contains(data.Prefab))
                    prefabNames.Add(data.Prefab);
            }
            for (int i = 0; i < ids.Length; i++)
            {
                string name = AssetDatabase.GUIDToAssetPath(ids[i]).Replace("Assets/BundleRes/Prefabs/", "").Replace(".prefab", "");
                if (!name.Contains("/") && prefabNames.Contains(name))
                    res.Add(name);
            }
            return res;
        }

        public static List<PlayerFaceData> GetPlayerFaceList()
        {
            List<PlayerFaceData> list = new List<PlayerFaceData>();
            string[] ids = AssetDatabase.FindAssets("t:Mesh", new string[] { "Assets/BundleRes/FBXRawData" });
            for (int i = 0; i < ids.Length; i++)
            {
                string name = AssetDatabase.GUIDToAssetPath(ids[i]);
                string houzui = name.Substring(name.LastIndexOf("/") + 1);
                string str1 = name.Replace(houzui, "").TrimEnd('/').Replace("/Common", "");
                string fuji = str1.Substring(str1.LastIndexOf("/") + 1);
                houzui = houzui.Replace(".asset", "");
                if (houzui.Contains("face"))
                {
                    list.Add(new PlayerFaceData(fuji.Replace("Player_", ""), uint.Parse(houzui.Substring(houzui.LastIndexOf("_") + 1))));
                }
            }
            return list;
        }

        public static List<WeaponData> GetWeaponList()
        {
            List<WeaponData> list = new List<WeaponData>();
            string[] ids = AssetDatabase.FindAssets("t:Mesh", new string[] { "Assets/BundleRes/FBXRawData" });
            for (int i = 0; i < ids.Length; i++)
            {
                string name = AssetDatabase.GUIDToAssetPath(ids[i]);
                string houzui = name.Substring(name.LastIndexOf("/") + 1);
                string str1 = name.Replace(houzui, "").TrimEnd('/');
                str1 = str1.Substring(0, str1.LastIndexOf("/"));
                string fuji = str1.Substring(str1.LastIndexOf("/") + 1);
                houzui = houzui.Replace(".asset", "");
                if(houzui.Contains("weapon"))
                {
                    string str2 = houzui.Substring(houzui.LastIndexOf('_') + 1);
                    string str3 = houzui.Substring(0, houzui.LastIndexOf('_'));
                    string str4 = str3.Substring(str3.LastIndexOf('_') + 1);
                    string shapeStr = fuji.Replace("Player_", "");
                    list.Add(new WeaponData(shapeStr, uint.Parse(str4), str2 == "1", houzui.Replace("Player_", "").Replace(shapeStr + "_", "")));
                }
            }
            return list;
        }

        public static Dictionary<string, string> GetMonsterCreaturesPath()
        {
            Dictionary<string, string> dict = new Dictionary<string, string>();
            string[] ids = AssetDatabase.FindAssets("t:Model", new string[] { "Assets/Creatures" });
            for (int i = 0; i < ids.Length; i++)
            {
                string path = AssetDatabase.GUIDToAssetPath(ids[i]);
                if (!path.EndsWith("bandpose.FBX"))
                    continue;
                if (!path.Contains("Monster"))
                    continue;
                if (!path.Contains("Bandpose"))
                    continue;

                string fbxPath = path.Substring(0, path.IndexOf("Bandpose") - 1);
                int start = path.LastIndexOf('/') + 1;
                int len = path.IndexOf("_bandpose.FBX") - start;
                string name = path.Substring(start, len);

                dict.Add(name, fbxPath);
                //Debug.Log(path);
                //Debug.Log(fbxPath);
                //Debug.Log(name);
            }
            return dict;
        }
    }
}