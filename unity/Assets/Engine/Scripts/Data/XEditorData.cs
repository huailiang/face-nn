#if UNITY_EDITOR
using CFUtilPoolLib;
using System.Linq;
using System.Collections.Generic;
using System;
using UnityEditor;
using System.IO;

namespace XEngine.Editor
{

    public class XResources : XSingleton<XResources>, IResourceHelp
    {
        public void CheckResource(UnityEngine.Object o, string path) { }

        public UnityEngine.Object LoadEditorResource(string path, string suffix, Type t)
        {
            return AssetDatabase.LoadAssetAtPath(path + suffix, t);
        }

        public bool Deprecated { get; set; }

        public static void LoadAllAssets(string folder, List<UnityEngine.Object> outputFiles)
        {
            DirectoryInfo direction = new DirectoryInfo(folder);
            FileSystemInfo[] fs = direction.GetFileSystemInfos();

            for (int i = 0; i < fs.Length; i++)
            {
                if (fs[i] is DirectoryInfo)
                {
                    LoadAllAssets(fs[i].FullName, outputFiles);
                }
                else if (fs[i] is FileInfo)
                {
                    if (fs[i].FullName.EndsWith(".meta")) continue;
                    int index = fs[i].FullName.IndexOf("Assets\\");
                    string path = fs[i].FullName.Substring(index).Replace('\\', '/');

                    var obj = AssetDatabase.LoadMainAssetAtPath(path);
                    if (obj != null) outputFiles.Add(obj);
                }
            }
        }
    }

    public class XTableReader
    {
        public static bool ReadFile(string location, CVSReader reader)
        {
            CVSReader.Init();
            XBinaryReader.Init();
            XInterfaceMgr.singleton.AttachInterface<IResourceHelp>(XCommon.singleton.XHash("XResourceHelper"), XResources.singleton);
            return XResourceLoaderMgr.singleton.ReadFile(location, reader);
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

    }

}

#endif