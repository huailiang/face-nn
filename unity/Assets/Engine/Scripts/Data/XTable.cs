#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Collections;
using CFUtilPoolLib;
using System.IO;

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

public class XTableWriter
{
    public static StreamWriter StartWriteFile(string path)
    {
        StreamWriter sw = File.CreateText(path);
        System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
        return sw;
    }

    public static void WriteCol(StreamWriter sw, string[] colname)
    {
        string s = "";
        for(int i = 0; i < colname.Length; ++i)
        {
            s += colname[i];

            if (i < colname.Length - 1) s += '\t';
        }

        sw.WriteLine(s);
    }

    public static void WriteComment(StreamWriter sw, string[] comments)
    {
        string s = "";
        for (int i = 0; i < comments.Length; ++i)
        {
            s += comments[i];

            if (i < comments.Length - 1) s += '\t';
        }

        sw.WriteLine(s);
    }

    public static void WriteContent(StreamWriter sw, string[] row)
    {
        string s = "";
        for (int i = 0; i < row.Length; ++i)
        {
            s += row[i];

            if (i < row.Length - 1) s += '\t';
        }

        sw.WriteLine(s);
    }


    public static void EndWriteFile(StreamWriter sw)
    {
        sw.Flush();
        sw.Close();
        AssetDatabase.Refresh();
    }
}
#endif