#if UNITY_EDITOR
using UnityEngine;
using System.Collections;
using CFUtilPoolLib;

public class XTableReader
{
    public static bool ReadFile(string location, CVSReader reader)
    {
        CVSReader.Init();
        XBinaryReader.Init();
        XInterfaceMgr.singleton.AttachInterface<IResourceHelp>(XCommon.singleton.XHash("XResourceHelper"), XResourceHelper.singleton);

        return XResourceLoaderMgr.singleton.ReadFile(location, reader);
    }
}
#endif