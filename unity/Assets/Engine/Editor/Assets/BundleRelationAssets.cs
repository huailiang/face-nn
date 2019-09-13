using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using CFEngine;

namespace CFEngine.Editor
{
    internal class BundleRelationAssets
    {
        private Dictionary<string, string> pathRemap = new Dictionary<string, string>();
        private static BundleRelationAssets g_BundleRelationAssets;
        public static BundleRelationAssets BundlePathRemap
        {
            get
            {
                if (g_BundleRelationAssets == null)
                {
                    g_BundleRelationAssets = new BundleRelationAssets();
                }
                return g_BundleRelationAssets;
            }
        }
        internal void Load()
        {
            pathRemap.Clear();
        }
        public void Reset()
        {
            pathRemap.Clear();
        }

        public void Add(string physicPath, string virtualPath)
        {
            if (!pathRemap.ContainsKey(physicPath))
            {
                pathRemap.Add(physicPath, virtualPath);
            }
        }

        public Dictionary<string, string> GetRemap()
        {
            return pathRemap;
        }

        public string GetVirtualPath(string physicPath)
        {
            string virtualPath;
            if (pathRemap.TryGetValue(physicPath, out virtualPath))
            {
                return virtualPath;
            }
            return physicPath;
        }

    }
}