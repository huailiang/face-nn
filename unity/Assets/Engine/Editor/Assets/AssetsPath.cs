using UnityEngine;
using UnityEditor;
using System.IO;

namespace CFEngine.Editor
{
    public class AssetsPath
    {
        public enum EFolderType
        {
            ECreatures,
            ELut,
            ESkinLookup,
            EEnvCube,
            EUI,
            ECommon
        }


        public static string GetCreatureBandposePath(string bandposeName)
        {
            return string.Format(AssetsConfig.GlobalAssetsConfig.Creature_Bandpose_Format_Path, AssetsConfig.GlobalAssetsConfig.Creature_Path, bandposeName);
        }

        public static bool GetCreatureMaterialPath(string path, string bandposeName, string creatureName, out string matPath)
        {
            string folderName = GetParentFolderName(path);
            if (folderName.StartsWith(AssetsConfig.GlobalAssetsConfig.SharedMaterialPrefix))
            {
                matPath = string.Format(AssetsConfig.GlobalAssetsConfig.Creature_SharedMaterial_Format_Path, AssetsConfig.GlobalAssetsConfig.Creature_Path, bandposeName, folderName);
                return true;
            }
            else
            {
                matPath = string.Format(AssetsConfig.GlobalAssetsConfig.Creature_Material_Format_Path, AssetsConfig.GlobalAssetsConfig.Creature_Path, bandposeName, creatureName);
                return false;
            }
        }

        public static string GetCreatureMaterialFolderName(string creatureName)
        {
            return string.Format(AssetsConfig.GlobalAssetsConfig.Creature_Material_Format_Folder, creatureName);
        }

        public static bool GetCreatureFolderName(string path, out string folderName)
        {
            folderName = "";
            path = path.Substring(AssetsConfig.GlobalAssetsConfig.Creature_Path.Length + 1);
            int index = path.IndexOf("/");
            if (index >= 0)
            {
                folderName = path.Substring(0, index);
                return true;
            }
            return false;
        }
        public static string GetParentFolderName(string path)
        {
            string folder = Path.GetDirectoryName(path);
            int index = folder.LastIndexOf("/");
            if (index >= 0)
            {
                folder = folder.Substring(index + 1);
                return folder;
            }
            return "";
        }
        public static bool GetBandposeFileName(string path, out string fileName)
        {
            fileName = "";
            string ext = Path.GetExtension(path).ToLower();
            if (ext == AssetsConfig.GlobalAssetsConfig.Fbx_Ext)
            {
                fileName = Path.GetFileName(path);
                int extIndex = fileName.LastIndexOf(".");
                fileName = fileName.Substring(0, extIndex);
                return Path.GetDirectoryName(path).Contains(AssetsConfig.GlobalAssetsConfig.Bandpose_Str);
            }
            return false;
        }

        public static bool GetFileName(string path, out string fileName)
        {
            fileName = "";
            int index = path.LastIndexOf(".");
            if (index > 0)
            {
                path = path.Substring(0, index);
                index = path.LastIndexOf("/");
                if (index > 0)
                {
                    fileName = path.Substring(index + 1);
                }
                else
                {
                    fileName = path;
                }
                return true;
            }
            return false;
        }
        public static string GetAssetRelativePath(UnityEngine.Object obj)
        {
            if (obj != null)
            {
                string path = AssetDatabase.GetAssetPath(obj);
                if (path.StartsWith(AssetsConfig.GlobalAssetsConfig.ResourcePath))
                {
                    path = path.Substring(AssetsConfig.GlobalAssetsConfig.ResourcePath.Length + 1);
                    int index = path.LastIndexOf(".");
                    if (index >= 0)
                    {
                        path = path.Substring(0, index);
                    }
                    return path;
                }
            }
            return "";
        }
    }
}