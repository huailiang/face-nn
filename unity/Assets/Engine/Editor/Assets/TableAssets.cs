using UnityEngine;
using UnityEditor;
using System.IO;
using XEngine;

namespace XEngine.Editor
{
    internal class TableAssets
    {
        public static string tableNames = "";

        public static bool IsTable(string path)
        {
            return path.StartsWith(AssetsConfig.GlobalAssetsConfig.Table_Path) && path.EndsWith(".txt");
        }

        public static string GetTableName(string tablePath)
        {
            string tableName = tablePath.Replace(AssetsConfig.GlobalAssetsConfig.Table_Path, "");
            tableName = tableName.Replace(".txt", "");
            return tableName.Replace("\\", "/");
        }

        public static void ExeTable2Bytes(string tables, string arg0 = "-q -tables ")
        {
#if UNITY_EDITOR_WIN
            System.Diagnostics.Process exep = new System.Diagnostics.Process();
            exep.StartInfo.FileName = @"..\LQProject\Shell\Table2Bytes.exe";
            exep.StartInfo.Arguments = arg0 + tables;
            exep.StartInfo.CreateNoWindow = true;
            exep.StartInfo.UseShellExecute = false;
            exep.StartInfo.RedirectStandardOutput = true;
            exep.StartInfo.StandardOutputEncoding = System.Text.Encoding.Default;
            exep.Start();
            string output = exep.StandardOutput.ReadToEnd();
            exep.WaitForExit();
            if (output != "")
            {
                int errorIndex = output.IndexOf("error:");
                if (errorIndex >= 0)
                {
                    string errorStr = output.Substring(errorIndex);
                    Debug.LogError(errorStr);
                    Debug.Log(output.Substring(0, errorIndex));
                }
                else
                {
                    Debug.Log(output);
                }
            }
            AssetDatabase.Refresh();
#endif
        }

        public static bool DeleteTable(string path)
        {
            string tableName = GetTableName(path);
            string des = AssetsConfig.GlobalAssetsConfig.Table_Bytes_Path + tableName + ".bytes";
            if (File.Exists(des))
            {
                File.Delete(des);
                return true;
            }
            return false;
        }
    }
}