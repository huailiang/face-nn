using UnityEditor;


namespace XEngine.Editor
{
    public class AssetsImporter : AssetPostprocessor
    {
        public static bool skipAutoImport = false;
        public void OnPreprocessModel()
        {
            if (!assetPath.ToLower().Contains("/test"))
            {
                ModelImporter modelImporter = (ModelImporter)assetImporter;
                if (!assetPath.StartsWith("Assets/Scenes"))
                    modelImporter.isReadable = false;
                modelImporter.importCameras = false;
                modelImporter.importLights = false;
                modelImporter.meshCompression = ModelImporterMeshCompression.Low;
            }
        }

        public void OnPreprocessAsset()
        {
            if (!assetPath.ToLower().Contains("/test"))
            {
                if (!skipAutoImport)
                {
                    if (assetPath.EndsWith(AssetsConfig.GlobalAssetsConfig.ReadableMeshSuffix))
                    {
                        MeshAssets.MakeMakeReadable(assetPath);
                    }
                }
            }
        }

        static void OnPostprocessAllAssets(string[] importedAssets, string[] deletedAssets, string[] movedAssets, string[] movedFromAssetPaths)
        {
            TableAssets.tableNames = "";
            bool deal = false;
            bool dealShader = false;
            if (importedAssets != null)
            {
                for (int i = 0; i < importedAssets.Length; ++i)
                {
                    string path = importedAssets[i];
                    if (TableAssets.IsTable(path))
                    {
                        TableAssets.tableNames += TableAssets.GetTableName(path);
                    }
                    else if (ShaderAssets.IsHLSLorCGINC(path))
                    {
                        dealShader = true;
                    }
                }
                if (dealShader)
                {
                    ShaderAssets.ReImportShader();
                    deal = true;
                }
                if (TableAssets.tableNames != "")
                {
                    TableAssets.ExeTable2Bytes(TableAssets.tableNames);
                    deal = true;
                }
            }
            for (int i = 0; i < deletedAssets.Length; ++i)
            {
                string path = deletedAssets[i];
                if (TableAssets.IsTable(path))
                {
                    deal |= TableAssets.DeleteTable(path);
                }
            }
            if (deal)
            {
                AssetDatabase.Refresh();
            }
        }

    }
}