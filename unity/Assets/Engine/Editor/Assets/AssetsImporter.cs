using UnityEditor;
using CFEngine;

namespace CFEngine.Editor
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
            for (int i = 0; i < AssetsConfig.GlobalAssetsConfig.MehsAutoExportType.Length; ++i)
            {
                string type = AssetsConfig.GlobalAssetsConfig.MehsAutoExportType[i];
                if (assetPath.StartsWith(type) && i < FBXAssets.modelAutoExportFun.Length)
                {
                    FBXAssets.MeshAutoExport fun = FBXAssets.modelAutoExportFun[i];
                    if (fun != null)
                    {
                        fun(assetPath, assetImporter as ModelImporter);
                    }
                }
            }

        }

        public void OnPreprocessTexture()
        {
            if (!assetPath.ToLower().Contains("/test"))
            {
                if (!skipAutoImport)
                {
                    TextureImporter textureImporter = assetImporter as TextureImporter;
                    //creatures
                    TextureAssets.SetTextureConfig(assetPath, textureImporter);
                }
            }
        }

        public void OnPreprocessAsset()
        {
            if (!assetPath.ToLower().Contains("/test"))
            {
                if (!skipAutoImport)
                {
                    if (assetPath.EndsWith(AssetsConfig.GlobalAssetsConfig.SpriteAtlasExt))
                    {
                        TextureAssets.SetSpriteAtlasConfig(assetPath);
                    }
                    else if (assetPath.EndsWith(AssetsConfig.GlobalAssetsConfig.ReadableMeshSuffix))
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
                    else if (MaterialShaderAssets.IsHLSLorCGINC(path))
                    {
                        dealShader = true;
                    }
                }
                if (dealShader)
                {
                    MaterialShaderAssets.ReImportShader();
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