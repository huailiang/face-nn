using UnityEditor;
using UnityEngine;
using CFEngine;
using CFEngine.Editor;

/// <summary>
/// 时装材质生成工具
/// 1. 根据部位自动生成不同的材质
/// 2. 给材质赋贴图
/// 3. fbx 重置原始材质为pbr材质
/// </summary>

namespace XEditor
{
    public class FashionMatExport
    {

        const string role_shader = "Custom/PBS/Role";
        const string hair_shader = "Custom/Hair/HairTest01";
        const string face_shader = "Custom/PBS/Skin";


        [MenuItem(@"Assets/Engine/Fbx_CreateMaterial")]

        private static void Player_CreateMaterial()
        {
            CommonAssets.enumFbx.cb = (fbx, modelImporter, path) =>
            {
                string folder = path.Substring(0, path.LastIndexOf('/'));
                if (path.Contains("_common_"))
                {
                    int index = fbx.name.LastIndexOf("common_");
                    string mat_dir = folder + "/Materials_" + fbx.name.Substring(0, index + 6);
                    if (!AssetDatabase.IsValidFolder(mat_dir))
                    {
                        AssetDatabase.CreateFolder(folder, "Materials_" + fbx.name.Substring(0, index + 6));
                    }
                    modelImporter.importMaterials = true;
                    modelImporter.materialLocation = ModelImporterMaterialLocation.InPrefab;
                    ExtractMaterialsFromAsset(modelImporter, mat_dir);
                }
                else if (path.Contains("Player_"))
                {
                    int index = folder.LastIndexOf('/');
                    string dir = folder.Substring(index + 1);
                    string mat_dir = folder + "/Materials_" + dir;
                    if (!AssetDatabase.IsValidFolder(mat_dir))
                    {
                        AssetDatabase.CreateFolder(folder, "Materials_" + dir);
                    }
                    modelImporter.importMaterials = true;
                    modelImporter.materialLocation = ModelImporterMaterialLocation.InPrefab;
                    ExtractMaterialsFromAsset(modelImporter, mat_dir);
                }
                else
                {
                    Debug.LogError("invalid fbx " + path);
                }
                return true;
            };
            CommonAssets.EnumAsset<GameObject>(CommonAssets.enumFbx, "CreateMaterial");
        }


        [MenuItem(@"Assets/Engine/Fbx_AssignMaterial")]
        private static void Player_AssignMaterial()
        {
            CommonAssets.enumFbx.cb = (fbx, modelImporter, path) =>
            {
                string folder = path.Substring(0, path.LastIndexOf('/'));
                string mat_dir = string.Empty;
                if (path.Contains("_common_"))
                {
                    int index = fbx.name.LastIndexOf("common_");
                    mat_dir = folder + "/Materials_" + fbx.name.Substring(0, index + 6);
                }
                else if (path.Contains("Player_"))
                {
                    int index = folder.LastIndexOf('/');
                    string dir = folder.Substring(index + 1);
                    mat_dir = folder + "/Materials_" + dir;
                }
                if (!string.IsNullOrEmpty(mat_dir))
                {
                    SerializedObject serializedObject = new UnityEditor.SerializedObject(modelImporter);
                    SerializedProperty externalObjects = serializedObject.FindProperty("m_ExternalObjects");
                    SerializedProperty materials = serializedObject.FindProperty("m_Materials");
                    externalObjects.arraySize = 0;
                    for (int i = 0; i < materials.arraySize; ++i)
                    {
                        SerializedProperty arrayElementAtIndex = materials.GetArrayElementAtIndex(i);
                        string stringValue = arrayElementAtIndex.FindPropertyRelative("name").stringValue;
                        string stringValue2 = arrayElementAtIndex.FindPropertyRelative("type").stringValue;
                        string stringValue3 = arrayElementAtIndex.FindPropertyRelative("assembly").stringValue;
                        externalObjects.arraySize++;
                        SerializedProperty arrayElementAtIndex2 = externalObjects.GetArrayElementAtIndex(i);
                        arrayElementAtIndex2.FindPropertyRelative("first.name").stringValue = stringValue;
                        arrayElementAtIndex2.FindPropertyRelative("first.type").stringValue = stringValue2;
                        arrayElementAtIndex2.FindPropertyRelative("first.assembly").stringValue = stringValue3;

                        Material mat = AssetDatabase.LoadAssetAtPath<Material>(mat_dir + "/" + stringValue + ".mat");
                        arrayElementAtIndex2.FindPropertyRelative("second").objectReferenceValue = mat;
                        SetPBSMaterial(mat, mat_dir, stringValue);
                    }
                    serializedObject.ApplyModifiedProperties();
                    modelImporter.SearchAndRemapMaterials(ModelImporterMaterialName.BasedOnMaterialName, ModelImporterMaterialSearch.Local);
                    AssetDatabase.WriteImportSettingsIfDirty(path);
                }
                return true;
            };
            CommonAssets.EnumAsset<GameObject>(CommonAssets.enumFbx, "ssignMaterial");

        }


        internal static void ExtractMaterialsFromAsset(ModelImporter modelImporter, string destinationPath)
        {
            SerializedObject serializedObject = new UnityEditor.SerializedObject(modelImporter);
            SerializedProperty materials = serializedObject.FindProperty("m_Materials");
            for (int i = 0; i < materials.arraySize; ++i)
            {
                SerializedProperty arrayElementAtIndex = materials.GetArrayElementAtIndex(i);
                string stringValue = arrayElementAtIndex.FindPropertyRelative("name").stringValue;
                string sshader = stringValue.Contains("_hair_") ? hair_shader : role_shader;
                if (stringValue.Contains("_face_")) sshader = face_shader;
                Material mat = new Material(Shader.Find(sshader));
                mat.name = stringValue;
                Material newMat = AssetDatabase.LoadAssetAtPath<Material>(string.Format("{0}/{1}.mat", destinationPath, stringValue));
                if (newMat == null)
                {
                    newMat = CommonAssets.CreateAsset<Material>(destinationPath, stringValue, ".mat", mat);
                }
                MaterialShaderAssets.ClearMat(newMat);
                if (newMat != mat)
                {
                    UnityEngine.Object.DestroyImmediate(mat);
                }
            }
        }


        internal static void SetPBSMaterial(Material mat, string materialFolder, string materialName)
        {
            if (!materialName.Contains("_hair_"))
            {
                string baseTexPath = materialFolder + "/" + materialName + "_base.tga";
                Texture2D baseTex = AssetDatabase.LoadAssetAtPath<Texture2D>(baseTexPath);
                if (baseTex == null)
                {
                    baseTexPath = materialFolder + "/" + materialName + "_base_a.tga";
                    baseTex = AssetDatabase.LoadAssetAtPath<Texture2D>(baseTexPath);
                }
                mat.SetTexture("_BaseTex", baseTex);

                TextureImporter assetImporter = AssetImporter.GetAtPath(baseTexPath) as TextureImporter;
                if (assetImporter != null)
                {
                    MaterialShaderAssets.SetupMaterialWithBlendMode(mat, assetImporter.DoesSourceTextureHaveAlpha() ? BlendMode.Cutout : BlendMode.Opaque);
                }
                string pbsTexPath = materialFolder + "/" + materialName + "_pbs.tga";
                mat.SetTexture("_PBSTex", AssetDatabase.LoadAssetAtPath<Texture2D>(pbsTexPath));
            }
        }

    }
}