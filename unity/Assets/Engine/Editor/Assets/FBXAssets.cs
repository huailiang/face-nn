using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace XEngine.Editor
{
    internal class FBXAssets
    {
        public static bool removeUV2 = false;
        public static bool removeColor = false;
        public static bool isNotReadable = true;

        [MenuItem(@"Assets/Engine/Fbx_CreateMaterial")]
        private static void Fbx_CreateMaterial()
        {
            CommonAssets.enumFbx.cb = (fbx, modelImporter, path) =>
            {
                string folderName;
                if (AssetsPath.GetCreatureFolderName(path, out folderName))
                {
                    string fileName;
                    if (AssetsPath.GetBandposeFileName(path, out fileName))
                    {
                        modelImporter.importMaterials = true;
                        modelImporter.materialLocation = ModelImporterMaterialLocation.InPrefab;
                        string bandposeFolder = AssetsPath.GetCreatureBandposePath(folderName);
                        string materialPath;
                        bool isSharedMaterials = AssetsPath.GetCreatureMaterialPath(path, folderName, fileName, out materialPath);
                        if (!AssetDatabase.IsValidFolder(materialPath))
                        {
                            if (isSharedMaterials)
                            {
                                AssetDatabase.CreateFolder(Path.GetDirectoryName(path), "Materials");
                            }
                            else
                            {
                                AssetDatabase.CreateFolder(bandposeFolder, AssetsPath.GetCreatureMaterialFolderName(fileName));
                            }

                        }
                        MaterialShaderAssets.ExtractMaterialsFromAsset(modelImporter, materialPath);

                    }
                }

                return true;
            };
            CommonAssets.EnumAsset<GameObject>(CommonAssets.enumFbx, "CreateMaterial");

        }

        [MenuItem(@"Assets/Engine/Fbx_AssignMaterial")]
        private static void Fbx_AssignMaterial()
        {
            CommonAssets.enumFbx.cb = (fbx, modelImporter, path) =>
            {
                string folderName;
                if (AssetsPath.GetCreatureFolderName(path, out folderName))
                {
                    string fileName;
                    if (AssetsPath.GetBandposeFileName(path, out fileName))
                    {
                        string materialPath;
                        AssetsPath.GetCreatureMaterialPath(path, folderName, fileName, out materialPath);
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

                            Material mat = AssetDatabase.LoadAssetAtPath<Material>(materialPath + "/" + stringValue + ".mat");
                            arrayElementAtIndex2.FindPropertyRelative("second").objectReferenceValue = mat;
                            MaterialShaderAssets.SetPBSMaterial(mat, materialPath, stringValue);
                        }
                        serializedObject.ApplyModifiedProperties();
                        modelImporter.SearchAndRemapMaterials(ModelImporterMaterialName.BasedOnMaterialName, ModelImporterMaterialSearch.Local);
                        AssetDatabase.WriteImportSettingsIfDirty(path);
                    }
                }

                return true;
            };
            CommonAssets.EnumAsset<GameObject>(CommonAssets.enumFbx, "ssignMaterial");

        }

        [MenuItem("Assets/Engine/Fbx_ExportAnim")]
        static void Fbx_ExportAnim()
        {
            CommonAssets.enumFbx.cb = (fbx, modelImporter, path) =>
            {
                string folderName;
                if (AssetsPath.GetCreatureFolderName(path, out folderName))
                {
                    if (Directory.Exists(AssetsConfig.GlobalAssetsConfig.ResourceAnimationPath) == false)
                    {
                        AssetDatabase.CreateFolder(AssetsConfig.GlobalAssetsConfig.ResourcePath, AssetsConfig.GlobalAssetsConfig.ResourceAnimation);
                    }
                    if (modelImporter.animationCompression != ModelImporterAnimationCompression.Optimal)
                    {

                        modelImporter.animationCompression = ModelImporterAnimationCompression.Optimal;
                    }
                    UnityEngine.Object[] allObjects = AssetDatabase.LoadAllAssetsAtPath(path);

                    string targetPath = string.Format("{0}/{1}", AssetsConfig.GlobalAssetsConfig.ResourceAnimationPath, folderName);

                    if (!Directory.Exists(targetPath))
                    {
                        AssetDatabase.CreateFolder(AssetsConfig.GlobalAssetsConfig.ResourceAnimationPath, folderName);
                    }
                    foreach (UnityEngine.Object o in allObjects)
                    {
                        AnimationClip oClip = o as AnimationClip;

                        if (oClip == null || oClip.name.StartsWith("__preview__Take 001")) continue;
                        string copyPath = targetPath + "/" + oClip.name + ".anim";
                        AnimationClip newClip = new AnimationClip();

                        EditorUtility.CopySerializedIfDifferent(oClip, newClip);

                        AssetDatabase.CreateAsset(newClip, copyPath);
                    }
                }

                return false;
            };
            CommonAssets.EnumAsset<GameObject>(CommonAssets.enumFbx, "ExportAnim");
        }

        [MenuItem(@"Assets/Engine/Fbx_ReduceKeyFrame")]
        private static void ReduceKeyFrame()
        {
            CommonAssets.enumAnimationClip.cb = (animClip, path) =>
            {
                string errorLog = "";
                int reduceCurveCount = 0;
                bool isMount = path.Contains("_Mount_");
                List<int> removeIndex = new List<int>();
                EditorCurveBinding[] curveBinding = AnimationUtility.GetCurveBindings(animClip);
                for (int i = 0; i < curveBinding.Length; ++i)
                {
                    var binding = curveBinding[i];
                    AnimationCurve curve = AnimationUtility.GetEditorCurve(animClip, binding);
                    if (curve.keys.Length > 1)
                    {
                        removeIndex.Clear();
                        bool scale = binding.propertyName.StartsWith("m_LocalScale");
                        bool pos = binding.propertyName.StartsWith("m_LocalPosition");

                        Keyframe[] keys = new Keyframe[curve.keys.Length];
                        for (int j = 0; j < curve.keys.Length; ++j)
                        {
                            Keyframe key = curve.keys[j];
                            if (scale || pos)
                            {
                                key.value = (float)Math.Round(key.value, 4);
                            }
                            else
                            {
                                key.value = key.value;
                            }
                            key.inTangent = (float)Math.Round(key.inTangent, 3);
                            key.outTangent = (float)Math.Round(key.outTangent, 3);
                            keys[j] = key;
                        }

                        Keyframe preKey = keys[0];
                        Keyframe midKey = keys[1];

                        float defaultValue = scale ? 1 : 0;
                        bool isDefaultValue = Mathf.Abs(preKey.value - defaultValue) < 0.01f && Mathf.Abs(midKey.value - defaultValue) < 0.01f;
                        string floatFormat = "f3";
                        for (int j = 2; j < keys.Length; ++j)
                        {
                            Keyframe key = keys[j];
                            key.value = float.Parse(key.value.ToString(floatFormat));
                            key.inTangent = float.Parse(key.inTangent.ToString(floatFormat));
                            key.outTangent = float.Parse(key.outTangent.ToString(floatFormat));

                            if (ComputerKeyDerivative(preKey, midKey, key))
                            {
                                removeIndex.Add(j - 1);
                            }

                            float defaultError = Mathf.Abs(key.value - defaultValue);
                            float defaultErrorPercent = scale ? defaultError / defaultValue : defaultError;
                            if (defaultErrorPercent > 0.01f)
                                isDefaultValue = false;

                            preKey = midKey;
                            midKey = key;
                        }
                        curve.keys = keys;

                        if (isDefaultValue)
                        {
                            if (isMount && pos && binding.propertyName == "m_LocalPosition.z" && !binding.path.Contains("/"))
                            {
                                Debug.Log("not opt curve" + binding.path);
                            }
                            else
                            {
                                AnimationUtility.SetEditorCurve(animClip, binding, null);
                                reduceCurveCount++;
                                if (!(pos || scale))
                                {
                                    errorLog += string.Format("{0}:{1}\r\n", binding.path, binding.propertyName);
                                }
                            }
                        }
                        else
                        {
                            for (int j = removeIndex.Count - 1; j >= 0; --j)
                            {
                                curve.RemoveKey(removeIndex[j]);
                            }
                            if (removeIndex.Count > 0)
                                reduceCurveCount++;

                            AnimationUtility.SetEditorCurve(animClip, binding, curve);
                        }

                    }
                    else
                    {
                        AnimationUtility.SetEditorCurve(animClip, binding, null);
                    }
                }
                if (reduceCurveCount > 0)
                {
                    Debug.LogWarning(string.Format("{0} reduceCurveCount/total:{1}/{2}\r\n{3}", path, reduceCurveCount, curveBinding.Length, errorLog));
                }
            };
            CommonAssets.EnumAsset<AnimationClip>(CommonAssets.enumAnimationClip, "ReduceKeyFrame");
        }

        private static bool ComputerKeyDerivative(Keyframe preKey, Keyframe midKey, Keyframe currentKey)
        {
            float ddx0 = (midKey.value - preKey.value);
            float ddx1 = (currentKey.value - midKey.value);
            float derivativeValue = Mathf.Abs(ddx1 - ddx0);

            float ddxInTangent0 = midKey.inTangent - preKey.inTangent;
            float ddxInTangent1 = currentKey.inTangent - midKey.inTangent;
            float derivativeInTangent = Mathf.Abs(ddxInTangent1 - ddxInTangent0);

            float ddxOutTangent0 = midKey.outTangent - preKey.outTangent;
            float ddxOutTangent1 = currentKey.outTangent - midKey.outTangent;
            float derivativeOutTangent = Mathf.Abs(ddxOutTangent1 - ddxOutTangent0);

            return derivativeValue < 0.01f &&
            derivativeInTangent < 0.01f &&
            derivativeOutTangent < 0.01f;
        }

        public static void GetMeshMat(Renderer render, out Mesh newMesh, out Material newMat, bool exportMesh = true, bool exportMat = true, bool mirror = false)
        {
            Mesh mesh = null;
            Material mat = null;
            newMat = null;
            newMesh = null;
            if (render is MeshRenderer)
            {
                MeshRenderer mr = render as MeshRenderer;
                MeshFilter mf = render.GetComponent<MeshFilter>();
                mesh = mf != null ? mf.sharedMesh : null;
                mat = mr.sharedMaterial;
            }
            else if (render is SkinnedMeshRenderer)
            {
                SkinnedMeshRenderer smr = render as SkinnedMeshRenderer;
                mesh = smr != null ? smr.sharedMesh : null;
                mat = smr.sharedMaterial;
            }
            if (exportMesh && mesh != null)
            {
                newMesh = UnityEngine.Object.Instantiate<Mesh>(mesh);
                newMesh.name = mesh.name;
                if (removeUV2)
                    newMesh.uv2 = null;
                newMesh.uv3 = null;
                newMesh.uv4 = null;
                if (removeColor)
                    newMesh.colors = null;
                if (mirror)
                {
                    int[] index = newMesh.triangles;
                    for (int i = 0; i < index.Length; i += 3)
                    {
                        int tmp = index[i + 2];
                        index[i + 2] = index[i + 1];
                        index[i + 1] = tmp;
                    }
                    newMesh.triangles = index;
                }
                MeshUtility.SetMeshCompression(newMesh, ModelImporterMeshCompression.Low);
                MeshUtility.Optimize(newMesh);
                newMesh.UploadMeshData(isNotReadable);
            }
            if (exportMat && mat != null)
            {
                newMat = new Material(mat);
                newMat.name = mat.name;
                MaterialShaderAssets.ClearMat(newMat);
            }
        }

        internal static void ExportMesh(GameObject prefab, string dir, string name)
        {
        }

        private static void ApplyModfy(string path)
        {
            AssetDatabase.WriteImportSettingsIfDirty(path);
            AssetDatabase.StartAssetEditing();
            AssetDatabase.ImportAsset(path);
            AssetDatabase.StopAssetEditing();
        }

        internal static void SceneMeshExport(string path, ModelImporter modelImporter)
        {
            GameObject fbx = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (fbx != null)
                ExportMesh(fbx, string.Format("{0}{1}", AssetsConfig.GlobalAssetsConfig.ResourcePath, AssetsConfig.GlobalAssetsConfig.EditorSceneRes), fbx.name);
        }




        [MenuItem(@"Assets/Engine/Fbx_ExportAvatar")]
        private static void Fbx_ExportAvatar()
        {
            CommonAssets.enumFbx.cb = (fbx, modelImporter, path) =>
            {
                Animator ator = fbx.GetComponent<Animator>();
                if (ator != null && ator.avatar != null)
                {
                    string avatarPath = string.Format("{0}/Avatar/{1}.asset", AssetsConfig.GlobalAssetsConfig.Creature_Path, ator.avatar.name);
                    Avatar avatar = UnityEngine.Object.Instantiate<Avatar>(ator.avatar);
                    avatar.name = ator.avatar.name;
                    CommonAssets.CreateAsset<Avatar>(avatarPath, ".asset", avatar);
                }
                return false;
            };
            CommonAssets.EnumAsset<GameObject>(CommonAssets.enumFbx, "ExportAvatar");
        }

        internal static void Fbx_BindAvatar(string dir, GameObject srcFbx, bool isHuman)
        {
            if (Directory.Exists(dir))
            {
                Animator ator = srcFbx.GetComponent<Animator>();
                if (ator != null && ator.avatar != null)
                {
                    CommonAssets.enumFbx.cb = (fbx, modelImporter, path) =>
                    {
                        if (srcFbx != fbx)
                        {
                            string srcAssetPath = AssetDatabase.GetAssetPath(srcFbx);
                            ModelImporter importer = AssetImporter.GetAtPath(srcAssetPath) as ModelImporter;
                            if (importer != null)
                            {
                                SerializedObject so = new SerializedObject(modelImporter);
                                SerializedProperty sp = CommonAssets.GetSerializeProperty(so, "m_AnimationType");
                                sp.intValue = isHuman ? (int)ModelImporterAnimationType.Human : (int)ModelImporterAnimationType.Generic;
                                if (isHuman)
                                {
                                    sp = CommonAssets.GetSerializeProperty(so, "m_CopyAvatar");
                                    sp.boolValue = true;
                                    sp = CommonAssets.GetSerializeProperty(so, "m_LastHumanDescriptionAvatarSource");
                                    sp.objectReferenceValue = ator.avatar;
                                }

                                SerializedObject srcSo = new SerializedObject(importer);
                                so.CopyFromSerializedProperty(srcSo.FindProperty("m_HumanDescription"));
                                so.ApplyModifiedPropertiesWithoutUndo();
                                ApplyModfy(path);
                            }

                        }

                        return false;
                    };
                    CommonAssets.EnumAsset<GameObject>(CommonAssets.enumFbx, "BindAvatar", dir);
                }
            }
        }

        internal static void Fbx_BindAnimation(string dir, AvatarMask avatarMask, bool isHuman)
        {
            CommonAssets.enumFbx.cb = (fbx, modelImporter, path) =>
            {
                if (path.ToLower().Contains("/animation/"))
                {
                    SerializedObject so = new SerializedObject(modelImporter);
                    SerializedProperty sp = CommonAssets.GetSerializeProperty(so, "m_ClipAnimations");
                    if (sp.arraySize == 0)
                    {
                        sp.InsertArrayElementAtIndex(0);
                    }
                    if (sp.arraySize == 1)
                    {
                        var takes = modelImporter.defaultClipAnimations;
                        if (takes.Length > 0)
                        {
                            var takeInfo = takes[0];

                            SerializedProperty clip = sp.GetArrayElementAtIndex(0);
                            SerializedProperty start = clip.FindPropertyRelative("firstFrame");
                            start.floatValue = takeInfo.firstFrame;
                            SerializedProperty end = clip.FindPropertyRelative("lastFrame");
                            end.floatValue = takeInfo.lastFrame;
                        }
                        SerializedProperty clipSp = sp.GetArrayElementAtIndex(0);
                        var subsp = clipSp.FindPropertyRelative("name");
                        subsp.stringValue = fbx.name;
                        if (isHuman)
                        {
                            subsp = clipSp.FindPropertyRelative("loopBlendOrientation");
                            subsp.boolValue = true;
                            subsp = clipSp.FindPropertyRelative("loopBlendPositionY");
                            subsp.boolValue = true;
                            subsp = clipSp.FindPropertyRelative("loopBlendPositionXZ");
                            subsp.boolValue = true;
                            subsp = clipSp.FindPropertyRelative("keepOriginalOrientation");
                            subsp.boolValue = true;
                            subsp = clipSp.FindPropertyRelative("keepOriginalPositionY");
                            subsp.boolValue = true;
                            subsp = clipSp.FindPropertyRelative("keepOriginalPositionXZ");
                            subsp.boolValue = true;
                            subsp = clipSp.FindPropertyRelative("heightFromFeet");
                            subsp.boolValue = false;
                            subsp = clipSp.FindPropertyRelative("maskType");
                            subsp.intValue = 1;
                            subsp = clipSp.FindPropertyRelative("maskSource");
                            subsp.objectReferenceValue = avatarMask;

                            SerializedProperty bodyMask = clipSp.FindPropertyRelative("bodyMask");

                            if (bodyMask != null && bodyMask.isArray)
                            {
                                for (AvatarMaskBodyPart i = 0; i < AvatarMaskBodyPart.LastBodyPart; i++)
                                {
                                    if ((int)i >= bodyMask.arraySize) bodyMask.InsertArrayElementAtIndex((int)i);
                                    bodyMask.GetArrayElementAtIndex((int)i).intValue = avatarMask.GetHumanoidBodyPartActive(i) ? 1 : 0;
                                }
                            }

                            SerializedProperty transformMask = clipSp.FindPropertyRelative("transformMask");
                            EditorCommon.CallInternalFunction(typeof(ModelImporter), "UpdateTransformMask", true, true, false, null, new object[] { avatarMask, transformMask });
                        }

                        so.ApplyModifiedPropertiesWithoutUndo();
                        ApplyModfy(path);
                    }
                }
                return false;
            };
            CommonAssets.EnumAsset<GameObject>(CommonAssets.enumFbx, "BindAnimation", dir);
        }

    }
}