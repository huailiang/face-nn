using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEngine;
using CFEngine;

namespace CFEngine.Editor
{
    internal class CommonAssets
    {

        internal delegate void EnumAssetPreprocessCallback(string path);

        internal delegate bool EnumAssetImportCallback<T, I>(T obj, I assetImporter, string path)
        where T : UnityEngine.Object where I : UnityEditor.AssetImporter;

        internal delegate void EnumAssetCallback<T>(T obj, string path)
        where T : UnityEngine.Object;

        internal class ObjectInfo
        {
            public UnityEngine.Object obj = null;
            public string path = "";
        }
        internal interface IAssetLoadCallback
        {
            bool verbose { get; set; }
            List<ObjectInfo> GetObjects(string dir);

            void PreProcess(string path);
            bool Process(UnityEngine.Object asset, string path);
            void PostProcess(string path);
        }
        internal class BaseAssetLoadCallback<T> where T : UnityEngine.Object
        {
            public bool is_verbose = true;
            public string extFilter = "";
            public string extFilter1 = "";
            public string extFilter2 = "";
            protected List<ObjectInfo> m_Objects = new List<ObjectInfo>();
            private static string assetsRoot = "Assets/";
            public BaseAssetLoadCallback(string ext)
            {
                extFilter = ext;
            }
            public BaseAssetLoadCallback(string ext, string ext1)
            {
                extFilter = ext;
                extFilter1 = ext1;
            }
            public BaseAssetLoadCallback(string ext, string ext1, string ext2)
            {
                extFilter = ext;
                extFilter1 = ext1;
                extFilter2 = ext2;
            }

            public bool verbose { get { return is_verbose; } set { is_verbose = value; } }
            private void GetObjectsInfolder(FileInfo[] files)
            {
                for (int i = 0; i < files.Length; ++i)
                {
                    FileInfo file = files[i];
                    string fileName = file.FullName.Replace("\\", "/");
                    int index = fileName.IndexOf(assetsRoot);
                    fileName = fileName.Substring(index);
                    ObjectInfo oi = new ObjectInfo();
                    oi.path = fileName;
                    oi.obj = AssetDatabase.LoadAssetAtPath<T>(fileName);
                    m_Objects.Add(oi);
                }

            }
            private void GetObjectsInfolder(string path)
            {
                DirectoryInfo di = new DirectoryInfo(path);
                FileInfo[] files = di.GetFiles(extFilter, SearchOption.AllDirectories);
                GetObjectsInfolder(files);
                if (!string.IsNullOrEmpty(extFilter1))
                {
                    files = di.GetFiles(extFilter1, SearchOption.AllDirectories);
                    GetObjectsInfolder(files);
                }
                if (!string.IsNullOrEmpty(extFilter2))
                {
                    files = di.GetFiles(extFilter2, SearchOption.AllDirectories);
                    GetObjectsInfolder(files);
                }

            }

            private void GetObjectsInfolder(UnityEditor.DefaultAsset folder)
            {
                string path = AssetDatabase.GetAssetPath(folder);
                GetObjectsInfolder(path);
            }

            public List<ObjectInfo> GetObjects(string dir)
            {
                m_Objects.Clear();
                if (string.IsNullOrEmpty(dir))
                {
                    UnityEngine.Object[] objs = Selection.GetFiltered(typeof(UnityEngine.Object), SelectionMode.Assets);
                    for (int i = 0; i < objs.Length; ++i)
                    {
                        UnityEngine.Object obj = objs[i];
                        if (obj is UnityEditor.DefaultAsset)
                        {
                            GetObjectsInfolder(obj as UnityEditor.DefaultAsset);
                        }
                        else
                        {
                            if (obj is T)
                            {
                                string path = AssetDatabase.GetAssetPath(obj);
                                ObjectInfo oi = new ObjectInfo();
                                oi.obj = obj;
                                oi.path = path;
                                m_Objects.Add(oi);
                            }
                        }
                    }
                }
                else
                {
                    GetObjectsInfolder(dir);
                }
                return m_Objects;
            }
        }

        internal class AssetLoadCallback<T, I> : BaseAssetLoadCallback<T>, IAssetLoadCallback
        where T : UnityEngine.Object where I : UnityEditor.AssetImporter
        {
            public EnumAssetPreprocessCallback preprocess = null;
            public Func<T, I, string, bool> cb = null;

            public AssetLoadCallback(string ext) : base(ext) { }
            public AssetLoadCallback(string ext, string ext1) : base(ext, ext1) { }
            public AssetLoadCallback(string ext, string ext1, string ext2) : base(ext, ext1, ext2) { }
            public virtual void PreProcess(string path)
            {
                if (preprocess != null)
                {
                    preprocess(path);
                }
            }
            public virtual bool Process(UnityEngine.Object asset, string path)
            {
                T obj = asset as T;
                if (cb != null && obj != null)
                {
                    I assetImporter = AssetImporter.GetAtPath(path) as I;
                    return cb(obj, assetImporter, path);
                }
                return false;
            }

            public virtual void PostProcess(string path)
            {
                AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
            }
        }
        internal class AssetLoadCallback<T> : BaseAssetLoadCallback<T>, IAssetLoadCallback where T : UnityEngine.Object
        {
            public Action<T, string> cb = null;

            public AssetLoadCallback(string ext) : base(ext) { }
            public AssetLoadCallback(string ext, string ext1) : base(ext, ext1) { }
            public virtual void PreProcess(string path)
            {
            }
            public virtual bool Process(UnityEngine.Object asset, string path)
            {
                T obj = asset as T;
                if (cb != null && obj != null)
                {
                    cb(obj, path);
                }
                return false;
            }

            public virtual void PostProcess(string path) { }
        }

        internal delegate bool EnumFbxCallback<GameObject, ModelImporter>(GameObject fbx, ModelImporter modelImporter, string path);
        internal delegate bool EnumTex2DCallback<Texture2D, TextureImporter>(Texture2D tex, TextureImporter textureImporter, string path);

        internal static AssetLoadCallback<GameObject, ModelImporter> enumFbx = new AssetLoadCallback<GameObject, ModelImporter>("*.fbx");
        internal static AssetLoadCallback<Texture2D, TextureImporter> enumTex2D = new AssetLoadCallback<Texture2D, TextureImporter>("*.png", "*.tga", "*.exr");

        internal static AssetLoadCallback<GameObject> enumPrefab = new AssetLoadCallback<GameObject>("*.prefab");
        internal static AssetLoadCallback<TextAsset> enumTxt = new AssetLoadCallback<TextAsset>("*.bytes", "*.txt");
        internal static AssetLoadCallback<Material> enumMat = new AssetLoadCallback<Material>("*.mat");
        internal static AssetLoadCallback<Mesh> enumMesh = new AssetLoadCallback<Mesh>("*.asset");
        internal static AssetLoadCallback<AnimationClip> enumAnimationClip = new AssetLoadCallback<AnimationClip>("*.anim");
        internal static AssetLoadCallback<SceneAsset> enumSceneAsset = new AssetLoadCallback<SceneAsset>("*.unity");
        internal static void EnumAsset<T>(IAssetLoadCallback cb, string title, string dir = "") where T : UnityEngine.Object
        {
            if (cb != null)
            {
                List<ObjectInfo> objInfoLst = cb.GetObjects(dir);
                for (int i = 0; i < objInfoLst.Count; ++i)
                {
                    ObjectInfo oi = objInfoLst[i];
                    T asset = oi.obj as T;
                    if (asset != null)
                    {
                        cb.PreProcess(oi.path);
                        if (cb.Process(asset, oi.path))
                        {
                            cb.PostProcess(oi.path);
                        }
                    }
                    if (cb.verbose)
                        EditorUtility.DisplayProgressBar(string.Format("{0}-{1}/{2}", title, i, objInfoLst.Count), oi.path, (float)i / objInfoLst.Count);
                }
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();
                if (cb.verbose)
                {
                    EditorUtility.ClearProgressBar();
                    EditorUtility.DisplayDialog("Finish", "All assets processed finish", "OK");
                }
                cb.verbose = false;
            }
        }

        internal static void EnumScript(IAssetLoadCallback cb, string title, string dir = "")
        {
            UnityEngine.Object[] objs = Selection.GetFiltered(typeof(UnityEngine.Object), SelectionMode.Assets);
            foreach (UnityEngine.Object obj in objs)
            {
                Debug.LogError(obj.name);
            }
        }
        private static bool multiSave = false;
        public static void MultiSave(bool enable)
        {
            multiSave = enable;
            if (!multiSave)
            {
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();
            }
        }
        internal static T CreateAsset<T>(string path, string ext, UnityEngine.Object asset) where T : UnityEngine.Object
        {
            if (asset == null)
                return default(T);
            T existingAsset = null;

            var assetPath = path;

            if (asset is Texture2D)
            {
                Texture2D tex = asset as Texture2D;
                if (ext == ".asset")
                {
                    AssetDatabase.CreateAsset(asset, assetPath);
                }
                else
                {
                    byte[] png = tex.EncodeToPNG();
                    File.WriteAllBytes(assetPath, png);
                }

                // AssetDatabase.CreateAsset(asset, assetPath);
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();
                existingAsset = AssetDatabase.LoadAssetAtPath<T>(assetPath);
            }
            else if (asset is RenderTexture && typeof(T) == typeof(Texture2D))
            {
                RenderTexture rt = asset as RenderTexture;
                RenderTexture prev = RenderTexture.active;
                RenderTexture.active = rt;
                Texture2D png = new Texture2D(rt.width, rt.height, TextureFormat.ARGB32, false);
                png.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                byte[] bytes = png.EncodeToPNG();
                FileStream file = File.Open(assetPath, FileMode.Create);
                BinaryWriter writer = new BinaryWriter(file);
                writer.Write(bytes);
                file.Close();
                Texture2D.DestroyImmediate(png);
                png = null;
                RenderTexture.active = prev;
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();
                existingAsset = AssetDatabase.LoadAssetAtPath<T>(assetPath);

            }
            else
            {
                existingAsset = AssetDatabase.LoadAssetAtPath<T>(assetPath);
                if (existingAsset != null)
                {
                    if (existingAsset is Material)
                    {
                        Material mat = asset as Material;
                        Material src = existingAsset as Material;
                        src.shader = mat.shader;
                        src.CopyPropertiesFromMaterial(mat);
                        src.shaderKeywords = mat.shaderKeywords;
                    }
                    else
                    {
                        EditorUtility.SetDirty(asset);
                        EditorUtility.CopySerializedIfDifferent(asset, existingAsset);
                    }

                }
                else
                {
                    AssetDatabase.CreateAsset(asset, assetPath);
                    existingAsset = (T)asset;
                }
                if (!multiSave)
                {
                    if (existingAsset is ScriptableObject)
                    {
                        EditorUtility.SetDirty(existingAsset);
                    }
                    AssetDatabase.SaveAssets();
                    AssetDatabase.Refresh();
                }

            }

            return existingAsset;
        }

        internal static T CreateAsset<T>(string dirPath, string targetName, string ext, UnityEngine.Object asset) where T : UnityEngine.Object
        {
            return CreateAsset<T>(dirPath + "/" + targetName + ext, ext, asset);
        }

        internal static void DeleteAsset(UnityEngine.Object asset)
        {

            if (asset == null)
                return;

            string path = AssetDatabase.GetAssetPath(asset);
            AssetDatabase.DeleteAsset(path);
        }

        internal static string GetAssetFolder(UnityEngine.Object obj)
        {
            string matPath = AssetDatabase.GetAssetPath(obj);
            int index = matPath.LastIndexOf("/");
            if (index > 0)
            {
                return matPath.Substring(0, index);
            }
            return "";
        }

        internal static void StandardRender(Renderer render)
        {
            render.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            render.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;
            render.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
            render.receiveShadows = false;
            render.allowOcclusionWhenDynamic = false;
            if (render is SkinnedMeshRenderer)
            {
                SkinnedMeshRenderer smr = render as SkinnedMeshRenderer;
                smr.updateWhenOffscreen = false;
                smr.skinnedMotionVectors = false;
            }
        }
        internal static void BakeRender(Renderer render)
        {
            render.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;
            render.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.BlendProbes;
            render.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
            render.receiveShadows = true;
            render.allowOcclusionWhenDynamic = false;
            if (render is SkinnedMeshRenderer)
            {
                SkinnedMeshRenderer smr = render as SkinnedMeshRenderer;
                smr.updateWhenOffscreen = false;
                smr.skinnedMotionVectors = false;
            }
        }


        internal static SerializedProperty GetSerializeProperty(UnityEngine.Object obj, string name)
        {
            SerializedObject so = new SerializedObject(obj);
            return so.FindProperty(name);
        }

        internal static SerializedProperty GetSerializeProperty(SerializedObject so, string name)
        {
            return so.FindProperty(name);
        }

        internal static float GetSerializeValue(UnityEngine.Object obj, string name)
        {
            SerializedProperty sp = GetSerializeProperty(obj, name);
            if (sp != null)
            {
                return sp.floatValue;
            }
            return 0.0f;
        }

        internal static void SetSerializeValue(UnityEngine.Object obj, string name, float value)
        {
            SerializedProperty sp = GetSerializeProperty(obj, name);
            if (sp != null)
            {
                sp.floatValue = value;
                sp.serializedObject.ApplyModifiedProperties();
            }
        }

        internal static void SetSerializeValue(UnityEngine.Object obj, string name, UnityEngine.Object value)
        {
            SerializedProperty sp = GetSerializeProperty(obj, name);
            if (sp != null)
            {
                sp.objectReferenceValue = value;
                sp.serializedObject.ApplyModifiedProperties();
            }
        }
        internal static void SaveAsset(UnityEngine.Object obj)
        {
            if (obj != null)
            {
                EditorUtility.SetDirty(obj);
                AssetDatabase.SaveAssets();
            }
        }

        internal static bool IsSameProperty(Color color0, Color color1)
        {
            int deltaR = (int)(color0.r * 10) - (int)(color1.r * 10);
            int deltaG = (int)(color0.g * 10) - (int)(color1.g * 10);
            int deltaB = (int)(color0.b * 10) - (int)(color1.b * 10);
            int deltaA = (int)(color0.a * 10) - (int)(color1.a * 10);
            return deltaR == 0 && deltaG == 0 && deltaB == 0 && deltaA == 0;
        }
        internal static bool IsSameProperty(Vector4 vector0, Vector4 vector1)
        {
            int deltaX = (int)(vector0.x * 10) - (int)(vector1.x * 10);
            int deltaY = (int)(vector0.y * 10) - (int)(vector1.y * 10);
            int deltaZ = (int)(vector0.z * 10) - (int)(vector1.z * 10);
            int deltaW = (int)(vector0.w * 10) - (int)(vector1.w * 10);
            return deltaX == 0 && deltaY == 0 && deltaZ == 0 && deltaW == 0;
        }

    }
}