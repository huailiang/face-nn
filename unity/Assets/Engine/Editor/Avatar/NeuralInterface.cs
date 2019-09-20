using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace XEngine.Editor
{

    public class NeuralData
    {
        public float[] boneArgs;
        public Action<string> callback;
        public RoleShape shape;
        public string name;
    }


    public class NeuralInterface
    {
        static RenderTexture rt;
        static Camera camera;
        static string export;
        static string model;
        const int CNT = 95;

        static string EXPORT
        {
            get
            {
                if (string.IsNullOrEmpty(export))
                {
                    export = Application.dataPath;
                    int i = export.IndexOf("unity/Assets");
                    export = export.Substring(0, i) + "export/";
                }
                return export;
            }
        }

        static string MODEL
        {
            get
            {
                if (string.IsNullOrEmpty(model))
                {
                    model = Application.dataPath;
                    int idx = model.IndexOf("/Assets");
                    model = model.Substring(0, idx);
                    model = model + "/models/";
                }
                return model;
            }
        }


        [MenuItem("Tools/Select")]
        public static void Select()
        {
            SetupEnv();
            string file = EditorUtility.OpenFilePanel("Select model file", MODEL, "bytes");
            FileInfo info = new FileInfo(file);
            ProcessFile(info);
            HelperEditor.Open(EXPORT);
        }


        [MenuItem("Tools/Batch")]
        public static void Batch()
        {
            DirectoryInfo dir = new DirectoryInfo(MODEL);
            var files = dir.GetFiles("*.bytes");
            for (int i = 0; i < files.Length; i++)
            {
                ProcessFile(files[i]);
            }
            HelperEditor.Open(EXPORT);
        }


        private static void ProcessFile(FileInfo info)
        {
            if (info != null)
            {
                string file = info.FullName;
                FileStream fs = new FileStream(file, FileMode.Open, FileAccess.Read);
                float[] args = new float[CNT];
                BinaryReader br = new BinaryReader(fs);
                RoleShape shape = (RoleShape)br.ReadInt32();
                for (int i = 0; i < CNT; i++)
                {
                    args[i] = br.ReadSingle();
                }
                NeuralData data = new NeuralData
                {
                    callback = Capture,
                    boneArgs = args,
                    shape = shape,
                    name = info.Name.Replace(".bytes", "")
                };
                NeuralInput(data);
                br.Close();
                fs.Close();
            }
        }
      

        private static void NeuralInput(NeuralData data)
        {
            var prev = ScriptableObject.CreateInstance<FashionPreview>();
            prev.NeuralProcess(data);
        }


        [MenuItem("Tools/SetupEnv")]
        private static void SetupEnv()
        {
            EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects);
            GameObject oldCamera = GameObject.Find(@"Main Camera");
            GameObject.DestroyImmediate(oldCamera);
            GameObject cam = AssetDatabase.LoadAssetAtPath("Assets/Engine/Editor/EditorResources/Main Camera.prefab", typeof(GameObject)) as GameObject;
            cam = GameObject.Instantiate<GameObject>(cam, null);
            cam.transform.position = new Vector3(0, 1, -10);
            Light light = GameObject.Find("Directional Light").GetComponent<Light>();
            light.transform.parent = cam.transform;
            var env = cam.GetComponent<XEngine.Environment>();
            env.roleLight0 = light;
            env.Update();
        }


        private static void Capture(string name)
        {
            if (camera == null)
                camera = GameObject.FindObjectOfType<Camera>();
            if (rt == null)
            {
                string path = "Assets/BundleRes/Config/CameraOuput.renderTexture";
                rt = AssetDatabase.LoadAssetAtPath<RenderTexture>(path);
            }

            camera.targetTexture = rt;
            camera.Render();
            SaveRenderTex(rt, name);
            Clear();
        }


        private static void Clear()
        {
            camera.targetTexture = null;
            RenderTexture.active = null;
            rt.Release();
        }


        private static void SaveRenderTex(RenderTexture rt, string name)
        {
            RenderTexture.active = rt;
            Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            byte[] bytes = tex.EncodeToJPG();
            if (bytes != null && bytes.Length > 0)
            {
                try
                {
                    if (!Directory.Exists(EXPORT))
                    {
                        Directory.CreateDirectory(EXPORT);
                    }
                    File.WriteAllBytes(EXPORT + name + ".jpg", bytes);
                }
                catch (IOException ex)
                {
                    Debug.Log("转换图片失败" + ex.Message);
                }
            }
        }

    }

}