using UnityEditor;
using UnityEngine;
using System.IO;
using System;

namespace XEditor
{

    public class NeuralData
    {
        public float[] boneArgs;
        public Action callback;
        public RoleShape shape;
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
            string dep = "models";
            string file = EditorUtility.OpenFilePanel("Select model file", dep, "bytes");
            string _scene = string.Empty;
            if (file.Length != 0)
            {
                Debug.Log(file);
                FileStream fs = new FileStream(MODEL + "test.bytes", FileMode.Open, FileAccess.Read);
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
                    shape = shape
                };
                NeuralInput(data);
                br.Close();
                fs.Close();
            }
        }

        [MenuItem("Tools/Neural")]
        public static void NeuralInput()
        {
            float[] ar = new float[CNT];
            for (int i = 0; i < ar.Length; i++) ar[i] = 0.5f;
            NeuralData data = new NeuralData
            {
                callback = Capture,
                boneArgs = ar,
                shape = RoleShape.MALE
            };
            NeuralInput(data);
        }

        private static void NeuralInput(NeuralData data)
        {
            var win = EditorWindow.GetWindowWithRect(typeof(FashionPreview), new Rect(0, 0, 440, 640), true, "FashionPreview");
            win.Show();
            FashionPreview prev = win as FashionPreview;
            prev.NeuralProcess(data);
        }


        [MenuItem("Tools/Capture")]
        private static void Capture()
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
            SaveRenderTex(rt);
            Clear();
        }

        private static void Clear()
        {
            camera.targetTexture = null;
            RenderTexture.active = null;
            rt.Release();
        }


        private static void SaveRenderTex(RenderTexture rt)
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
                    File.WriteAllBytes(EXPORT + "export.jpg", bytes);
                    Debug.Log("save success");
                    HelperEditor.Open(EXPORT);
                }
                catch (IOException ex)
                {
                    Debug.Log("转换图片失败" + ex.Message);
                }
            }
        }

    }

}