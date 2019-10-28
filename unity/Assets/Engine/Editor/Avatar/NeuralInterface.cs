using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace XEngine.Editor
{
    public class NeuralData
    {
        public float[] boneArgs;
        public Action<string, RoleShape> callback;
        public RoleShape shape;
        public string name;
    }

    public class NeuralInterface : EditorWindow
    {
        static RenderTexture rt;
        static Camera camera;
        static string export;
        static string model;
        const int CNT = 95;
        static Connect connect;
        static FashionPreview prev;

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

        RoleShape shape = RoleShape.FEMALE;
        bool complete = true;
        int datacnt = 10000;
        float weight = 0.4f;

        private void OnGUI()
        {
            GUILayout.BeginVertical();
            GUILayout.Label("Generate Database", XEditorUtil.titleLableStyle);
            GUILayout.Space(12);

            GUILayout.BeginHorizontal();
            GUILayout.Label("Role Shape");
            shape = (RoleShape)EditorGUILayout.EnumPopup(shape, GUILayout.Width(120));
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            GUILayout.Label("complate show");
            complete = GUILayout.Toggle(complete, "");
            GUILayout.EndHorizontal();

            GUILayout.BeginHorizontal();
            GUILayout.Label("data capacity");
            weight = GUILayout.HorizontalSlider(weight, 0, 1, GUILayout.Width(120));
            GUILayout.EndHorizontal();

            GUILayout.Label("  database " + (int)(datacnt * weight));
            GUILayout.Label("  trainset  " + (int)(datacnt * weight * 0.8));
            GUILayout.Label("  testset   " + (int)(datacnt * weight * 0.2));
            GUILayout.Space(8);
            if (GUILayout.Button("Generate"))
            {
                RandomExportModels((int)(datacnt * weight * 0.8), shape, "trainset", true, complete);
                RandomExportModels((int)(datacnt * weight * 0.2), shape, "testset", false, complete);
                EditorUtility.Open(EXPORT);
            }
            GUILayout.EndVertical();
        }


        [MenuItem("Tools/SelectModel")]
        public static void Model2Image()
        {
            XEditorUtil.SetupEnv();
            string file = UnityEditor.EditorUtility.OpenFilePanel("Select model file", MODEL, "bytes");
            FileInfo info = new FileInfo(file);
            ProcessFile(info, true);
            MoveDestDir("model_*", "regular/");
            EditorUtility.Open(EXPORT + "regular/");
        }

        [MenuItem("Tools/SelectPicture")]
        public static void Picture2Model()
        {
            XEditorUtil.SetupEnv();
            string picture = UnityEditor.EditorUtility.OpenFilePanel("Select model file", EXPORT, "jpg");
            int idx = picture.LastIndexOf('/') + 1;
            string descript = picture.Substring(0, idx) + "db_description";
            if (!string.IsNullOrEmpty(descript))
            {
                string key = picture.Substring(idx).Replace(".jpg", "");
                FileInfo info = new FileInfo(descript);
                FileStream fs = new FileStream(descript, FileMode.Open, FileAccess.Read);
                BinaryReader reader = new BinaryReader(fs);
                int cnt = reader.ReadInt32();
                float[] args = new float[CNT];
                while (cnt-- > 0)
                {
                    string name = reader.ReadString();
                    for (int i = 0; i < CNT; i++) args[i] = reader.ReadSingle();
                    string str = string.Empty;
                    if (name == key)
                    {
                        int shape = int.Parse(name[name.Length - 1].ToString());
                        for (int i = 0; i < CNT; i++) str += args[i].ToString("f3") + " ";
                        Debug.Log(str);
                        NeuralData data = new NeuralData
                        {
                            callback = Capture,
                            boneArgs = args,
                            shape = (RoleShape)shape,
                            name = name
                        };
                        NeuralInput(data, true);
                        break;
                    }
                }
                reader.Close();
                fs.Close();
            }
        }


        [MenuItem("Tools/BatchExportModels")]
        public static void BatchModels()
        {
            XEditorUtil.SetupEnv();
            DirectoryInfo dir = new DirectoryInfo(MODEL);
            var files = dir.GetFiles("*.bytes");
            for (int i = 0; i < files.Length; i++)
            {
                ProcessFile(files[i], true);
            }
            MoveDestDir("model_*", "regular/");
            EditorUtility.Open(EXPORT + "regular/");
        }


        [MenuItem("Tools/GenerateDatabase")]
        private static void GenerateDatabase2()
        {
            var window = EditorWindow.GetWindowWithRect<NeuralInterface>(new Rect(0, 0, 320, 400));
            window.Show();
        }


        private static void RandomExportModels(int expc, RoleShape shape, string prefix, bool noise, bool complete)
        {
            XEditorUtil.SetupEnv();
            float[] args = new float[CNT];

            FileStream fs = new FileStream(EXPORT + "db_description", FileMode.OpenOrCreate, FileAccess.Write);
            BinaryWriter bw = new BinaryWriter(fs);
            bw.Write(expc);
            for (int j = 0; j < expc; j++)
            {
                string name = string.Format("db_{0:0000}_{1}", j, (int)shape);
                bw.Write(name);
                for (int i = 0; i < CNT; i++)
                {
                    args[i] = UnityEngine.Random.Range(0.0f, 1.0f);
                    bw.Write(noise ? AddNoise(args[i], i) : args[i]);
                }
                NeuralData data = new NeuralData
                {
                    callback = Capture,
                    boneArgs = args,
                    shape = shape,
                    name = name
                };
                UnityEditor.EditorUtility.DisplayProgressBar(prefix, string.Format("is generating {0}/{1}", j, expc), (float)j / expc);
                NeuralInput(data, complete);
            }
            UnityEditor.EditorUtility.DisplayProgressBar(prefix, "post processing, wait for a moment", 1);
            bw.Close();
            fs.Close();
            MoveDestDir("db_*", prefix + "_" + shape.ToString().ToLower() + "/");
            UnityEditor.EditorUtility.ClearProgressBar();
        }

        private static float AddNoise(float arg, int indx)
        {
            int rnd = UnityEngine.Random.Range(0, CNT);
            if (indx == rnd)
            {
                rnd = UnityEngine.Random.Range(-10, 10);
                return ((arg * 80) + 10 + rnd) / 100.0f;
            }
            return arg;
        }


        private static void MoveDestDir(string pattern, string sub, bool delete = true)
        {
            try
            {
                var path = EXPORT + sub;
                if (Directory.Exists(path))
                {
                    if (delete) Directory.Delete(path, true);
                }
                if (!Directory.Exists(path)) Directory.CreateDirectory(path);
                DirectoryInfo dir = new DirectoryInfo(EXPORT);
                var files = dir.GetFiles(pattern);
                for (int i = 0; i < files.Length; i++)
                {
                    files[i].MoveTo(path + files[i].Name);
                }
            }
            catch (Exception e)
            {
                Debug.LogError(e.Message + "\n" + e.StackTrace);
                UnityEditor.EditorUtility.ClearProgressBar();
            }
        }


        private static void ProcessFile(FileInfo info, bool complete)
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
                    name = "model_" + info.Name.Replace(".bytes", "")
                };
                NeuralInput(data, complete);
                br.Close();
                fs.Close();
            }
        }


        private static void NeuralInput(NeuralData data, bool complete)
        {
            if (prev == null) prev = ScriptableObject.CreateInstance<FashionPreview>();
            prev.NeuralProcess(data, complete);
            FashionPreview.preview = prev;
        }


        [MenuItem("Tools/Connect", priority = 2)]
        private static void Connect()
        {
            if (connect == null)
            {
                connect = new Connect();
            }
            else
            {
                connect.Quit();
            }
            connect.Initial(5011);
            EditorApplication.update -= Update;
            EditorApplication.update += Update;
        }

        private static void Update()
        {
            var msg = connect.FetchMessage();
            if (msg != null)
            {
                NeuralData data = new NeuralData
                {
                    callback = Capture,
                    boneArgs = msg.param,
                    shape = msg.shape,
                    name = "neural_" + msg.name
                };
                NeuralInput(data, false);
                MoveDestDir("neural_*", "cache/", false);
            }
            if (!connect.Connected) EditorApplication.update -= Update;
        }


        [MenuItem("Tools/Close", priority = 2)]
        private static void Quit()
        {
            if (FashionPreview.preview != null)
            {
                ScriptableObject.DestroyImmediate(FashionPreview.preview);
            }
            if (connect != null)
            {
                connect.Quit();
            }
            EditorApplication.update -= Update;
        }

        private static void Capture(string name, RoleShape shape)
        {
            if (camera == null)
                camera = GameObject.FindObjectOfType<Camera>();
            if (rt == null)
            {
                string path = "Assets/Engine/Editor/EditorResources/CameraOuput.renderTexture";
                rt = AssetDatabase.LoadAssetAtPath<RenderTexture>(path);
            }
            rt.Release();
            camera.targetTexture = rt;
            camera.Render();
            camera.Render();
            SaveRenderTex(rt, name, shape);
            Clear();
        }


        private static void Clear()
        {
            camera.targetTexture = null;
            RenderTexture.active = null;
            rt.Release();
        }


        private static void SaveRenderTex(RenderTexture rt, string name, RoleShape shape)
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
            GameObject.DestroyImmediate(tex);
        }

    }

}