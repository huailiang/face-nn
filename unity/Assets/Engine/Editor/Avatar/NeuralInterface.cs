using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace XEngine.Editor
{
    public class NeuralData
    {
        public float[] boneArgs;
        public float[] paintArgs;
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
        public const int CNT = 95;
        public const int CNT2 = 4;
        static Connect connect;
        static FashionPreview prev;

        public static string EXPORT
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

        public static string MODEL
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

        enum Dataset { TRAINSET, TESTSET }
        RoleShape shape = RoleShape.FEMALE;
        Dataset set = Dataset.TRAINSET;
        bool complete = true, addNoise = true;
        int datacnt = 16000;
        float weight = 0.4f;

        private void OnGUI()
        {
            GUILayout.BeginVertical();
            GUILayout.Label("Generate Dataset", XEditorUtil.titleLableStyle);
            GUILayout.Space(10);
            shape = (RoleShape)EditorGUILayout.EnumPopup("Gender", shape);
            set = (Dataset)EditorGUILayout.EnumPopup("Dataset", set);
            complete = EditorGUILayout.Toggle("complete show", complete);
            addNoise = EditorGUILayout.Toggle("trainset noise", addNoise);
            weight = EditorGUILayout.Slider("data capacity", weight, 0, 1);
            GUILayout.Label("count: " + (int)(datacnt * weight));
            GUILayout.Space(8);
            if (GUILayout.Button("Generate"))
            {
                if (UnityEditor.EditorUtility.DisplayDialog("warn", "make sure enough memory's left to generate?", "ok", "cancel"))
                {
                    RandomExportModels((int)(datacnt * weight), shape, set.ToString().ToLower(), addNoise, complete);
                    EditorUtility.Open(EXPORT);
                }
            }
            GUILayout.EndVertical();
        }


        [MenuItem("Tools/SelectModel")]
        public static void Model2Image()
        {
            XEditorUtil.SetupEnv();
            string file = UnityEditor.EditorUtility.OpenFilePanel("Select model file", MODEL, "bytes");
            if (!string.IsNullOrEmpty(file))
            {
                FileInfo info = new FileInfo(file);
                ProcessFile(info, true);
                MoveDestDir("model_*", "regular/");
                EditorUtility.Open(EXPORT + "regular/");
            }
        }

        [MenuItem("Tools/SelectPicture")]
        public static void Picture2Model()
        {
            XEditorUtil.SetupEnv();
            string name = "";
            float[] args = new float[CNT];
            float[] args2 = new float[CNT2];
            if (ParseFromPicture(ref args, ref args2, ref name))
            {
                string str = "";
                int shape = int.Parse(name[name.Length - 1].ToString());
                for (int i = 0; i < CNT; i++) str += i + "-" + args[i].ToString("f3") + " ";
                Debug.Log(str);
                NeuralData data = new NeuralData
                {
                    callback = Capture,
                    boneArgs = args,
                    paintArgs = args2,
                    shape = (RoleShape)shape,
                    name = name
                };
                NeuralInput(data, true, true);
            }
        }


        [MenuItem("Tools/GenerateDatabase")]
        private static void GenerateDatabase2()
        {
            var window = EditorWindow.GetWindowWithRect<NeuralInterface>(new Rect(0, 0, 320, 200));
            window.Show();
        }

        public static bool ParseFromPicture(ref float[] args, ref float[] args2, ref string name)
        {
            string picture = UnityEditor.EditorUtility.OpenFilePanel("select picture", EXPORT, "jpg");
            int idx = picture.LastIndexOf('/') + 1;
            string descript = picture.Substring(0, idx) + "db_description";
            if (!string.IsNullOrEmpty(picture))
            {
                string key = picture.Substring(idx).Replace(".jpg", "");
                FileInfo info = new FileInfo(descript);
                FileStream fs = new FileStream(descript, FileMode.Open, FileAccess.Read);
                BinaryReader reader = new BinaryReader(fs);
                int cnt = reader.ReadInt32();
                while (cnt-- > 0)
                {
                    name = reader.ReadString();
                    for (int i = 0; i < CNT; i++) args[i] = reader.ReadSingle();
                    for (int i = 0; i < CNT2; i++) args2[i] = reader.ReadSingle();
                    if (name == key) return true;
                }
                reader.Close();
                fs.Close();
            }
            return false;
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
                string name = string.Format("db_{0:00000}_{1}", j, (int)shape);
                bw.Write(name);
                for (int i = 0; i < CNT; i++)
                {
                    args[i] = UnityEngine.Random.Range(0.0f, 1.0f);
                    bw.Write(noise ? AddNoise(args[i], i) : args[i]);
                }
                float[] args2 = new float[CNT2];
                int r = UnityEngine.Random.Range(1, 4);
                args2[r] = 1;
                args2[0] = UnityEngine.Random.Range(0.0f, 1.0f);
                for (int i = 0; i < CNT2; i++) bw.Write(args2[i]);
                NeuralData data = new NeuralData
                {
                    callback = Capture,
                    boneArgs = args,
                    paintArgs = args2,
                    shape = shape,
                    name = name
                };
                UnityEditor.EditorUtility.DisplayProgressBar(prefix, string.Format("is generating {0}/{1}", j, expc), (float)j / expc);
                NeuralInput(data, complete, true);
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
                rnd = UnityEngine.Random.Range(-40, 40);
                return (arg * 30 + 30 + rnd) / 100.0f;
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
                float[] args, args2;
                var shape = ProcessFile(info, out args, out args2);
                NeuralData data = new NeuralData
                {
                    callback = Capture,
                    boneArgs = args,
                    paintArgs = args2,
                    shape = shape,
                    name = "model_" + info.Name.Replace(".bytes", "")
                };
                NeuralInput(data, complete, true);
            }
        }

        public static RoleShape ProcessFile(FileInfo info, out float[] args, out float[] args2)
        {
            string file = info.FullName;
            FileStream fs = new FileStream(file, FileMode.Open, FileAccess.Read);
            args = new float[CNT];
            args2 = new float[CNT2];
            BinaryReader br = new BinaryReader(fs);
            RoleShape shape = (RoleShape)br.ReadInt32();
            for (int i = 0; i < CNT; i++) args[i] = br.ReadSingle();
            for (int i = 0; i < CNT2; i++) args2[i] = br.ReadSingle();
            br.Close();
            fs.Close();
            return shape;
        }


        private static void NeuralInput(NeuralData data, bool complete, bool repaint)
        {
            if (repaint)
            {
                if (prev != null) { ScriptableObject.DestroyImmediate(prev); prev = null; }
            }
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
            float[] args1 = new float[CNT];
            float[] args2 = new float[CNT2];
            Array.Copy(msg.param, args1, CNT);
            Array.Copy(msg.param, 95, args2, 0, CNT2);
            if (msg != null)
            {
                NeuralData data = new NeuralData
                {
                    callback = Capture,
                    boneArgs = args1,
                    paintArgs = args2,
                    shape = msg.shape,
                    name = "neural_" + msg.name
                };
                NeuralInput(data, false, false);
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
                RenderTexture.active = rt;
            }
            else rt.Release();

            camera.targetTexture = rt;
            camera.Render();
            camera.Render();
            SaveRenderTex(rt, name, shape);
            rt.Release();
            Clear();
        }


        private static void Clear()
        {
            camera.targetTexture = null;
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