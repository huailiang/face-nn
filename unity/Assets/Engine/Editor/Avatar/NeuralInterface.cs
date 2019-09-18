using UnityEditor;
using UnityEngine;
using System.IO;

namespace XEditor
{

    public class NeuralInterface
    {

        static RenderTexture rt;
        static Camera camera;
        static string export;


        [MenuItem("Tools/Neural")]
        public static void NeuralInput()
        {
            var win = EditorWindow.GetWindowWithRect(typeof(FashionPreviewWindow), new Rect(0, 0, 440, 640), true, "FashionPreview");
            win.Show();
            FashionPreviewWindow prev = win as FashionPreviewWindow;
            float[] ar = new float[95];
            for (int i = 0; i < ar.Length; i++) ar[i] = 0.2f;
            prev.NeuralProcess(ar, RoleShape.MALE, Capture);
        }


        private static void NeuralProcess(float[] arr, RoleShape shape)
        {
            var win = EditorWindow.GetWindowWithRect(typeof(FashionPreviewWindow), new Rect(0, 0, 440, 640), true, "FashionPreview");
            win.Show();
            Capture();
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
            if (string.IsNullOrEmpty(export))
            {
                export = Application.dataPath;
                int i = export.IndexOf("unity/Assets");
                export = export.Substring(0, i) + "export/";
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
                    if (!Directory.Exists(export))
                    {
                        Directory.CreateDirectory(export);
                    }
                    File.WriteAllBytes(export + "sample.jpg", bytes);
                    Debug.Log("save success");
                    HelperEditor.Open(export);
                }
                catch (IOException ex)
                {
                    Debug.Log("转换图片失败" + ex.Message);
                }
            }
        }

    }

}