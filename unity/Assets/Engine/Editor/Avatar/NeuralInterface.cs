using UnityEditor;
using UnityEngine;
using System.IO;
using UnityEngine.UI;

public class NeuralInterface
{

    static RenderTexture rt;
    static Camera camera;
    static string export;

    public static void NeuralInput(float[] arr)
    {

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
            }
            catch (IOException ex)
            {
                Debug.Log("转换图片失败" + ex.Message);
            }
        }
    }

}