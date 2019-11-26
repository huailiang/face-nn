using CFUtilPoolLib;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/*
 *  捏脸-妆容
 */

namespace XEngine.Editor
{

    public class FacePaint
    {
        FaceData data;
        RoleShape roleShape;
        Texture2D mainTex;
        Texture2D tex1, tex2, tex3, tex4, tex5;
        Color color1 = Color.gray;
        Color color2 = Color.gray;
        Color color3 = Color.gray;
        Color color4 = Color.gray;
        Color color5 = Color.gray;

        Vector3 hsv1, hsv2, hsv3, hsv4, hsv5;

        Vector2 offset = new Vector2(256, 256);
        Vector3 rotScale = new Vector3(0, 1, 1);

        RenderTexture mainRt;
        static Material mat;
        Material outputMat;

        GameObject helmet;
        Camera camera;
        Vector3 cam1 = new Vector3(0, 1.0f, -10.0f);
        Vector3 cam2 = new Vector3(0, 1.72f, -8.7f);
        Vector3 cam3 = new Vector3(0, 1.72f, -8.6f);
        bool focusFace;

        public FacePaint(FaceData dt)
        {
            data = dt;
            focusFace = true;
        }

        public void Initial(GameObject go, RoleShape shape)
        {
            if (mat == null) mat = AssetDatabase.LoadAssetAtPath<Material>("Assets/Resource/RawData/FaceMakeup.mat");
            string child = "Player_" + shape.ToString().ToLower() + "_face";
            Transform face = go.transform.Find(child);
            var skr = face.gameObject.GetComponent<SkinnedMeshRenderer>();
            child = "Player_" + shape.ToString().ToLower() + "_helmet";
            helmet = go.transform.Find(child).gameObject;
            outputMat = skr.sharedMaterial;
            roleShape = shape;
            if (camera == null) camera = GameObject.FindObjectOfType<Camera>();
            if (mainTex == null) FecthMainTex();
            if (mainRt == null) CreateRT();
            Update();
            EditorSceneManager.sceneClosed -= OnSceneClose;
            EditorSceneManager.sceneClosed += OnSceneClose;
        }


        private void FecthMainTex()
        {
            var pbs = outputMat.GetTexture(ShaderIDs.PBSTex);
            string path = AssetDatabase.GetAssetPath(pbs);
            path = path.Replace("_pbs", "_b");
            mainTex = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        }

        int[] ibrows, ieyes, iface, imouths, ipupil;
        string[] brows, eyes, faces, mouths, pupils;
        int iBrow, iEye, iNose, iMouth, iEar;


        private void AnlyData()
        {
            if (brows == null && data != null)
            {
                Dictionary<string, int> bwdic = new Dictionary<string, int>();
                Dictionary<string, int> eydic = new Dictionary<string, int>();
                Dictionary<string, int> nsdic = new Dictionary<string, int>();
                Dictionary<string, int> mtdic = new Dictionary<string, int>();
                Dictionary<string, int> pupdic = new Dictionary<string, int>();
                for (int i = 0; i < data.paintData.Length; i++)
                {
                    PaintData it = data.paintData[i];
                    if (it.type == PaintSubType.BROW)
                    {
                        bwdic.Add(it.name, i);
                    }
                    else if (it.type == PaintSubType.EYE)
                    {
                        eydic.Add(it.name, i);
                    }
                    else if (it.type == PaintSubType.FACE)
                    {
                        nsdic.Add(it.name, i);
                    }
                    else if (it.type == PaintSubType.MOUTH)
                    {
                        mtdic.Add(it.name, i);
                    }
                    else if (it.type == PaintSubType.Pupil)
                    {
                        pupdic.Add(it.name, i);
                    }
                }
                brows = new string[bwdic.Count];
                ibrows = new int[bwdic.Count];
                bwdic.Keys.CopyTo(brows, 0);
                bwdic.Values.CopyTo(ibrows, 0);
                eyes = new string[eydic.Count];
                ieyes = new int[eydic.Count];
                eydic.Keys.CopyTo(eyes, 0);
                eydic.Values.CopyTo(ieyes, 0);
                faces = new string[nsdic.Count];
                iface = new int[nsdic.Count];
                nsdic.Keys.CopyTo(faces, 0);
                nsdic.Values.CopyTo(iface, 0);
                mouths = new string[mtdic.Count];
                imouths = new int[mtdic.Count];
                mtdic.Keys.CopyTo(mouths, 0);
                mtdic.Values.CopyTo(imouths, 0);
                pupils = new string[pupdic.Count];
                ipupil = new int[pupdic.Count];
                pupdic.Keys.CopyTo(pupils, 0);
                pupdic.Values.CopyTo(ipupil, 0);
            }
        }

        private int MaxIndex(float[] args, int start, int len)
        {
            int r = 0; float tmp = args[start];
            for (int i = 1; i < len; i++)
            {
                if (args[start + i] > tmp)
                {
                    tmp = args[start + i];
                    r = i;
                }
            }
            return r;
        }

        private void ParseArgs(float[] args)
        {
            iBrow = MaxIndex(args, 1, 3);
            float s = args[0] * 0.6f + 0.2f;
            color1 = Color.white * s;
        }

        public void OnGui()
        {
            AnlyData();
            GUILayout.BeginVertical();
            focusFace = GUILayout.Toggle(focusFace, " focus face");
            GuiItem("brew ", brows, ref iBrow, ref color1);
            GuiItem("eye  ", eyes, ref iEye, ref color2);
            GuiItem("face ", faces, ref iNose, ref color3);
            GuiItem("mouth", mouths, ref iMouth, ref color4);
            GuiItem("pupil", pupils, ref iEar, ref color5);
            GUILayout.EndVertical();
            UpdatePainTex();
            UpdateHsv();
            FocusFace();
        }

        private void GuiItem(string name, string[] ctx, ref int idx, ref Color color)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(name, GUILayout.Width(60));
            GUILayout.FlexibleSpace();
            idx = EditorGUILayout.Popup(idx, ctx, GUILayout.Width(140));
            GUILayout.FlexibleSpace();
            color = EditorGUILayout.ColorField(color, GUILayout.Width(75));
            GUILayout.EndHorizontal();
        }

        public void NeuralProcess(float[] args)
        {
            AnlyData();
            ParseArgs(args);
            UpdatePainTex();
            UpdateHsv();
            FocusFace();
            Update();
        }

        private void UpdatePainTex()
        {
            tex1 = GetPaintTex(data.paintData[ibrows[iBrow]].texture, roleShape);
            tex2 = GetPaintTex(data.paintData[ieyes[iEye]].texture, roleShape);
            tex3 = GetPaintTex(data.paintData[iface[iNose]].texture, roleShape);
            tex4 = GetPaintTex(data.paintData[imouths[iMouth]].texture, roleShape);
            tex5 = GetPaintTex(data.paintData[ipupil[iEar]].texture, roleShape);
        }

        private void UpdateHsv()
        {
            float h, s, v;
            Color.RGBToHSV(color1, out h, out s, out v);
            hsv1 = new Vector3(h, s, v);
            Color.RGBToHSV(color2, out h, out s, out v);
            hsv2 = new Vector3(h, s, v);
            Color.RGBToHSV(color3, out h, out s, out v);
            hsv3 = new Vector3(h, s, v);
            Color.RGBToHSV(color4, out h, out s, out v);
            hsv4 = new Vector3(h, s, v);
            Color.RGBToHSV(color5, out h, out s, out v);
            hsv5 = new Vector3(h, s, v);
        }

        private void FocusFace()
        {
            if (helmet != null && camera != null)
            {
                helmet.SetActive(!focusFace);
                var cam = roleShape == RoleShape.FEMALE ? cam3 : cam2;
                camera.transform.position = focusFace ? cam : cam1;
                camera.fieldOfView = focusFace ? 18 : 60;
            }
        }

        private void OnSceneClose(Scene scene)
        {
            if (outputMat)
            {
                outputMat.SetTexture(ShaderIDs.BaseTex, mainTex);
            }
        }

        private void CreateRT()
        {
            mainRt = new RenderTexture(mainTex.width, mainTex.height, 0, 0, RenderTextureReadWrite.Linear)
            {
                name = "_FaceTex",
                hideFlags = HideFlags.DontSave,
                filterMode = mainTex.filterMode,
                wrapMode = mainTex.wrapMode,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            mainRt.Create();
        }


        private Texture2D GetPaintTex(string name, RoleShape shape)
        {
            string path = "Assets/Resource/Knead/" + name + "_" + shape.ToString().ToLower() + ".tga";
            return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        }


        public void Update()
        {
            if (mat && outputMat)
            {
                Shader.SetGlobalTexture("_Part1_Tex", tex1);
                Shader.SetGlobalVector("_Part1_Offset", offset);
                Shader.SetGlobalVector("_Part1_RotScale", rotScale);
                Shader.SetGlobalVector("_Part1_HSB", hsv1);

                Shader.SetGlobalTexture("_Part2_Tex", tex2);
                Shader.SetGlobalVector("_Part2_Offset", offset);
                Shader.SetGlobalVector("_Part2_RotScale", rotScale);
                Shader.SetGlobalVector("_Part2_HSB", hsv2);

                Shader.SetGlobalTexture("_Part3_Tex", tex3);
                Shader.SetGlobalVector("_Part3_Offset", offset);
                Shader.SetGlobalVector("_Part3_RotScale", rotScale);
                Shader.SetGlobalVector("_Part3_HSB", hsv3);

                Shader.SetGlobalTexture("_Part4_Tex", tex4);
                Shader.SetGlobalVector("_Part4_Offset", offset);
                Shader.SetGlobalVector("_Part4_RotScale", rotScale);
                Shader.SetGlobalVector("_Part4_HSB", hsv4);

                Shader.SetGlobalTexture("_Part5_Tex", tex5);
                Shader.SetGlobalVector("_Part5_Offset", offset);
                Shader.SetGlobalVector("_Part5_RotScale", rotScale);
                Shader.SetGlobalVector("_Part5_HSB", hsv5);

                Graphics.Blit(mainTex, mainRt, mat);
                outputMat.SetTexture(ShaderIDs.BaseTex, mainRt);
            }
        }

    }
}