using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;
using CFEngine;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;
using System.Collections.Generic;

/*
 *  捏脸-妆容
 */

public class FacePaint
{

    FaceData data;
    RoleShape roleShape;
    public Texture2D mainTex;
    public Texture2D tex1;
    public Vector2 offset1 = new Vector2(256, 256);
    public Vector3 rotScale1 = new Vector3(0, 1, 1);
    [Range(0, 1)]
    public float hue1 = 0.5f;
    [Range(0, 1)]
    public float saturate1 = 0.5f;
    [Range(0, 1)]
    public float heavy1 = 0.5f;

    public Texture2D tex2;
    public Vector2 offset2 = new Vector2(256, 256);
    public Vector3 rotScale2 = new Vector3(0, 1, 1);

    [Range(0, 1)]
    public float hue2 = 0.5f;
    [Range(0, 1)]
    public float saturate2 = 0.5f;
    [Range(0, 1)]
    public float heavy2 = 0.5f;

    public Texture2D tex3;
    public Vector2 offset3 = new Vector2(256, 256);
    public Vector3 rotScale3 = new Vector3(0, 1, 1);
    [Range(0, 1)]
    public float hue3 = 0.5f;
    [Range(0, 1)]
    public float saturate3 = 0.5f;
    [Range(0, 1)]
    public float heavy3 = 0.5f;

    public Texture2D tex4;
    public Vector2 offset4 = new Vector2(256, 256);
    public Vector3 rotScale4 = new Vector3(0, 1, 1);
    [Range(0, 1)]
    public float hue4 = 0.5f;
    [Range(0, 1)]
    public float saturate4 = 0.5f;
    [Range(0, 1)]
    public float heavy4 = 0.5f;

    public Texture2D tex5;
    public Vector2 offset5 = new Vector2(256, 256);
    public Vector3 rotScale5 = new Vector3(0, 1, 1);
    [Range(0, 1)]
    public float hue5 = 0.5f;
    [Range(0, 1)]
    public float saturate5 = 0.5f;
    [Range(0, 1)]
    public float heavy5 = 0.5f;
    public RenderTexture mainRt;
    public Material mat;
    public Material outputMat;

    public void Initial(GameObject go, RoleShape shape)
    {
        data = AssetDatabase.LoadAssetAtPath<FaceData>("Assets/BundleRes/Config/FaceData.asset");
        mat = AssetDatabase.LoadAssetAtPath<Material>("Assets/BundleRes/MatShader/FaceMakeup.mat");
        string child = "Player_" + shape.ToString().ToLower() + "_face";
        Transform face = go.transform.Find(child);
        var skr = face.gameObject.GetComponent<SkinnedMeshRenderer>();
        outputMat = skr.sharedMaterial;
        mainTex = outputMat.GetTexture(ShaderIDs.BaseTex) as Texture2D;
        roleShape = shape;
        CreateRT();
        Update();
        EditorSceneManager.sceneClosed += OnSceneClose;
    }

    int[] ibrows, ieyes, inoses, imouths, iears;
    string[] brows, eyes, noses, mouths, ears;
    int iBrow, iEye, iNose, iMouth, iEar;


    private void AnlyData()
    {
        if (brows == null && data != null)
        {
            Dictionary<string, int> bwdic = new Dictionary<string, int>();
            Dictionary<string, int> eydic = new Dictionary<string, int>();
            Dictionary<string, int> nsdic = new Dictionary<string, int>();
            Dictionary<string, int> mtdic = new Dictionary<string, int>();
            Dictionary<string, int> eadic = new Dictionary<string, int>();
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
                    eadic.Add(it.name, i);
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
            noses = new string[nsdic.Count];
            inoses = new int[nsdic.Count];
            nsdic.Keys.CopyTo(noses, 0);
            nsdic.Values.CopyTo(inoses, 0);
            mouths = new string[mtdic.Count];
            imouths = new int[mtdic.Count];
            mtdic.Keys.CopyTo(mouths, 0);
            mtdic.Values.CopyTo(imouths, 0);
            ears = new string[eadic.Count];
            iears = new int[eadic.Count];
            eadic.Keys.CopyTo(ears, 0);
            eadic.Values.CopyTo(iears, 0);
        }
    }

    public void OnGui()
    {
        AnlyData();
        GUILayout.BeginVertical();
        GUILayout.Space(16);
        GUILayout.Label("Face Paint");
        GuiItem("brew", brows, ref iBrow);
        GuiItem("eye", eyes, ref iEye);
        GuiItem("face", noses, ref iNose);
        GuiItem("mouth", mouths, ref iMouth);
        GuiItem("pupil", ears, ref iEar);
        GUILayout.EndVertical();
        UpdatePainTex();
    }


    private void GuiItem(string name, string[] ctx, ref int idx)
    {
        GUILayout.BeginHorizontal();
        GUILayout.Label(name);
        GUILayout.FlexibleSpace();
        idx = EditorGUILayout.Popup(idx, ctx);
        GUILayout.EndHorizontal();
    }

    private void UpdatePainTex()
    {
        tex1 = GetPaintTex(data.paintData[ibrows[iBrow]].texture, roleShape);
        tex2 = GetPaintTex(data.paintData[ieyes[iEye]].texture, roleShape);
        tex3 = GetPaintTex(data.paintData[inoses[iNose]].texture, roleShape);
        tex4 = GetPaintTex(data.paintData[imouths[iMouth]].texture, roleShape);
        tex5 = GetPaintTex(data.paintData[iears[iEar]].texture, roleShape);
    }

    private void OnSceneClose(Scene scene)
    {
        Debug.Log("close scene:" + scene.name);
        if (outputMat)
        {
            outputMat.SetTexture(ShaderIDs.BaseTex, mainTex);
        }
    }

    private void CreateRT()
    {
        if (mainRt != null)
        {
            mainRt.Release();
        }
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
        string path = "Assets/BundleRes/Knead/" + name + "_" + shape.ToString().ToLower() + ".tga";
        return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
    }


    public void Update()
    {
        Shader.SetGlobalTexture("_Part1_Tex", tex1);
        Shader.SetGlobalVector("_Part1_Offset", offset1);
        Shader.SetGlobalVector("_Part1_RotScale", rotScale1);
        Shader.SetGlobalVector("_Part1_HSB", new Vector3(hue1, saturate1, heavy1));

        Shader.SetGlobalTexture("_Part2_Tex", tex2);
        Shader.SetGlobalVector("_Part2_Offset", offset2);
        Shader.SetGlobalVector("_Part2_RotScale", rotScale2);
        Shader.SetGlobalVector("_Part2_HSB", new Vector3(hue2, saturate2, heavy2));

        Shader.SetGlobalTexture("_Part3_Tex", tex3);
        Shader.SetGlobalVector("_Part3_Offset", offset3);
        Shader.SetGlobalVector("_Part3_RotScale", rotScale3);
        Shader.SetGlobalVector("_Part3_HSB", new Vector3(hue3, saturate3, heavy3));

        Shader.SetGlobalTexture("_Part4_Tex", tex4);
        Shader.SetGlobalVector("_Part4_Offset", offset4);
        Shader.SetGlobalVector("_Part4_RotScale", rotScale4);
        Shader.SetGlobalVector("_Part4_HSB", new Vector3(hue4, saturate4, heavy4));

        Shader.SetGlobalTexture("_Part5_Tex", tex5);
        Shader.SetGlobalVector("_Part5_Offset", offset5);
        Shader.SetGlobalVector("_Part5_RotScale", rotScale5);
        Shader.SetGlobalVector("_Part5_HSB", new Vector3(hue5, saturate5, heavy5));

        Graphics.Blit(mainTex, mainRt, mat);
        if (outputMat) outputMat.SetTexture(ShaderIDs.BaseTex, mainRt);
    }

}