using System;
using System.Collections.Generic;
using XEngine;
using CFUtilPoolLib;
using UnityEngine;
using UnityEngine.Rendering;


public struct EffectLerpContext
{
    public ParameterOverride srcParam;
    public ParameterOverride runtimeParam;

    public bool IsValid()
    {
        return srcParam != null && runtimeParam != null;
    }

    public void Reset()
    {
        if (IsValid())
        {
            runtimeParam.SetOverride(false);
        }
    }

    public void LerpTo(float value, float percent)
    {
        if (IsValid())
        {
            runtimeParam.SetOverride(true);
            runtimeParam.Interp(srcParam, value, percent);
        }
    }

    public void LerpRecover(float value, float percent)
    {
        if (IsValid())
        {
            runtimeParam.SetOverride(true);
            runtimeParam.Interp(value, srcParam, percent);
        }
    }

    public void LerpTo(Vector4 value, float percent)
    {
        if (IsValid())
        {
            runtimeParam.SetOverride(true);
            runtimeParam.Interp(srcParam, value, percent);
        }
    }

    public void LerpRecover(Vector4 value, float percent)
    {
        if (IsValid())
        {
            runtimeParam.SetOverride(true);
            runtimeParam.Interp(value, srcParam, percent);
        }
    }

    public void LerpTo(Enum value, float percent)
    {
        if (IsValid())
        {
            runtimeParam.SetOverride(true);
            runtimeParam.Interp(srcParam, value, percent);
        }
    }

    public void LerpRecover(Enum value, float percent)
    {
        if (IsValid())
        {
            runtimeParam.SetOverride(true);
            runtimeParam.Interp(value, srcParam, percent);
        }
    }
}
public sealed class RenderingManager : IRenderManager
{
    delegate RenderTexture GetCreateRT();
    static RenderingManager s_Instance;

    public static RenderingManager instance
    {
        get
        {
            if (s_Instance == null)
                s_Instance = new RenderingManager();

            return s_Instance;
        }
    }
    public bool Deprecated { get; set; }
    public RenderingEnvironment renderingEnvironment;
    public bool settingsUpdateNeeded = true;
    public bool alwaysEnablePostproecss = false;
    private List<ParameterOverride> m_CachedRuntimeParam;
    private List<RenderBatch> afterAlphaBatchesRef = null;
    public Vector4 sunForward = new Vector4(0, -1, 0, 0);
    
    public float rtScale = 1;
    public int width = 1136;
    public int height = 640;
    public RenderTextureFormat format;
    public RenderTexture sceneRT0; //same size & format as frame buffer 2.8mb
    public RenderTexture sceneRT1; //same size & format as frame buffer 2.8mb
    private RenderTexture halfSceneRT; //half size & format as frame buffer 0.7mb
    private RenderTexture halfDepthRT; //half size & format as frame buffer 0.7mb
    private RenderTexture halfSceneFP16RT; //half size & fp16 format as frame buffer 2.8mb
    public RenderTexture depthRT; //same size & format as frame buffer  1.4mb
    public RenderTexture halfQuarterDepthRT; //1/8 0.04375mb
    private RenderTexture quarterRT0; // 1/4 0.175mb
    private RenderTexture quarterRT1; //     0.175mb
    private RenderTexture halfQuarterRT0; // 1/8 0.04375mb
    private RenderTexture halfQuarterRT1; // 1/8 0.04375mb
    private RenderTexture hexRWRT0; // 1/16 
    private RenderTexture hexRWRT1; // 1/16
    private RenderTexture shadowRT;
    private RenderTexture shadowMapProjectRT;

    private RenderTexture halfSceneRWRT0; //for dof
    private RenderTexture halfSceneRWRT1; //for dof
    private RenderTexture halfSceneRWRT2; //for dof
#if UNITY_EDITOR
    private RenderTexture debugRT;
#endif
    private int rtSize = 0;
    //total 16.49375
    //2 shadow rt 6mb+4mb
    public uint postFlag;

    private readonly GetCreateRT[] createRT = new GetCreateRT[(int)ERTType.Num];
    private readonly LoadEnv[] loadEnv = new LoadEnv[(int)EnverimentModifyType.Num];
#if UNITY_EDITOR
    public static SaveEnv[] saveEnv = new SaveEnv[(int) EnverimentModifyType.Num];
#endif


    RenderingManager()
    {
#if UNITY_EDITOR
        ReloadBaseTypes ();
#endif
        createRT[(int)ERTType.ESceneRT0] = GetSceneRT0;
        createRT[(int)ERTType.ESceneRT1] = GetSceneRT1;
        createRT[(int)ERTType.EHalfSceneRT] = GetHalfRT;
        createRT[(int)ERTType.EHalfDepthRT] = GetHalfDepthRT;
        createRT[(int)ERTType.EHalfSceneFP16RT] = GetHalfSceneFP16RT;
        createRT[(int)ERTType.EDepthRT] = GetDepthRT;
        createRT[(int)ERTType.EQquarterRT0] = GetQuarterRT0;
        createRT[(int)ERTType.EQquarterRT1] = GetQuarterRT1;
        createRT[(int)ERTType.EHalfQuarterRT0] = GetHalfQuarterRT0;
        createRT[(int)ERTType.EShadowRT] = GetShadowRT;
        createRT[(int)ERTType.EShadowMapProjectRT] = GetShadowProjectRT;

        loadEnv[(int)EnverimentModifyType.Lighting] = LightingModify.Load;
        loadEnv[(int)EnverimentModifyType.Ambient] = AmbientModify.Load;
        loadEnv[(int)EnverimentModifyType.Fog] = FogModify.Load;
        loadEnv[(int)EnverimentModifyType.PPBloom] = BloomModify.Load;
        loadEnv[(int)EnverimentModifyType.PPLut] = LutModify.Load;
        loadEnv[(int)EnverimentModifyType.Effect] = EffectModify.Load;

#if UNITY_EDITOR
        saveEnv[(int) EnverimentModifyType.Lighting] = LightingModify.Save;
        saveEnv[(int) EnverimentModifyType.Ambient] = AmbientModify.Save;
        saveEnv[(int) EnverimentModifyType.Fog] = FogModify.Save;
        saveEnv[(int) EnverimentModifyType.PPBloom] = BloomModify.Save;
        saveEnv[(int) EnverimentModifyType.Effect] = EnverinmentContext.SaveEffect;
#endif
    }

#if UNITY_EDITOR
    // Called every time Unity recompile scripts in the editor. We need this to keep track of
    // any new custom effect the user might add to the project
    [UnityEditor.Callbacks.DidReloadScripts]
    static void OnEditorReload ()
    {
        instance.ReloadBaseTypes ();
    }
    void CleanBaseTypes ()
    {
    }
    // This will be called only once at runtime and everytime script reload kicks-in in the
    // editor as we need to keep track of any compatible post-processing effects in the project
    void ReloadBaseTypes ()
    {
        CleanBaseTypes ();
    }
#endif


    //runtime change volumn
    public void Release()
    {
        if (renderingEnvironment != null)
        {
            renderingEnvironment.Uninit();
        }
        afterAlphaBatchesRef = null;
    }
    public void SetEnable(bool enable)
    {
    }

    public void AlwaysEnablePostproecss(bool enable)
    {
        alwaysEnablePostproecss = enable;
    }


    public void EnableEffect(int effectType, bool enable)
    {
    }

    public void SetEffectParam(int effectType, float param0, float param1, float param2, float param3)
    {
    }

    public IParameterOverride GetEffectParamOverride(int effectType, int paramIndex)
    {
        return null;
    }
    public void GetEffectParamLerp(ref EffectLerpContext context, int effectType, int paramIndex)
    {
    }
    public void DirtyEffect(int effectType)
    {
    }

    public void InitRender(UnityEngine.Camera camera)
    {
        if (renderingEnvironment != null)
        {
            renderingEnvironment.InitRender(camera);
        }
    }

    public void UpdateRender()
    {
        if (renderingEnvironment != null)
        {
            renderingEnvironment.ManualUpdate();
        }
    }

    public bool HasPostAlphaCommand()
    {
        return afterAlphaBatchesRef != null && afterAlphaBatchesRef.Count > 0;
    }

    public void SetPostAlphaCommand(List<RenderBatch> batches)
    {
        afterAlphaBatchesRef = batches;
    }


    public void CreateSceneRT(bool needDepth)
    {
        GetSceneRT0();
        GetSceneRT1();
        if (needDepth)
        {
            GetDepthRT();
        }
    }

    #region createRTs
    public RenderTexture GetSceneRT0()
    {
        if (sceneRT0 == null)
        {
            sceneRT0 = new RenderTexture(width, height, 0, format, RenderTextureReadWrite.Linear)
            {
                name = "_SceneRT0",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            sceneRT0.Create();
            rtSize += width * height * 4;
        }
        return sceneRT0;
    }
    public RenderTexture GetSceneRT1()
    {
        if (sceneRT1 == null)
        {
            sceneRT1 = new RenderTexture(width, height, 0, format, RenderTextureReadWrite.Linear)
            {
                name = "_SceneRT1",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            sceneRT1.Create();
            rtSize += width * height * 4;
        }
        return sceneRT1;
    }
    public RenderTexture GetDepthRT()
    {
        if (depthRT == null)
        {
            depthRT = new RenderTexture(width, height, 24, RenderTextureFormat.Depth, RenderTextureReadWrite.Linear)
            {
                name = "_DepthRT",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = true,
                useMipMap = false
            };
            depthRT.Create();
            Shader.SetGlobalTexture(ShaderIDs.CameraDepthTex, depthRT);
            rtSize += width * height * 3;
        }
        return depthRT;
    }

    public RenderTexture GetHalfRT()
    {
        if (halfSceneRT == null)
        {
            halfSceneRT = new RenderTexture(width / 2, height / 2, 0, format, RenderTextureReadWrite.Linear)
            {
                name = "_HalfSceneRT",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            halfSceneRT.Create();
            rtSize += width * height;
        }
        return halfSceneRT;
    }

    public RenderTexture GetHalfDepthRT()
    {
        if (halfDepthRT == null)
        {
            halfDepthRT = new RenderTexture(width / 2, height / 2, 16, RenderTextureFormat.Depth, RenderTextureReadWrite.Linear)
            {
                name = "_HalfDepthRT",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            halfDepthRT.Create();
            rtSize += width * height / 2;
        }
        return halfDepthRT;
    }

    public RenderTexture GetHalfQuarterDepthRT()
    {
        if (halfQuarterDepthRT == null)
        {
            halfQuarterDepthRT = new RenderTexture(width / 8, height / 8, 0, RenderTextureFormat.R8, RenderTextureReadWrite.Linear)
            {
                name = "_HalfQuarterDepthRT",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            halfQuarterDepthRT.Create();
            rtSize += width * height / 32;

        }
        return halfQuarterDepthRT;
    }

    public RenderTexture GetHalfSceneFP16RT()
    {
        if (halfSceneFP16RT == null)
        {
            RenderTextureFormat f = RenderTextureFormat.ARGB32;
            int scale = 1;
            if (RenderTextureFormat.ARGBHalf.IsSupported())
            {
                f = RenderTextureFormat.ARGBHalf;
                scale = 2;
            }

            halfSceneFP16RT = new RenderTexture(width / 2, height / 2, 0, f, RenderTextureReadWrite.Linear)
            {
                name = "_HalfSceneFP16RT",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            halfSceneFP16RT.Create();
            rtSize += width * height * scale;
        }
        return halfSceneFP16RT;
    }

    public RenderTexture GetQuarterRT0()
    {
        if (quarterRT0 == null)
        {
            quarterRT0 = new RenderTexture(width / 4, height / 4, 0, format, RenderTextureReadWrite.Linear)
            {
                name = "_QuarterRT0",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            quarterRT0.Create();
            rtSize += width * height / 4;
        }
        return quarterRT0;
    }
    public RenderTexture GetQuarterRT1()
    {
        if (quarterRT1 == null)
        {
            quarterRT1 = new RenderTexture(width / 4, height / 4, 0, format, RenderTextureReadWrite.Linear)
            {
                name = "_QuarterRT1",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            quarterRT1.Create();
            rtSize += width * height / 4;
        }
        return quarterRT1;
    }

    public RenderTexture GetHalfQuarterRT0()
    {
        if (halfQuarterRT0 == null)
        {
            halfQuarterRT0 = new RenderTexture(width / 8, height / 8, 0, format, RenderTextureReadWrite.Linear)
            {
                name = "_HalfQuarterRT0",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            halfQuarterRT0.Create();
            rtSize += width * height / 16;
        }
        return halfQuarterRT0;
    }

    public RenderTexture GetHalfQuarterRT1()
    {
        if (halfQuarterRT1 == null)
        {
            halfQuarterRT1 = new RenderTexture(width / 8, height / 8, 0, format, RenderTextureReadWrite.Linear)
            {
                name = "_HalfQuarterRT1",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            halfQuarterRT1.Create();
            rtSize += width * height / 16;
        }
        return halfQuarterRT1;
    }

    public RenderTexture GetShadowRT()
    {
        if (shadowRT == null)
        {
            RenderTextureFormat f = RenderTextureFormat.RGHalf;
            if (!SystemInfo.SupportsRenderTextureFormat(format))
            {
                f = RenderTextureFormat.RGB111110Float;
            }

            shadowRT = new RenderTexture(1024, 1024, 16, f, RenderTextureReadWrite.Linear)
            {
                name = "Shadowmap",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            shadowRT.Create();
            rtSize += 1024 * 1024 * 8;
        }

        return shadowRT;
    }

    public RenderTexture GetShadowProjectRT()
    {
        if (shadowMapProjectRT == null)
        {
            shadowMapProjectRT = new RenderTexture(1024, 1024, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
            {
                name = "ShadowmapProject",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false
            };
            shadowMapProjectRT.Create();
            rtSize += 1024 * 1024 * 4;
        }

        return shadowMapProjectRT;
    }
    public RenderTexture GetHalfSceneRWRT0(bool rw = true)
    {
        if (halfSceneRWRT0 == null)
        {
            RenderTextureFormat f = RenderTextureFormat.ARGB32;
            if (RenderTextureFormat.ARGBHalf.IsSupported())
            {
                f = RenderTextureFormat.ARGBHalf;
            }
            halfSceneRWRT0 = new RenderTexture(width / 2, height / 2, 0, f, RenderTextureReadWrite.Linear)
            {
                name = "_HalfSceneRWRT0",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false,
                enableRandomWrite = rw,
            };
            halfSceneRWRT0.Create();
            rtSize += width * height;
        }
        return halfSceneRWRT0;
    }

    public RenderTexture GetHalfSceneRWRT1()
    {
        if (halfSceneRWRT1 == null)
        {
            halfSceneRWRT1 = new RenderTexture(width / 2, height / 2, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
            {
                name = "_HalfSceneRWRT1",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false,
                enableRandomWrite = true,
            };
            halfSceneRWRT1.Create();
            rtSize += width * height;
        }
        return halfSceneRWRT1;
    }

    public RenderTexture GetHalfSceneRWRT2()
    {
        if (halfSceneRWRT2 == null)
        {
            halfSceneRWRT2 = new RenderTexture(width / 2, height / 2, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
            {
                name = "_HalfSceneRWRT2",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false,
                enableRandomWrite = true,
            };
            halfSceneRWRT2.Create();
            rtSize += width * height;
        }
        return halfSceneRWRT2;
    }
    public RenderTexture GetHexRWRT0()
    {
        if (hexRWRT0 == null)
        {
            hexRWRT0 = new RenderTexture(width / 16, height / 16, 0, format, RenderTextureReadWrite.Linear)
            {
                name = "_160",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false,
                enableRandomWrite = true,
            };
            hexRWRT0.Create();
            rtSize += width * height / 64;
        }
        return hexRWRT0;
    }
    public RenderTexture GetHexRWRT1()
    {
        if (hexRWRT1 == null)
        {
            hexRWRT1 = new RenderTexture(width / 16, height / 16, 0, format, RenderTextureReadWrite.Linear)
            {
                name = "_161",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false,
                enableRandomWrite = true,
            };
            hexRWRT1.Create();
            rtSize += width * height / 64;
        }
        return hexRWRT1;
    }

#if UNITY_EDITOR
    public void RenderToDebugRt(RenderTexture src, CommandBuffer cmd, Shader copy)
    {
        if (debugRT == null)
        {
            debugRT = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear)
            {
                name = "_DebugRT",
                hideFlags = HideFlags.DontSave,
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                autoGenerateMips = false,
                useMipMap = false,
            };
            debugRT.Create();
        }
        cmd.BlitFullscreenTriangle(src, debugRT, copy, true);
    }
#endif
    public void ReleaseSceneRT()
    {
        if (sceneRT0 != null)
        {
            RuntimeUtilities.Destroy(sceneRT0);
            sceneRT0 = null;
        }
        if (sceneRT1 != null)
        {
            RuntimeUtilities.Destroy(sceneRT1);
            sceneRT1 = null;
        }
        if (halfSceneRT != null)
        {
            RuntimeUtilities.Destroy(halfSceneRT);
            halfSceneRT = null;
        }
        if (halfSceneFP16RT != null)
        {
            RuntimeUtilities.Destroy(halfSceneFP16RT);
            halfSceneFP16RT = null;
        }
        if (depthRT != null)
        {
            RuntimeUtilities.Destroy(depthRT);
            depthRT = null;
        }
        if (quarterRT0 != null)
        {
            RuntimeUtilities.Destroy(quarterRT0);
            quarterRT0 = null;
        }

        if (quarterRT1 != null)
        {
            RuntimeUtilities.Destroy(quarterRT1);
            quarterRT1 = null;
        }
        if (halfQuarterRT0 != null)
        {
            RuntimeUtilities.Destroy(halfQuarterRT0);
            halfQuarterRT0 = null;
        }
        if (hexRWRT0 != null)
        {
            RuntimeUtilities.Destroy(hexRWRT0);
            hexRWRT0 = null;
        }
        if (hexRWRT1 != null)
        {
            RuntimeUtilities.Destroy(hexRWRT1);
            hexRWRT1 = null;
        }
        if (shadowRT != null)
        {
            RuntimeUtilities.Destroy(shadowRT);
            shadowRT = null;
        }
        if (shadowMapProjectRT != null)
        {
            RuntimeUtilities.Destroy(shadowMapProjectRT);
            shadowMapProjectRT = null;
        }
        if (halfSceneRWRT0 != null)
        {
            RuntimeUtilities.Destroy(halfSceneRWRT0);
            halfSceneRWRT0 = null;
        }
        if (halfSceneRWRT1 != null)
        {
            RuntimeUtilities.Destroy(halfSceneRWRT1);
            halfSceneRWRT1 = null;
        }
        if (halfSceneRWRT2 != null)
        {
            RuntimeUtilities.Destroy(halfSceneRWRT2);
            halfSceneRWRT2 = null;
        }
        rtSize = 0;
#if UNITY_EDITOR
        if (debugRT != null)
        {
            RuntimeUtilities.Destroy(debugRT);
            debugRT = null;
        }
#endif
    }
    public RenderTexture GetRT(ERTType rtType)
    {
        return createRT[(int)rtType]();
    }

    public void SetEffectFlag(uint flag)
    {
        postFlag = flag;
    }
    #endregion
    #region envData

    public void LoadEnvData(XBinaryReader reader, ref ListObjectWrapper<ISceneObject> listObject, ref int envObjStart, ref int envObjEnd, byte dynamicSceneId)
    {
    }

    #endregion

}
