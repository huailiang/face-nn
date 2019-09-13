using System;
using System.Linq.Expressions;
using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;

using UnityEngineEditor  = UnityEditor.Editor;

namespace CFEngine.Editor
{
    [CustomEditor (typeof (RenderingEnvironment))]
    public class RenderingEnvironmentEditor : BaseEditor<RenderingEnvironment>
    {
        private SerializedProperty enveriomentCubePath;
        private Cubemap envCube;

        private SerializedProperty skyboxMatPath;
        private Material skyMat;
        private SerializedProperty hdrScale;
        private SerializedProperty hdrPow;
        private SerializedProperty hdrAlpha;
        private SerializedProperty ambientMax;
        private SerializedParameter lightmapShadowMask;
        private SerializedParameter shadowIntensity;
        private SerializedProperty fogEnable;
        private SerializedProperty enableWind;
        private SerializedProperty isBigWorld;
        private SerializedProperty isStreamLoad;


        public void OnEnable ()
        {
            enveriomentCubePath = FindProperty (x => x.EnveriomentCubePath);
            if (!string.IsNullOrEmpty (enveriomentCubePath.stringValue))
            {
                string suffix = enveriomentCubePath.stringValue.EndsWith("HDR") ? ".exr" : ".tga";
                string path = string.Format("{0}/{1}{2}", AssetsConfig.GlobalAssetsConfig.ResourcePath, enveriomentCubePath.stringValue, suffix);
                envCube = AssetDatabase.LoadAssetAtPath<Cubemap>(path);
            }
            skyboxMatPath = FindProperty(x => x.SkyboxMatPath);
            if (!string.IsNullOrEmpty(skyboxMatPath.stringValue))
            {
                string path = string.Format("{0}/{1}.mat", AssetsConfig.GlobalAssetsConfig.ResourcePath, skyboxMatPath.stringValue);
                skyMat = AssetDatabase.LoadAssetAtPath<Material>(path);
            }
            hdrScale = FindProperty(x => x.hdrScale);
            hdrPow = FindProperty(x => x.hdrPow);
            hdrAlpha = FindProperty(x => x.hdrAlpha);            
            ambientMax = FindProperty(x => x.ambient.AmbientMax);
            lightmapShadowMask = FindParameter(x => x.lightmapShadowMask);
            shadowIntensity = FindParameter(x => x.shadowIntensity);
            fogEnable = FindProperty(x => x.fogEnable);
            enableWind = FindProperty(x => x.enableWind);
            isBigWorld = FindProperty(x => x.sceneData.isBigWorld);
            isStreamLoad = FindProperty(x => x.sceneData.isStreamLoad);
        }

        public override void OnInspectorGUI ()
        {
            serializedObject.Update ();
            EditorGUI.BeginChangeCheck ();

            RenderingEnvironment env = target as RenderingEnvironment;

            if(ToolsUtility.BeginFolderGroup("Texture",ref env.textureFolder))
            {
                Cubemap cube = EditorGUILayout.ObjectField(envCube, typeof(Cubemap), false) as Cubemap;
                if (cube != envCube)
                {
                    envCube = cube;
                    if (envCube == null)
                    {
                        env.EnveriomentCubePath = "";
                    }
                    else
                    {
                        string path = AssetDatabase.GetAssetPath(envCube);
                        env.EnveriomentCubePath = path.Substring(AssetsConfig.GlobalAssetsConfig.ResourcePath.Length + 1);
                        string suffix = env.EnveriomentCubePath.EndsWith(".tga") ? ".tga" : ".exr";
                        env.EnveriomentCubePath = env.EnveriomentCubePath.Replace(suffix, "");
                        env.LoadRes(true, false);
                    }
                }
                Material sky = EditorGUILayout.ObjectField(skyMat, typeof(Material), false) as Material;
                if (sky != skyMat)
                {
                    skyMat = sky;
                    if (skyMat == null)
                    {
                        env.SkyboxMatPath = "";
                    }
                    else
                    {
                        string path = AssetDatabase.GetAssetPath(skyMat);
                        env.SkyboxMatPath = path.Substring(AssetsConfig.GlobalAssetsConfig.ResourcePath.Length + 1);
                        env.SkyboxMatPath = env.SkyboxMatPath.Replace(".mat", "");
                        env.LoadRes(false, true);
                    }
                }
                ToolsUtility.EndFolderGroup();
            }

            if (ToolsUtility.BeginFolderGroup("EnvLighting", ref env.envLighingFolder))
            {
                EditorGUILayout.PropertyField(hdrScale);
                EditorGUILayout.PropertyField(hdrPow);
                EditorGUILayout.PropertyField(hdrAlpha);
                EditorGUILayout.PropertyField(ambientMax);
                PropertyField(lightmapShadowMask);
                PropertyField(shadowIntensity);

                ToolsUtility.EndFolderGroup();
            }

            // EditorGUILayout.LabelField ("CameraPointLight", EditorStyles.boldLabel);
            // EditorGUILayout.PropertyField (cameraLightAtten);
            // EditorGUILayout.PropertyField (cameraLightSqDistance);

            if (ToolsUtility.BeginFolderGroup("Fog", ref env.fogFolder))
            {
                EditorGUILayout.PropertyField(fogEnable);
                
                if (fogEnable.boolValue)
                {
                    FogModify fogParam = env.fog;
                    EditorGUI.BeginChangeCheck();
                    var fogDensity = EditorGUILayout.Slider("Fog Density", fogParam.Density, 0.0f, 1f);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "Fog Density");
                        fogParam.Density = fogDensity;
                    }

                    EditorGUI.BeginChangeCheck();
                    var FogMoveSpeed = EditorGUILayout.Slider("Fog Skybox Height", fogParam.SkyboxHeight, 0.0f, 10.0f);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "Fog Scale");
                        fogParam.SkyboxHeight = FogMoveSpeed;
                    }

                    EditorGUI.BeginChangeCheck();
                    var fogEndHeight = EditorGUILayout.Slider("Fog End Height", fogParam.EndHeight, 0.0f, 1000.0f);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "Fog End Height");
                        fogParam.EndHeight = fogEndHeight;
                    }

                    EditorGUI.BeginChangeCheck();
                    var fogStartDistance = EditorGUILayout.Slider("Fog StartDistance", fogParam.StartDistance, 0.0f, 200.0f);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "Fog StartDistance");
                        fogParam.StartDistance = fogStartDistance;
                    }

                    EditorGUI.BeginChangeCheck();
                    var fogColor0 = EditorGUILayout.ColorField("Fog Color 0", fogParam.Color0);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "Fog Color 0");
                        fogParam.Color0 = fogColor0;
                    }
                    EditorGUI.BeginChangeCheck();
                    var fogColor1 = EditorGUILayout.ColorField("Fog Color 1", fogParam.Color1);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "Fog Color 1");
                        fogParam.Color1 = fogColor1;
                    }
                    EditorGUI.BeginChangeCheck();
                    var fogColor2 = EditorGUILayout.ColorField("Fog Color 2", fogParam.Color2);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "Fog Color 2");
                        fogParam.Color2 = fogColor2;
                    }
                }
                ToolsUtility.EndFolderGroup();
            }

            env.windFolder = EditorGUILayout.Foldout(env.windFolder, "Wind");
            if (env.windFolder)
            {
                if (enableWind.boolValue)
                {
                    ToolsUtility.BeginGroup("Random Wind");
                    RandomWindModify wind = env.randomWind;
                    wind.WindPos = EditorGUILayout.Vector3Field("Pos", wind.WindPos);
                    wind.WindStrengthCurve = EditorGUILayout.CurveField("Strength", wind.WindStrengthCurve);
                    wind.WindStrengthUpdateTime = EditorGUILayout.FloatField("StrengthUpdateTime", wind.WindStrengthUpdateTime);
                    wind.WindDirectionCurve = EditorGUILayout.CurveField("Direction", wind.WindDirectionCurve);
                    wind.WindDirectionUpdateTime = EditorGUILayout.FloatField("DirectionUpdateTime", wind.WindDirectionUpdateTime);
                    wind.WindStrengthRange = EditorGUILayout.Vector2Field("StrengthRange", wind.WindStrengthRange);
                    wind.RotateAxis = EditorGUILayout.Vector3Field("RotateAxis", wind.RotateAxis);
                    wind.StartAngle = EditorGUILayout.FloatField("StartAngle", wind.StartAngle);
                    wind.EndAngle = EditorGUILayout.FloatField("EndAngle", wind.EndAngle);
                    wind.NeedWindStrengthReserve = EditorGUILayout.Toggle("StrengthReserve", wind.NeedWindStrengthReserve);
                    wind.NeedWindDirectionReserve = EditorGUILayout.Toggle("DirectionReserve", wind.NeedWindDirectionReserve);                    
                    ToolsUtility.EndFolderGroup();

                    ToolsUtility.BeginGroup("Wave Wind");
                    WaveWindModify waveWind = env.waveWind;

                    EditorGUI.BeginChangeCheck();
                    float fastWindTime = EditorGUILayout.FloatField("Fast Wind Time", waveWind.fastWindTime);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "Fast Wind Time");
                        waveWind.fastWindTime = fastWindTime;
                    }


                    EditorGUI.BeginChangeCheck();
                    float slowWindScale = EditorGUILayout.FloatField("Slow Wind Scale", waveWind.slowWindScale);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "Slow Wind Scale");
                        waveWind.slowWindScale = slowWindScale;
                    }

                    EditorGUI.BeginChangeCheck();
                    float strength = EditorGUILayout.FloatField("Strength", waveWind.strength);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "Wind Strength");
                        waveWind.strength = strength;
                    }
                    EditorGUI.BeginChangeCheck();
                    float strengthScale = EditorGUILayout.FloatField("StrengthScale", waveWind.strengthScale);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "Wind Strength Scale");
                        waveWind.strengthScale = strengthScale;
                    }


                    // EditorGUI.BeginChangeCheck();
                    // waveWind.maxYFalloff = EditorGUILayout.FloatField("Max Y Falloff", waveWind.maxYFalloff);
                    // if (EditorGUI.EndChangeCheck())
                    // {
                    //     Undo.RecordObject(target, "Max Y Falloff");
                    // }
                    // EditorGUI.BeginChangeCheck();
                    // waveWind.windRange = EditorGUILayout.FloatField("Wind Range", waveWind.windRange);
                    // if (EditorGUI.EndChangeCheck())
                    // {
                    //     Undo.RecordObject(target, "Wind Range");
                    //     waveWind.windChange.y = waveWind.windRange;
                    // }
                    // EditorGUI.BeginChangeCheck();
                    // waveWind.windRangeScale = EditorGUILayout.FloatField("Wind Range Scale", waveWind.windRangeScale);
                    // if (EditorGUI.EndChangeCheck())
                    // {
                    //     Undo.RecordObject(target, "Wind Range Scale");
                    // }

                    // EditorGUI.BeginChangeCheck();
                    // waveWind.windSpeed = EditorGUILayout.FloatField("Wind Speed", waveWind.windSpeed);
                    // if (EditorGUI.EndChangeCheck())
                    // {
                    //     Undo.RecordObject(target, "Wind Speed");
                    // }

                    // EditorGUI.BeginChangeCheck();
                    // waveWind.windChange.x = EditorGUILayout.FloatField("Random Change Time", waveWind.windChange.x);
                    // if (EditorGUI.EndChangeCheck())
                    // {
                    //     Undo.RecordObject(target, "Random Change Time");
                    // }
                    // EditorGUI.BeginChangeCheck();
                    // waveWind.windChange.z = EditorGUILayout.FloatField("Random Speed Change", waveWind.windChange.z);
                    // if (EditorGUI.EndChangeCheck())
                    // {
                    //     Undo.RecordObject(target, "Random Speed Change");
                    // }

                    // EditorGUI.BeginChangeCheck();
                    // waveWind.windChange.w = EditorGUILayout.FloatField("Random Range Change", waveWind.windChange.w);
                    // if (EditorGUI.EndChangeCheck())
                    // {
                    //     Undo.RecordObject(target, "Random Range Change");
                    // }

                    // EditorGUI.BeginChangeCheck();
                    // waveWind.windBlockRange = EditorGUILayout.FloatField("Wind Width", waveWind.windBlockRange);
                    // if (EditorGUI.EndChangeCheck())
                    // {
                    //     Undo.RecordObject(target, "Wind Width");
                    // }

                    // EditorGUI.BeginChangeCheck();
                    // waveWind.windBlockOffset = EditorGUILayout.Vector2Field("Block Offset", waveWind.windBlockOffset);
                    // if (EditorGUI.EndChangeCheck())
                    // {
                    //     Undo.RecordObject(target, "Block Offset");
                    // }
                }
                ToolsUtility.EndFolderGroup();
            }

            EditorGUILayout.Toggle("Is Big World", isBigWorld.boolValue);
            EditorGUILayout.Toggle("Is Stream Load", isStreamLoad.boolValue);
            if (GUILayout.Button("Sync Game Camera"))
            {
                env.SyncGameCamera();
            }
            if (env != null)
            {


                // sceneData.enableShadow = EditorGUILayout.Toggle ("Enable Shadow", sceneData.enableShadow);
                // sceneData.enableShadowCsm = EditorGUILayout.Toggle ("Enable CSM Shadow", sceneData.enableShadowCsm);

                // sceneData.shadowParam.x = EditorGUILayout.Slider ("Shadow Center Offset", sceneData.shadowParam.x, 0, 1);
                // sceneData.shadowParam.y = EditorGUILayout.Slider ("Shadow CSM0 Size", sceneData.shadowParam.y, 0.1f, 10);
                // sceneData.shadowParam.z = EditorGUILayout.Slider ("Shadow CSM1Size", sceneData.shadowParam.z, 0.1f, 30);

                // shadowDepthBias.floatValue = EditorGUILayout.Slider ("Shadow Depth Bias", shadowDepthBias.floatValue, -1, 1);
                // shadowNormalBias.floatValue = EditorGUILayout.Slider ("Shadow Normal Bias", shadowNormalBias.floatValue, -1, 4);

                // shadowSmoothMin.floatValue = EditorGUILayout.Slider ("Shadow Smooth Min", shadowSmoothMin.floatValue, 0, 100);
                // shadowSmoothMax.floatValue = EditorGUILayout.Slider ("Shadow Smooth Max", shadowSmoothMax.floatValue, 0, 100);
                // shadowSampleSize.floatValue = EditorGUILayout.Slider ("Shadow Sample Size", shadowSampleSize.floatValue, 0, 5);
                // shadowPower.floatValue = EditorGUILayout.Slider ("Shadow Smooth Power", shadowPower.floatValue, 0, 5);



            }

            if (EditorGUI.EndChangeCheck())
            {
                if (env != null)
                {
                    env.enabled = false;
                    env.enabled = true;
                }
            }
            serializedObject.ApplyModifiedProperties ();
        }

        // void OnSceneGUI ()
        // {
        //     if (SceneView.lastActiveSceneView != null && SceneView.lastActiveSceneView.camera != null)
        //     {
        //         Transform t = SceneView.lastActiveSceneView.camera.transform;
        //         Vector3 pos = t.position + t.forward * 10;
        //         RenderingEnvironment env = target as RenderingEnvironment;

        //         if (windParam != null && enableWind.boolValue)
        //         {
        //             EditorGUI.BeginChangeCheck ();

        //             Vector3 windpos = Handles.PositionHandle (windParam.WindPos, Quaternion.identity);
        //             if (EditorGUI.EndChangeCheck ())
        //             {
        //                 Undo.RecordObject (this, "Wind Pos");
        //                 windParam.WindPos = windpos;
        //             }

        //             pos = t.position + t.forward * 10;
        //             if (env != null)
        //             {
        //                 EditorGUI.BeginChangeCheck ();
        //                 Quaternion rot = Handles.RotationHandle (Quaternion.LookRotation (waveWind.windPlane), pos);
        //                 if (EditorGUI.EndChangeCheck ())
        //                 {
        //                     Vector3 normal = rot * Vector3.forward;
        //                     Vector2 mormalXZ = new Vector2 (normal.x, normal.z);
        //                     mormalXZ.Normalize ();
        //                     waveWind.windPlane.x = mormalXZ.x;
        //                     waveWind.windPlane.z = mormalXZ.y;
        //                 }
        //                 Handles.ArrowHandleCap (100, pos, rot, 1, EventType.Repaint);
        //                 Handles.Label (pos, "Wind Dir");
        //             }
        //         }
        //     }
        // }
    }
}
