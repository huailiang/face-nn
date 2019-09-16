using System;
using System.Linq.Expressions;
using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;

using UnityEngineEditor = UnityEditor.Editor;

namespace XEngine.Editor
{
    [CustomEditor(typeof(RenderingEnvironment))]
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
        private SerializedProperty isStreamLoad;

        bool textureFolder = true;
        bool envLighingFolder = true;
        bool fogFolder = true;

        public void OnEnable()
        {
            enveriomentCubePath = FindProperty(x => x.EnveriomentCubePath);
            if (!string.IsNullOrEmpty(enveriomentCubePath.stringValue))
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
            isStreamLoad = FindProperty(x => x.sceneData.isStreamLoad);
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUI.BeginChangeCheck();

            RenderingEnvironment env = target as RenderingEnvironment;
            textureFolder = EditorGUILayout.Foldout(textureFolder, "Texture");
            if (textureFolder)
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
            }
            envLighingFolder = EditorGUILayout.Foldout(envLighingFolder, "EnvLighting");
            if (envLighingFolder)
            {
                EditorGUILayout.PropertyField(hdrScale);
                EditorGUILayout.PropertyField(hdrPow);
                EditorGUILayout.PropertyField(hdrAlpha);
                EditorGUILayout.PropertyField(ambientMax);
                PropertyField(lightmapShadowMask);
                PropertyField(shadowIntensity);
            }
            fogFolder = EditorGUILayout.Foldout(fogFolder, "Fog");
            if (fogFolder)
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
            }

            EditorGUILayout.Toggle("Is Stream Load", isStreamLoad.boolValue);
            if (GUILayout.Button("Sync Game Camera"))
            {
                env.SyncGameCamera();
            }
            if (EditorGUI.EndChangeCheck())
            {
                if (env != null)
                {
                    env.enabled = false;
                    env.enabled = true;
                }
            }
            serializedObject.ApplyModifiedProperties();
        }

    }
}
