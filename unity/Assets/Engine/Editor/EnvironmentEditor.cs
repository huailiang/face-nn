using UnityEditor;
using UnityEngine;

namespace XEngine.Editor
{
    [CustomEditor(typeof(Environment))]
    public class EnvironmentEditor : BaseEditor<Environment>
    {
        private SerializedProperty enveriomentCubePath;
        private Cubemap envCube;

        private SerializedProperty skyboxMatPath;
        private Material skyMat;
        private SerializedProperty hdrScale;
        private SerializedProperty hdrPow;
        private SerializedProperty hdrAlpha;
        private SerializedParameter lightmapShadowMask;
        private SerializedParameter shadowIntensity;
        private SerializedProperty fogEnable;
        //lighting
        private SerializedProperty roleLight0;
        private SerializedProperty roleLight1;
        //debugShadow
        private SerializedProperty lookTarget;
        private SerializedProperty shadowMapLevel;
        private SerializedProperty shadowBound;
        private SerializedProperty drawShadowLighing;

        private SerializedProperty drawType;
        private SerializedProperty showObjects;

        private SerializedProperty debugMode;
        private SerializedProperty debugDisplayType;
        private SerializedParameter splitAngle;
        private SerializedProperty splitPos;
        private float splitLeft = -1;
        private float splitRight = 1;

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
            lightmapShadowMask = FindParameter(x => x.lightmapShadowMask);
            shadowIntensity = FindParameter(x => x.shadowIntensity);
            fogEnable = FindProperty(x => x.fogEnable);

            roleLight0 = FindProperty(x => x.roleLight0);
            roleLight1 = FindProperty(x => x.roleLight1);

            shadowMapLevel = FindProperty(x => x.shadowMapLevel);
            shadowBound = FindProperty(x => x.shadowBound);
            lookTarget = FindProperty(x => x.lookTarget);
            drawShadowLighing = FindProperty(x => x.drawShadowLighing);

            drawType = FindProperty(x => x.drawType);
            showObjects = FindProperty(x => x.showObjects);

            debugMode = FindProperty(x => x.debugContext.debugMode);
            debugDisplayType = FindProperty(x => x.debugContext.debugDisplayType);
            splitAngle = FindParameter(x => x.debugContext.splitAngle);
            splitPos = FindProperty(x => x.debugContext.splitPos);

            AssetsConfig.RefreshShaderDebugNames();
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUI.BeginChangeCheck();

            Environment env = target as Environment;
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
            if (EditorGUI.EndChangeCheck())
            {
                if (env != null)
                {
                    env.enabled = false;
                    env.enabled = true;
                }
            }
            serializedObject.ApplyModifiedProperties();

            env.lightingFolder = EditorGUILayout.Foldout(env.lightingFolder, "Lighting");
            if (env.lightingFolder)
            {
                ToolsUtility.BeginGroup("RoleLight");
                LightInstpetorGui(roleLight0, env.roleLight0Rot, "RoleLight0");
                LightInstpetorGui(roleLight1, env.roleLight1Rot, "RoleLight1");
                ToolsUtility.EndGroup();
            }

            if (ToolsUtility.BeginFolderGroup("Shadow", ref env.shadowFolder))
            {
                shadowMapLevel.floatValue = EditorGUILayout.Slider("ShadowMap Scale", shadowMapLevel.floatValue, 0, 1);

                if (env.shadowMapLevel == 0 || env.shadowMap == null)
                {
                    EditorGUILayout.ObjectField(env.shadowMap, typeof(RenderTexture), false);
                }
                else
                {
                    float size = 256 * env.shadowMapLevel;
                    GUILayout.Space(size + 20);
                    Rect r = GUILayoutUtility.GetLastRect();
                    r.y += 10;
                    r.width = size;
                    r.height = size;
                    EditorGUI.DrawPreviewTexture(r, env.shadowMap);

                }
                EditorGUILayout.PropertyField(shadowBound);
                EditorGUILayout.PropertyField(lookTarget);
                EditorGUILayout.PropertyField(drawShadowLighing);
                ToolsUtility.EndFolderGroup();
            }

            if (ToolsUtility.BeginFolderGroup("Debug", ref env.debugFolder))
            {
                EditorGUILayout.PropertyField(drawType);

                string[] debugNames = AssetsConfig.shaderDebugNames;
                env.debugContext.shaderID = Environment.debugShaderIDS;

                if (debugNames != null)
                {
                    EditorGUI.BeginChangeCheck();
                    debugMode.intValue = EditorGUILayout.Popup("DebugMode", debugMode.intValue, debugNames);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "GlobalDebugMode");
                        env.debugContext.modeModify = true;
                    }
                    EditorGUI.BeginChangeCheck();
                    EditorGUILayout.PropertyField(debugDisplayType);
                    if (EditorGUI.EndChangeCheck())
                    {
                        Undo.RecordObject(target, "DebugDisplay");
                        env.debugContext.typeModify = true;
                    }
                    if (debugDisplayType.intValue == (int)DebugDisplayType.Split)
                    {
                        EditorGUI.BeginChangeCheck();
                        PropertyField(splitAngle);
                        if (EditorGUI.EndChangeCheck())
                        {
                            Undo.RecordObject(target, "SplitAngle");
                            env.debugContext.angleModify = true;
                        }
                        EditorGUI.BeginChangeCheck();
                        splitPos.floatValue = EditorGUILayout.Slider("SplitPos", splitPos.floatValue, splitLeft, splitRight);
                        if (EditorGUI.EndChangeCheck())
                        {
                            Undo.RecordObject(target, "SplitPos");
                            env.debugContext.posModify = true;
                        }
                    }
                }
                ToolsUtility.EndFolderGroup();
            }
        }


        void LightInstpetorGui(SerializedProperty lightSP, TransformRotationGUIWrapper wrapper, string name)
        {
            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(lightSP);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(target, "lightSP" + name);
            }

            Light light = lightSP.objectReferenceValue as Light;
            if (light != null)
            {
                EditorGUI.BeginChangeCheck();
                Color color = EditorGUILayout.ColorField("Color", light.color);
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RecordObject(target, "Color" + name);
                    light.color = color;
                }
                EditorGUI.BeginChangeCheck();
                float intensity = EditorGUILayout.Slider("Intensity", light.intensity, 0, 10.0f);
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RecordObject(target, "Intensity" + name);
                    light.intensity = intensity;
                }
                EditorGUI.BeginChangeCheck();
                if (wrapper != null)
                    wrapper.OnGUI();
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RecordObject(target, "rot" + name);
                }
            }
        }
    }

}