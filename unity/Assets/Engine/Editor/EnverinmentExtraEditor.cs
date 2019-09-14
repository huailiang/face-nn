using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;

namespace CFEngine.Editor
{
    [CustomEditor(typeof(EnverinmentExtra))]
    public class EnverinmentExtraEditor : BaseEditor<EnverinmentExtra>
    {
        //lighting
        private SerializedProperty roleLight0;
        private SerializedProperty roleLight1;
        private SerializedProperty baleSceneLight0;
        private SerializedProperty bakeSceneLight1;
        private SerializedProperty sceneRuntimeLight0;
        private SerializedProperty sceneRuntimeLight1;

        private SerializedProperty sunLight;

        private SerializedProperty fastEditLight;
        private SerializedProperty fastEditEnvLight;
        private SerializedProperty useUnityLighting;

        //testObj
        private SerializedProperty pointLight;
        private SerializedProperty roleDummy;
        private SerializedProperty interactiveParam;
        private SerializedProperty debugEnvArea;
        //fast run
        private SerializedProperty loadGameAtHere;
        private SerializedProperty useCurrentScene;
        private SerializedProperty sceneID;
        private SerializedProperty replaceStartScene;

        private SerializedProperty gotoScene;
        private SerializedProperty useStaticBatch;

        //freeCamera
        private SerializedProperty forceUpdateFreeCamera;
        private SerializedProperty holdRightMouseCapture;
        private SerializedProperty lookSpeed;
        private SerializedProperty moveSpeed;
        private SerializedProperty sprintSpeed;

        //debugShadow
        private SerializedProperty fitWorldSpace;
        // private SerializedProperty shadowCasterProxy;
        private SerializedProperty lookTarget;
        private SerializedProperty shadowMapLevel;
        private SerializedProperty shadowBound;
        private SerializedProperty drawShadowLighing;

        //debug
        private SerializedProperty drawFrustum;
        private SerializedProperty drawLodGrid;
        private SerializedProperty drawTerrainGrid;

        private SerializedProperty drawType;
        private SerializedProperty quadLevel;
        private SerializedProperty quadIndex;
        private SerializedProperty showObjects;
        private SerializedProperty drawInvisibleObj;
        private SerializedProperty drawPointLight;
        private SerializedProperty drawTerrainHeight;
        private SerializedProperty drawLightBox;
        private SerializedProperty isDebugLayer;

        private SerializedProperty debugMode;
        private SerializedProperty debugDisplayType;
        private SerializedParameter splitAngle;
        private SerializedProperty splitPos;

        //voxelLight
        private SerializedProperty lightMode;
        private SerializedProperty lightGridSize;
        private SerializedProperty minLightCount;
        private SerializedProperty maxLightCount;
        private SerializedProperty previewLightCount;
        private float splitLeft = -1;
        private float splitRight = 1;

        public void OnEnable()
        {
            roleLight0 = FindProperty(x => x.roleLight0);
            roleLight1 = FindProperty(x => x.roleLight1);
            baleSceneLight0 = FindProperty(x => x.bakeSceneLight0);
            bakeSceneLight1 = FindProperty(x => x.bakeSceneLight1);
            sceneRuntimeLight0 = FindProperty(x => x.sceneRuntimeLight0);
            sceneRuntimeLight1 = FindProperty(x => x.sceneRuntimeLight1);
            sunLight = FindProperty(x => x.sunLight);

            fastEditLight = FindProperty(x => x.fastEditLight);
            fastEditEnvLight = FindProperty(x => x.fastEditEnvLight);
            useUnityLighting = FindProperty(x => x.useUnityLighting);
            pointLight = FindProperty(x => x.pointLight);
            roleDummy = FindProperty(x => x.roleDummy);
            interactiveParam = FindProperty(x => x.interactiveParam);

            loadGameAtHere = FindProperty(x => x.loadGameAtHere);
            useCurrentScene = FindProperty(x => x.useCurrentScene);
            sceneID = FindProperty(x => x.sceneID);
            replaceStartScene = FindProperty(x => x.replaceStartScene);
            gotoScene = FindProperty(x => x.gotoScene);
            useStaticBatch = FindProperty(x => x.useStaticBatch);

            forceUpdateFreeCamera = FindProperty(x => x.forceUpdateFreeCamera);
            holdRightMouseCapture = FindProperty(x => x.holdRightMouseCapture);
            lookSpeed = FindProperty(x => x.lookSpeed);
            moveSpeed = FindProperty(x => x.moveSpeed);
            sprintSpeed = FindProperty(x => x.sprintSpeed);

            fitWorldSpace = FindProperty(x => x.fitWorldSpace);
            shadowMapLevel = FindProperty(x => x.shadowMapLevel);
            shadowBound = FindProperty(x => x.shadowBound);
            lookTarget = FindProperty(x => x.lookTarget);
            drawShadowLighing = FindProperty(x => x.drawShadowLighing);

            drawFrustum = FindProperty(x => x.drawFrustum);
            drawLodGrid = FindProperty(x => x.drawLodGrid);
            drawTerrainGrid = FindProperty(x => x.drawTerrainGrid);

            lightMode = FindProperty(x => x.lightMode);
            lightGridSize = FindProperty(x => x.lightLoopContext.lightGridSize);
            minLightCount = FindProperty(x => x.minLightCount);
            maxLightCount = FindProperty(x => x.maxLightCount);
            previewLightCount = FindProperty(x => x.previewLightCount);

            drawType = FindProperty(x => x.drawType);
            quadLevel = FindProperty(x => x.quadLevel);
            quadIndex = FindProperty(x => x.quadIndex);
            showObjects = FindProperty(x => x.showObjects);
            drawInvisibleObj = FindProperty(x => x.drawInvisibleObj);
            drawPointLight = FindProperty(x => x.drawPointLight);
            drawTerrainHeight = FindProperty(x => x.drawTerrainHeight);
            debugEnvArea = FindProperty(x => x.debugEnvArea);
            drawLightBox = FindProperty(x => x.drawLightBox);

            isDebugLayer = FindProperty(x => x.isDebugLayer);
            debugMode = FindProperty(x => x.debugContext.debugMode);
            debugDisplayType = FindProperty(x => x.debugContext.debugDisplayType);
            splitAngle = FindParameter(x => x.debugContext.splitAngle);
            splitPos = FindProperty(x => x.debugContext.splitPos);

            AssetsConfig.RefreshShaderDebugNames();
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
        private void CalcSplitLeftRight()
        {
            float k = Mathf.Tan(Mathf.Deg2Rad * (90 - splitAngle.value.floatValue));
            float b = 1 + k;
            splitLeft = -b / k;
            splitRight = -splitLeft;

        }
        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            EditorGUI.BeginChangeCheck();
            EditorGUILayout.PropertyField(fastEditLight);

            EditorGUILayout.PropertyField(fastEditEnvLight);
            EnverinmentExtra ee = target as EnverinmentExtra;
            if (ee != null)
            {
                ee.lightingFolder = EditorGUILayout.Foldout(ee.lightingFolder, "Lighting");
                if (ee.lightingFolder)
                {

                    ToolsUtility.BeginGroup("RoleLight");
                    LightInstpetorGui(roleLight0, ee.roleLight0Rot, "RoleLight0");
                    LightInstpetorGui(roleLight1, ee.roleLight1Rot, "RoleLight1");
                    ToolsUtility.EndGroup();


                    ToolsUtility.BeginGroup("SceneLight");
                    LightInstpetorGui(baleSceneLight0, ee.bakeSceneLight0Rot, "SceneLight0");
                    LightInstpetorGui(bakeSceneLight1, ee.bakeSceneLight1Rot, "SceneLight1");
                    ToolsUtility.EndGroup();


                    ToolsUtility.BeginGroup("RuntimeSceneLight");
                    LightInstpetorGui(sceneRuntimeLight0, ee.sceneRuntimeLight0Rot, "RuntimeSceneLight0");
                    LightInstpetorGui(sceneRuntimeLight1, ee.sceneRuntimeLight1Rot, "RuntimeSceneLight1");
                    ToolsUtility.EndGroup();
                    ToolsUtility.BeginGroup("Sun Light");
                    LightInstpetorGui(sunLight, ee.sunLightRot, "SunLight");
                    ToolsUtility.EndGroup();
                }

                if (ToolsUtility.BeginFolderGroup("TestObj", ref ee.testObjFolder))
                {
                    EditorGUILayout.PropertyField(pointLight);
                    EditorGUILayout.Space();

                    EditorGUILayout.PropertyField(roleDummy);
                    EditorGUILayout.PropertyField(interactiveParam);
                    EditorGUILayout.PropertyField(debugEnvArea);
                    if (GUILayout.Button("Refresh"))
                    {
                        ee.RefreshEnvArea();
                    }
                    for (int i = 0; i < ee.envObjects.Count; ++i)
                    {
                        var envObj = ee.envObjects[i];
                        EditorGUILayout.ObjectField(envObj, typeof(EnverinmentArea), true);
                    }
                    ToolsUtility.EndFolderGroup();
                }

                if (ToolsUtility.BeginFolderGroup("FastRun", ref ee.fastRunFolder))
                {
                    EditorGUILayout.PropertyField(loadGameAtHere);
                    EditorGUILayout.PropertyField(useCurrentScene);
                    EditorGUILayout.PropertyField(gotoScene);
                    EditorGUILayout.PropertyField(sceneID);
                    EditorGUILayout.PropertyField(replaceStartScene);
                    EditorGUILayout.PropertyField(useStaticBatch);
                    ToolsUtility.EndFolderGroup();
                }

                if (ToolsUtility.BeginFolderGroup("FreeCamera", ref ee.freeCameraFolder))
                {
                    EditorGUILayout.PropertyField(forceUpdateFreeCamera);
                    EditorGUILayout.PropertyField(holdRightMouseCapture);
                    EditorGUILayout.PropertyField(lookSpeed);
                    EditorGUILayout.PropertyField(moveSpeed);
                    EditorGUILayout.PropertyField(sprintSpeed);
                    ToolsUtility.EndFolderGroup();
                }

                if (ToolsUtility.BeginFolderGroup("Shadow", ref ee.shadowFolder))
                {
                    shadowMapLevel.floatValue = EditorGUILayout.Slider("ShadowMap Scale", shadowMapLevel.floatValue, 0, 1);

                    if (ee.shadowMapLevel == 0 || ee.shadowMap == null)
                    {
                        EditorGUILayout.ObjectField(ee.shadowMap, typeof(RenderTexture), false);
                    }
                    else
                    {
                        float size = 256 * ee.shadowMapLevel;
                        GUILayout.Space(size + 20);
                        Rect r = GUILayoutUtility.GetLastRect();
                        r.y += 10;
                        r.width = size;
                        r.height = size;
                        EditorGUI.DrawPreviewTexture(r, ee.shadowMap);

                    }
                    EditorGUILayout.PropertyField(shadowBound);
                    EditorGUILayout.PropertyField(lookTarget);
                    EditorGUILayout.PropertyField(fitWorldSpace);
                    EditorGUILayout.PropertyField(drawShadowLighing);
                    ToolsUtility.EndFolderGroup();
                }

                if (ToolsUtility.BeginFolderGroup("VoxelLight", ref ee.voxelLightFolder))
                {
                    EditorGUILayout.PropertyField(lightMode);
                    if (lightMode.intValue == (int)LightingMode.Voxel)
                    {
                        Shader.EnableKeyword("_VOXEL_LIGHT");
                        if (GUILayout.Button("Collect Lights", GUILayout.MaxWidth(160)))
                        {
                            ee.CollectLights();
                        }
                        int size = lightGridSize.intValue;
                        EditorGUI.BeginChangeCheck();

                        int newSize = EditorGUILayout.IntSlider("Light Grid Size", lightGridSize.intValue, 1, 5);
                        if (EditorGUI.EndChangeCheck())
                        {
                            Undo.RecordObject(target, "Light Grid Size");
                            if (newSize != size)
                            {
                                lightGridSize.intValue = newSize;
                            }
                        }
                        EditorGUILayout.PropertyField(lightGridSize);
                    }
                    else
                    {
                        Shader.DisableKeyword("_VOXEL_LIGHT");
                    }
                    ToolsUtility.EndFolderGroup();
                }

                if (ToolsUtility.BeginFolderGroup("Debug", ref ee.debugFolder))
                {
                    EditorGUILayout.PropertyField(drawFrustum);
                    EditorGUILayout.PropertyField(drawInvisibleObj);
                    EditorGUILayout.PropertyField(drawLodGrid);
                    EditorGUILayout.PropertyField(drawTerrainGrid);
                    EditorGUILayout.PropertyField(drawType);

                    int level = quadLevel.intValue;
                    EditorGUILayout.PropertyField(quadLevel);

                    if (quadLevel.intValue == (int)QuadTreeLevel.Level3)
                    {
                        int index = quadIndex.intValue;

                        EditorGUI.BeginChangeCheck();
                        int newindex = EditorGUILayout.IntSlider("cull block", quadIndex.intValue, -1, 15);
                        if (EditorGUI.EndChangeCheck())
                        {
                            Undo.RecordObject(target, "cull back");
                            if (index != newindex)
                            {
                                quadIndex.intValue = newindex;
                                ee.updateSceneObject = true;

                            }

                        }
                        if (quadIndex.intValue > -1)
                        {
                            EditorGUILayout.PropertyField(showObjects);
                            if (showObjects.boolValue)
                            {
                                for (int i = 0; i < ee.sceneObjects.Count; ++i)
                                {
                                    var so = ee.sceneObjects[i];
                                    string tex = (so.draw ? "draw:" : "cull:") + so.id.ToString();
                                    EditorGUILayout.ObjectField(tex, so.asset.obj as Mesh, typeof(Mesh), false);
                                }
                            }
                        }

                    }
                    else
                    {
                        if (level != quadLevel.intValue)
                        {
                            ee.updateSceneObject = true;
                        }
                    }

                    EditorGUILayout.PropertyField(drawPointLight);
                    EditorGUILayout.PropertyField(drawTerrainHeight);
                    EditorGUILayout.PropertyField(drawLightBox);
                    if (drawLightBox.boolValue)
                    {
                        int size = previewLightCount.intValue;
                        EditorGUI.BeginChangeCheck();

                        int newSize = EditorGUILayout.IntSlider("LightCount", previewLightCount.intValue, 0, ee.maxLightCount);
                        if (EditorGUI.EndChangeCheck())
                        {
                            Undo.RecordObject(target, "LightCount");
                            if (newSize != size)
                            {
                                previewLightCount.intValue = newSize;
                            }
                        }
                    }
                    EditorGUI.BeginChangeCheck();
                    if (EditorGUI.EndChangeCheck())
                    {
                        debugMode.intValue = 0;
                        debugDisplayType.intValue = 0;
                        splitAngle.value.floatValue = 0;
                        splitPos.floatValue = 0;
                    }
                    string[] debugNames = null;
                    bool refreshDebug = false;

                    if (GUILayout.Button("Refresh"))
                    {
                        refreshDebug = true;
                    }
                    debugNames = AssetsConfig.shaderDebugNames;
                    ee.debugContext.shaderID = EnverinmentExtra.debugShaderIDS;


                    if (debugNames != null)
                    {
                        EditorGUI.BeginChangeCheck();
                        debugMode.intValue = EditorGUILayout.Popup("DebugMode", debugMode.intValue, debugNames);
                        if (EditorGUI.EndChangeCheck())
                        {
                            Undo.RecordObject(target, "GlobalDebugMode");
                            ee.debugContext.modeModify = true;
                        }
                        EditorGUI.BeginChangeCheck();
                        EditorGUILayout.PropertyField(debugDisplayType);
                        if (EditorGUI.EndChangeCheck())
                        {
                            Undo.RecordObject(target, "DebugDisplayType");
                            ee.debugContext.typeModify = true;
                        }
                        if (debugDisplayType.intValue == (int)DebugDisplayType.Split)
                        {
                            EditorGUI.BeginChangeCheck();
                            PropertyField(splitAngle);
                            if (EditorGUI.EndChangeCheck())
                            {
                                Undo.RecordObject(target, "SplitAngle");
                                ee.debugContext.angleModify = true;
                            }
                            EditorGUI.BeginChangeCheck();
                            splitPos.floatValue = EditorGUILayout.Slider("SplitPos", splitPos.floatValue, splitLeft, splitRight);
                            if (EditorGUI.EndChangeCheck())
                            {
                                Undo.RecordObject(target, "SplitPos");
                                ee.debugContext.posModify = true;
                            }
                        }

                    }
                    ToolsUtility.EndFolderGroup();
                    if (refreshDebug)
                    {
                        AssetsConfig.RefreshShaderDebugNames();
                    }
                }
            }
            serializedObject.ApplyModifiedProperties();
        }

        void DrawLightHandle(Transform t, Vector3 centerPos, float right, float up, Light l, string text)
        {
            if (l != null)
            {
                Vector3 pos = centerPos + t.right * right + t.up * up;
                EditorGUI.BeginChangeCheck();
                Transform lt = l.transform;
                Quaternion rot = Handles.RotationHandle(lt.rotation, pos);
                if (EditorGUI.EndChangeCheck())
                {
                    Undo.RecordObject(target, text);
                    lt.rotation = rot;
                }

                Handles.color = l.color;
                Handles.ArrowHandleCap(100, pos, rot, 2 * l.intensity, EventType.Repaint);
                Handles.Label(pos, text);
            }
        }

        void OnSceneGUI()
        {
            if (SceneView.lastActiveSceneView != null &&
                SceneView.lastActiveSceneView.camera != null)
            {

                EnverinmentExtra ee = target as EnverinmentExtra;
                if (ee != null)
                {
                    Color temp = Handles.color;
                    Transform t = SceneView.lastActiveSceneView.camera.transform;
                    Vector3 pos = t.position + t.forward * 10;
                    if (fastEditLight != null && fastEditLight.boolValue)
                    {
                        DrawLightHandle(t, pos, -4, 3, ee.roleLight0, "RoleLight0");
                        DrawLightHandle(t, pos, -4, -3, ee.roleLight1, "RoleLight1");

                        bool isRuntime = RenderingEnvironment.isPreview;
                        Light sceneLight0;
                        Light sceneLight1;
                        if (isRuntime)
                        {
                            sceneLight0 = ee.sceneRuntimeLight0;
                            sceneLight1 = ee.sceneRuntimeLight1;
                        }
                        else
                        {
                            sceneLight0 = ee.bakeSceneLight0;
                            sceneLight1 = ee.bakeSceneLight1;
                        }
                        DrawLightHandle(t, pos, 4, 3, sceneLight0, "SceneLight0");
                        DrawLightHandle(t, pos, 4, -3, sceneLight1, "SceneLight1");
                    }
                    if (fastEditEnvLight != null && fastEditEnvLight.boolValue)
                    {
                        DrawLightHandle(t, pos, -4, 3, ee.sunLight, "SunLight");
                    }
                    Handles.color = temp;
                }
            }
        }
    }
}
