using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;
using UnityEditor.IMGUI.Controls;

namespace CFEngine.Editor
{
    [CustomEditor(typeof(EnverinmentArea))]
    public class EnverinmentAreaEditor : BaseEditor<EnverinmentArea>
    {
        private SerializedProperty dynamicSceneName;
        private SerializedParameter lerpTime;
        private SerializedProperty color;

        public void OnEnable()
        {
            dynamicSceneName = FindProperty(x => x.dynamicSceneName);
            lerpTime = FindParameter(x => x.lerpTime);
            color = FindProperty(x => x.color);
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EnverinmentArea ea = target as EnverinmentArea;
            if (ea != null)
            {
                EditorGUILayout.PropertyField(dynamicSceneName);
                PropertyField(lerpTime);
                if (ToolsUtility.BeginFolderGroup("Area", ref ea.areaFolder))
                {
                    if (GUILayout.Button("Add"))
                    {
                        ea.areaList.Add(new EnvBox()
                        {
                            center = new Vector3(0, 5, 0),
                            size = new Vector3(10, 10, 10),
                        });
                    }
                    int deleteIndex = -1;
                    Transform t = ea.transform;
                    for (int i = 0; i < ea.areaList.Count; ++i)
                    {
                        var box = ea.areaList[i];
                        ToolsUtility.BeginGroup("Box" + i.ToString());
                        EditorGUI.BeginChangeCheck();
                        box.center = EditorGUILayout.Vector3Field("Center", box.center);
                        box.size = EditorGUILayout.Vector3Field("Size", box.size);
                        box.rotY = EditorGUILayout.FloatField("Rot", box.rotY);
                        if (EditorGUI.EndChangeCheck())
                        {
                            SceneView.RepaintAll();
                        }
                        // EditorGUILayout.BeginHorizontal();

                        if (GUILayout.Button("Delete"))
                        {
                            deleteIndex = i;
                        }
                        // EditorGUILayout.EndHorizontal();
                        ToolsUtility.EndGroup();
                    }
                    if (deleteIndex >= 0)
                    {
                        ea.areaList.RemoveAt(deleteIndex);
                    }
                    EditorGUILayout.PropertyField(color);
                    ToolsUtility.EndFolderGroup();
                }

                if (ToolsUtility.BeginFolderGroup("Modifys", ref ea.envModifyFolder))
                {
                    for (int i = 0; i < ea.envModifyList.Length; ++i)
                    {
                        var modify = ea.envModifyList[i];
                        EnverimentModifyType type = (EnverimentModifyType)i;
                        ToolsUtility.BeginGroup(type.ToString());
                        if (modify != null)
                        {
                            modify.valid = EditorGUILayout.Toggle("Edit", modify.valid);
                            if (modify.valid)
                            {
                                modify.OnGUI();
                            }
                        }

                        ToolsUtility.EndGroup();
                    }
                    ToolsUtility.EndFolderGroup();
                }

            }

            serializedObject.ApplyModifiedProperties();
        }
        private Vector3 TransformColliderCenterToHandleSpace(Transform t, Vector3 boxCenter)
        {
            return Handles.inverseMatrix * (t.localToWorldMatrix * boxCenter);
        }

        protected Vector3 TransformHandleCenterToColliderSpace(Transform colliderTransform, Vector3 handleCenter)
        {
            return colliderTransform.localToWorldMatrix.inverse * (Handles.matrix * handleCenter);
        }
        void OnSceneGUI()
        {
            EnverinmentArea ea = target as EnverinmentArea;
            if (ea != null)
            {
                Transform t = ea.transform;
                for (int i = 0; i < ea.areaList.Count; ++i)
                {
                    var areaBox = ea.areaList[i];

                    areaBox.boundsHandle.SetColor(ea.color);

                    Quaternion rot = Quaternion.Euler(0, areaBox.rotY, 0);                    
                    Vector3 worldPos = t.position + areaBox.center;

                    using (new Handles.DrawingScope(Matrix4x4.TRS(Vector3.zero, rot, Vector3.one)))
                    {
                        areaBox.boundsHandle.center = Handles.inverseMatrix * worldPos;
                        areaBox.boundsHandle.size = areaBox.size;
                        EditorGUI.BeginChangeCheck();
                        areaBox.boundsHandle.DrawHandle();
                        if (EditorGUI.EndChangeCheck())
                        {
                            Undo.RecordObject(ea, string.Format("Modify {0}", ObjectNames.NicifyVariableName(target.GetType().Name)));
                            areaBox.size = areaBox.boundsHandle.size;
                        }
                    }

                    if (Tools.current == Tool.Move)
                    {
                        EditorGUI.BeginChangeCheck();
                        Vector3 pos = Handles.PositionHandle(worldPos, rot);
                        if (EditorGUI.EndChangeCheck())
                        {
                            areaBox.boundsHandle.center = pos;
                            areaBox.center = pos - t.position;
                        }
                    }
                    else if (Tools.current == Tool.Rotate)
                    {
                        EditorGUI.BeginChangeCheck();
                        Quaternion newRot = Handles.RotationHandle(rot, worldPos);
                        if (EditorGUI.EndChangeCheck())
                        {
                            Vector3 euler = newRot.eulerAngles;
                            areaBox.rotY = euler.y;
                        }
                    }
                }
            }
        }
    }

}
