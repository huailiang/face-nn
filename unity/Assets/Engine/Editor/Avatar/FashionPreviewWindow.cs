using CFUtilPoolLib;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using System;
using System.Linq;


namespace XEditor
{
    public class FashionPreviewWindow : EditorWindow
    {
        private float clipTime = 0f;
        private int suit_select = 0;
        private int suit_pre = -1;
        private FashionSuit.RowData[] fashionInfo;
        private string[] fashionDesInfo;
        private RoleShape shape = RoleShape.FEMALE;
        private RoleShape shape_pre = RoleShape.FEMALE;
        private uint presentid = 101;
        private AnimationClip clip;
        private GameObject go;
        private FacePaint paint;
        private FaceBone bone;
        private XEntityPresentation.RowData pData;
        private FaceData fData;
        private NeuralData nData;


        [MenuItem("Tools/FashionPreview")]
        static void AnimExportTool()
        {
            if (XEditorUtil.MakeNewScene())
            {
                EditorWindow.GetWindowWithRect(typeof(FashionPreviewWindow), new Rect(0, 0, 440, 640), true, "FashionPreview");
            }
        }

        public void NeuralProcess(NeuralData data)
        {
            this.shape = data.shape;
            this.nData = data;
        }

        private void OnEnable()
        {
            suit_select = 0;
            suit_pre = -1;
            shape = RoleShape.FEMALE;
            if (fData == null)
            {
                fData = AssetDatabase.LoadAssetAtPath<FaceData>("Assets/BundleRes/Config/FaceData.asset");
            }
            if (paint == null)
            {
                paint = new FacePaint(fData);
            }
            if (bone == null)
            {
                bone = new FaceBone(fData);
            }
        }


        void OnGUI()
        {
            GUILayout.BeginVertical();
            GUILayout.Label(XEditorUtil.Config.suit_pre, XEditorUtil.titleLableStyle);
            GUILayout.Space(8);

            GUILayout.BeginHorizontal();
            GUILayout.Label("Role Shape");
            shape = (RoleShape)EditorGUILayout.EnumPopup(shape);
            GUILayout.EndHorizontal();
            GUILayout.Space(4);

            GUILayout.BeginHorizontal();
            GUILayout.Label("Select Clip");
            clip = (AnimationClip)EditorGUILayout.ObjectField(clip, typeof(AnimationClip), true);
            GUILayout.EndHorizontal();
            GUILayout.Space(4);

            if (go == null || (go != null && go.name != shape.ToString()) || shape.ToString() != go.name)
            {
                XEditorUtil.ClearCreatures();

                List<int> list = new List<int>();
                var table = XFashionLibrary._profession.Table;
                presentid = table.Where(x => x.Shape == (int)shape).Select(x => x.PresentID).First();
                string path = "Assets/BundleRes/Prefabs/Player_" + shape.ToString().ToLower() + ".prefab";
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                if (prefab != null)
                {
                    GameObject root = GameObject.Find("Player");
                    if (root == null)
                    {
                        root = new GameObject("Player");
                        root.transform.position = new Vector3(0f, 0f, -8f);
                    }
                    go = Instantiate(prefab);
                    go.transform.SetParent(root.transform);
                    go.name = shape.ToString();
                    go.transform.localScale = Vector3.one;
                    go.transform.rotation = Quaternion.Euler(0, 180, 0);
                    go.transform.localPosition = Vector3.zero;
                    Selection.activeGameObject = go;
                    fashionInfo = XFashionLibrary.GetFashionsInfo(shape);
                    fashionDesInfo = new string[fashionInfo.Length];
                    for (int i = 0; i < fashionInfo.Length; i++)
                    {
                        fashionDesInfo[i] = fashionInfo[i].name;
                    }
                }
            }

            if (fashionInfo != null)
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label("Select Suit");
                suit_select = EditorGUILayout.Popup(suit_select, fashionDesInfo);
                if (suit_pre != suit_select || shape != shape_pre)
                {
                    DrawSuit();
                    suit_pre = suit_select;
                    shape_pre = shape;
                    paint.Initial(go, shape);
                    bone.Initial(go, shape);
                }
                GUILayout.EndHorizontal();
            }
            GUILayout.EndVertical();
            paint.OnGui();
            bone.OnGui();
            HandleNeural();
        }

        private void HandleNeural()
        {
            if (nData != null)
            {
                paint.Update();
                bone.NeuralProcess(nData.boneArgs);
                nData.callback();
                nData = null;
            }
        }


        private void Update()
        {
            if (go != null)
            {
                PlayAnim();
                if (paint != null)
                {
                    paint.Update();
                }
            }
        }

        private void DrawSuit()
        {
            if (fashionInfo.Length <= suit_select) suit_select = 0;
            FashionSuit.RowData rowData = fashionInfo[suit_select];
            FashionUtil.DrawSuit(go, rowData, (uint)presentid, 1);
        }

        private void PlayAnim()
        {
            if (clip != null && go != null)
            {
                clipTime += Time.deltaTime;
                if (clipTime >= clip.length) clipTime = 0f;
                clip.SampleAnimation(go, clipTime);
            }
        }


    }

}