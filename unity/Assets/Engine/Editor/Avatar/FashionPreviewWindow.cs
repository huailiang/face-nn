using CFUtilPoolLib;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;


/// <summary>
/// 时装预览工具
/// 使用前需要确保资源导出和fashionsuit表格配好
/// </summary>

namespace XEditor
{
    public class FashionPreviewWindow : EditorWindow
    {
        private bool isAutoRot = true;
        private float clipTime = 0f;
        private int suit_select = 0;
        private int suit_pre = -1;
        private FashionSuit.RowData[] fashionInfo;
        private string[] fashionDesInfo;
        private RoleShape shape = RoleShape.FEMALE;
        private int[] presentids = { 101 };
        private string[] spids = { "101" };
        private int presentid = 101;
        private int weapon_index = 1;
        private int pre_presentid = 0;
        private int pre_weapon_index = 0;
        private AnimationClip clip;
        private GameObject go;
        private FacePaint paint;
        private XEntityPresentation.RowData pData;


        [MenuItem("Tools/FashionPreview")]
        static void AnimExportTool()
        {
            if (XEditorUtil.MakeNewScene())
            {
                EditorWindow.GetWindowWithRect(typeof(FashionPreviewWindow), new Rect(0, 0, 440, 500), true, "FashionPreview");
            }
        }


        private void OnEnable()
        {
            isAutoRot = true;
            suit_select = 0;
            suit_pre = -1;
            shape = RoleShape.FEMALE;
            if (paint == null)
            {
                paint = new FacePaint();
            }
        }


        void OnGUI()
        {
            GUILayout.BeginVertical();
            GUILayout.Label(XEditorUtil.Config.suit_pre, XEditorUtil.titleLableStyle);
            GUILayout.Space(8);

            isAutoRot = EditorGUILayout.Toggle("Auto Rot    ", isAutoRot);
            GUILayout.Space(4);

            GUILayout.BeginHorizontal();
            GUILayout.Label("Role Shape");
            shape = (RoleShape)EditorGUILayout.EnumPopup(shape);
            GUILayout.EndHorizontal();
            GUILayout.Space(4);
            GUILayout.BeginHorizontal();
            GUILayout.Label("Presentid: ");
            presentid = EditorGUILayout.IntPopup(presentid, spids, presentids);
            GUILayout.EndHorizontal();
            weapon_index = EditorGUILayout.IntField("weaponIndex", weapon_index);
            GUILayout.Space(8);

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
                for (int i = 0; i < table.Length; i++)
                {
                    if (table[i].Shape == (int)shape)
                    {
                        list.Add((int)table[i].PresentID);
                        if (table[i].SecondaryPresentID != 0)
                        {
                            list.Add((int)table[i].SecondaryPresentID);
                        }
                    }
                }
                list.Sort();
                presentids = list.ToArray();
                if (presentids == null || presentids.Length <= 0) return;
                presentid = presentids[0];
                spids = new string[presentids.Length];
                for (int i = 0; i < presentids.Length; i++)
                {
                    spids[i] = presentids[i].ToString();
                }

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
                    go.transform.rotation = Quaternion.identity;
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
                if (suit_pre != suit_select)
                {
                    DrawSuit();
                    suit_pre = suit_select;
                }
                if (presentid != pre_presentid || weapon_index != pre_weapon_index)
                {
                    DrawSuit();
                    pre_presentid = presentid;
                    pre_weapon_index = weapon_index;
                    paint.Initial(go, shape);
                }
                GUILayout.EndHorizontal();
            }
            GUILayout.EndVertical();
        }


        private void Update()
        {
            if (go != null)
            {
                if (isAutoRot) go.transform.Rotate(Vector3.up, 0.1f);
                PlayAnim();
                // if (paint != null)
                // {
                //     paint.Update();
                // }
            }
        }


        private void DrawSuit()
        {
            if (fashionInfo.Length <= suit_select) suit_select = 0;
            FashionSuit.RowData rowData = fashionInfo[suit_select];
            FashionUtil.DrawSuit(go, rowData, (uint)presentid, weapon_index);
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