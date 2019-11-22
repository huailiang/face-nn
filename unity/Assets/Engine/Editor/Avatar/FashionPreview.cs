using CFUtilPoolLib;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace XEngine.Editor
{
    public class FashionPreview : EditorWindow
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

        public static FashionPreview preview;

        [MenuItem("Tools/Preview")]
        static void AnimExportTool()
        {
            if (XEditorUtil.MakeNewScene())
            {
                if (preview != null)
                {
                    ScriptableObject.DestroyImmediate(preview);
                }
                var window = EditorWindow.GetWindowWithRect(typeof(FashionPreview), new Rect(0, 0, 440, 730), true, XEditorUtil.Config.preview);
                preview = window as FashionPreview;
                preview.Show();
            }
        }

        public void NeuralProcess(NeuralData data, bool complete)
        {
            OnEnable();
            this.shape = data.shape;
            bool isnew = CreateAvatar();
            suit_select = UnityEngine.Random.Range(0, fashionInfo.Length - 1);
            DrawSuit(isnew, complete);
            Update();
            bone.NeuralProcess(data.boneArgs);
            paint.NeuralProcess(data.paintArgs);
            data.callback(data.name, data.shape);
        }

        private void OnEnable()
        {
            suit_select = 0;
            suit_pre = -1;
            shape = RoleShape.FEMALE;
            if (fData == null)
            {
                fData = AssetDatabase.LoadAssetAtPath<FaceData>("Assets/Resource/Config/FaceData.asset");
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
            GUILayout.Space(4);
            shape = (RoleShape)EditorGUILayout.EnumPopup("Gender", shape);
            clip = (AnimationClip)EditorGUILayout.ObjectField("Clip    ", clip, typeof(AnimationClip), true);

            if (go == null || (go != null && go.name != shape.ToString()) || shape.ToString() != go.name)
            {
                CreateAvatar();
            }
            if (fashionInfo != null)
            {
                suit_select = EditorGUILayout.Popup("Suit    ", suit_select, fashionDesInfo);
                if (suit_pre != suit_select || shape != shape_pre)
                {
                    DrawSuit(true, true);
                    suit_pre = suit_select;
                    shape_pre = shape;
                }
            }
            GUILayout.EndVertical();
            GUILayout.Space(12);
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Sync-Picture", GUILayout.Width(120)))
            {
                string name = "";
                float[] args = new float[NeuralInterface.CNT];
                float[] args2 = new float[NeuralInterface.CNT2];
                if (NeuralInterface.ParseFromPicture(ref args, ref args2, ref name))
                {
                    bone.NeuralProcess(args);
                    paint.NeuralProcess(args2);
                }
            }
            if (GUILayout.Button("Sync-Model", GUILayout.Width(120)))
            {
                string file = UnityEditor.EditorUtility.OpenFilePanel("Select model file", NeuralInterface.MODEL, "bytes");
                if (!string.IsNullOrEmpty(file))
                {
                    FileInfo info = new FileInfo(file);
                    float[] args, args2;
                    NeuralInterface.ProcessFile(info, out args, out args2);
                    bone.NeuralProcess(args);
                    paint.NeuralProcess(args2);
                }
            }
            GUILayout.EndHorizontal();
            paint.OnGui();
            bone.OnGui();
        }

        private bool CreateAvatar()
        {
            if (go == null || go.name != shape.ToString())
            {
                XEditorUtil.ClearCreatures();
                List<int> list = new List<int>();
                var table = XFashionLibrary._profession.Table;
                presentid = table.Where(x => x.Shape == (int)shape).Select(x => x.PresentID).First();
                string path = "Assets/Resource/Prefabs/Player_" + shape.ToString().ToLower() + ".prefab";
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
                return true;
            }
            return false;
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

        private void DrawSuit(bool initial, bool complete)
        {
            if (initial)
            {
                if (fashionInfo.Length <= suit_select) suit_select = 0;
                FashionSuit.RowData rowData = fashionInfo[suit_select];
                FashionUtility.DrawSuit(go, rowData, (uint)presentid, 1, complete);
                paint.Initial(go, shape);
                bone.Initial(go, shape);
            }
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