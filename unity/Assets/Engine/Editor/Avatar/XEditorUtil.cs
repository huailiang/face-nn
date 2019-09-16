using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using XEngine;

namespace XEditor
{
    /// <summary>
    /// config配置数据
    /// 只在editor里使用
    /// </summary>
    public class XEditorConfig
    {
        public string title;
        public string tip;
        public string selected;
        public string make;
        public string part;
        public string destruct;
        public string show;
        public string config;
        public string anim;
        public string anim_select;
        public string preview;
        public string clip_info;
        public string bone_info;
        public string skill_exp;
        public string select_exp;
        public string delete;
        public string tl_wait;
        public string tl_lerp;
        public string tl_anim;
        public string tl_end;
        public string eff_angry;
        public string eff_tired;
        public string suit_pre;
        public string suit_shape;
    }


    public class XEditorUtil
    {
        public static XEditorConfig _config;

        public static XEditorConfig Config
        {
            get
            {
                if (_config == null)
                {
                    ReadConfig();
                }
                return _config;
            }
        }

        private static void ReadConfig()
        {
            string path = Application.dataPath + "/Engine/Editor/EditorResources/config.txt";
            using (FileStream fs = new FileStream(path, FileMode.Open))
            {
                _config = new XEditorConfig();
                StreamReader reader = new StreamReader(fs, Encoding.UTF8);
                reader.ReadLine();
                _config.title = reader.ReadLine();
                _config.tip = reader.ReadLine();
                _config.selected = reader.ReadLine();
                _config.make = reader.ReadLine();
                _config.part = reader.ReadLine();
                _config.destruct = reader.ReadLine();
                _config.show = reader.ReadLine();
                _config.config = reader.ReadLine();
                _config.anim = reader.ReadLine();
                _config.anim_select = reader.ReadLine();
                _config.preview = reader.ReadLine();
                _config.clip_info = reader.ReadLine();
                _config.bone_info = reader.ReadLine();
                _config.skill_exp = reader.ReadLine();
                _config.select_exp = reader.ReadLine();
                _config.delete = reader.ReadLine();
                _config.tl_wait = reader.ReadLine();
                _config.tl_lerp = reader.ReadLine();
                _config.tl_anim = reader.ReadLine();
                _config.tl_end = reader.ReadLine();
                _config.eff_angry = reader.ReadLine();
                _config.eff_tired = reader.ReadLine();
                _config.suit_pre = reader.ReadLine();
                _config.suit_shape = reader.ReadLine();
                reader.Close();
            }
        }


        public static bool MakeNewScene()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
            {
                return false;
            }
            else
            {
                EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects);

                GameObject oldCamera = GameObject.Find(@"Main Camera");
                GameObject.DestroyImmediate(oldCamera);
                GameObject camera = AssetDatabase.LoadAssetAtPath("Assets/Engine/Editor/EditorResources/Main Camera.prefab", typeof(GameObject)) as GameObject;
                camera = GameObject.Instantiate<GameObject>(camera, null);
                camera.transform.position = new Vector3(0, 1, -10);
                Light light = GameObject.Find("Directional Light").GetComponent<Light>();
                light.transform.parent = camera.transform;
                EnverinmentExtra ee = camera.GetComponent<EnverinmentExtra>();
                ee.roleLight0 = light;

                GameObject plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
                plane.name = "Ground";
                plane.layer = LayerMask.NameToLayer("Terrain");
                plane.transform.position = new Vector3(0, -0.01f, 0);
                plane.transform.localScale = new Vector3(1000, 1, 1000);
                plane.GetComponent<Renderer>().sharedMaterial = new Material(Shader.Find("Diffuse"));
                plane.GetComponent<Renderer>().sharedMaterial.SetColor("_Color", new Color(90 / 255.0f, 90 / 255.0f, 90 / 255.0f));
                return true;
            }
        }

        public static void ClearCreatures()
        {
            var objs = GameObject.FindObjectsOfType<Animator>();
            if (objs != null)
            {
                foreach (var item in objs)
                {
                    GameObject.DestroyImmediate(item.gameObject);
                }
            }
            var bjs = GameObject.FindGameObjectsWithTag("EditorOnly");
            if (bjs != null)
            {
                foreach (var item in bjs)
                {
                    GameObject.DestroyImmediate(item.gameObject);
                }
            }
        }

        [System.NonSerialized] private static GUIStyle _labelStyle = null;

        [System.NonSerialized] private static GUIStyle _buttonStyle = null;

        [System.NonSerialized] private static GUIStyle _folderStyle = null;

        public static GUIStyle folderStyle
        {
            get
            {
                if (_folderStyle == null) _folderStyle = new GUIStyle(EditorStyles.foldout);
                _folderStyle.fontStyle = FontStyle.Bold;
                _folderStyle.fontSize = 20;
                return _folderStyle;
            }
        }

        public static GUIStyle titleLableStyle
        {
            get
            {
                if (_labelStyle == null)
                    _labelStyle = new GUIStyle(EditorStyles.label);
                _labelStyle.fontStyle = FontStyle.Bold;
                _labelStyle.fontSize = 22;
                return _labelStyle;
            }
        }


        public static GUIStyle boldLableStyle
        {
            get
            {
                if (_labelStyle == null)
                    _labelStyle = new GUIStyle(EditorStyles.label);
                _labelStyle.fontStyle = FontStyle.Bold;
                return _labelStyle;
            }
        }


        public static GUIStyle boldButtonStyle
        {
            get
            {
                if (_buttonStyle == null)
                    _buttonStyle = new GUIStyle(GUI.skin.button);
                _buttonStyle.fontStyle = FontStyle.Bold;
                return _buttonStyle;
            }
        }

        public static string[] Transf2Str(Transform[] transf)
        {
            if (transf != null)
            {
                string[] arr = new string[transf.Length];
                for (int i = 0; i < transf.Length; i++)
                {
                    arr[i] = transf != null ? transf[i].name : null;
                }
                return arr;
            }
            return null;
        }

        public static Transform[] Str2Transf(string[] str, GameObject go)
        {
            if (str != null && go != null)
            {
                XBoneTree tree = new XBoneTree(string.Empty, go.name, 0);
                tree.FillChilds(go.transform);

                Transform[] rst = new Transform[str.Length];
                for (int i = 0; i < str.Length; i++)
                {
                    var child = tree.SearchChild(str[i], go);
                    rst[i] = child;
                }
                return rst;
            }
            return null;
        }


        public static void CheckDBNil(string[] str, Transform[] tfs)
        {
            if (str != null && tfs != null)
            {
                for (int i = 0; i < str.Length; i++)
                {
                    if (tfs[i] != null)
                    {
                        str[i] = string.Empty;
                    }
                }
            }
        }

        public static void ClearMesh(Mesh newMesh)
        {
            //newMesh.uv2 = null;
            newMesh.uv3 = null;
            newMesh.uv4 = null;
            newMesh.uv5 = null;
            newMesh.uv6 = null;
            newMesh.uv7 = null;
            newMesh.colors = null;
            newMesh.colors32 = null;
        }



    }

}