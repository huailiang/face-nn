#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;


namespace XEngine
{
    public class TransformRotationGUIWrapper
    {
        public Transform t;
        public System.Object guiObj;
        public MethodInfo mi;

        public void OnGUI()
        {
            if (guiObj != null && mi != null)
            {
                mi.Invoke(guiObj, null);
            }
        }
    }
    public class ReflectFun
    {
        public MethodInfo fun;

        public object Call(object instance, object[] parameters)
        {
            if (fun != null)
            {
                return fun.Invoke(instance, parameters);
            }
            return null;
        }
    }
    public static class EditorCommon
    {
        public delegate void EnumTransform(Transform t, object param);
        static Type transformRotationGUIType = null;
        static Assembly unityEditorAssembly = null;

        public static object CallInternalFunction(Type type, string function, bool isStatic, bool isPrivate, bool isInstance, object obj, object[] parameters)
        {
            System.Reflection.BindingFlags flag = System.Reflection.BindingFlags.Default;
            if (isStatic)
            {
                flag |= System.Reflection.BindingFlags.Static;
            }
            if (isPrivate)
            {
                flag |= System.Reflection.BindingFlags.NonPublic;
            }
            else
            {
                flag |= System.Reflection.BindingFlags.Public;
            }
            if (isInstance)
            {
                flag |= System.Reflection.BindingFlags.Instance;
            }
            System.Reflection.MethodInfo mi = type.GetMethod(function, flag);
            if (mi != null)
            {
                return mi.Invoke(obj, parameters);
            }
            return null;
        }

        public static ReflectFun GetInternalFunction(Type type, string function, bool isStatic, bool isPrivate, bool isInstance, bool baseType, bool all = false)
        {
            System.Reflection.BindingFlags flag = System.Reflection.BindingFlags.Default;
            if(all)
            {
                flag = BindingFlags.NonPublic |
                        BindingFlags.Public |
                        BindingFlags.Instance |
                        BindingFlags.Static;
            }
            else
            {
                if (isStatic)
                {
                    flag |= System.Reflection.BindingFlags.Static;
                }
                if (isPrivate)
                {
                    flag |= System.Reflection.BindingFlags.NonPublic;
                }
                else
                {
                    flag |= System.Reflection.BindingFlags.Public;
                }
                if (isInstance)
                {
                    flag |= System.Reflection.BindingFlags.Instance;
                }
            }

            System.Reflection.MethodInfo mi = baseType ? type.BaseType.GetMethod(function, flag) : type.GetMethod(function, flag);
            if (mi != null)
            {
                return new ReflectFun() { fun = mi };
            }
            return null;
        }

        public static TransformRotationGUIWrapper GetTransformRotatGUI(Transform tran)
        {
            if (transformRotationGUIType == null)
            {
                if (unityEditorAssembly == null)
                {
                    unityEditorAssembly = System.Reflection.Assembly.GetAssembly(typeof(Editor));
                }
                if (unityEditorAssembly != null)
                {
                    transformRotationGUIType = unityEditorAssembly.GetType("UnityEditor.TransformRotationGUI");
                }
            }
            TransformRotationGUIWrapper wrapper = null;
            if (transformRotationGUIType != null)
            {
                System.Object guiObj = Activator.CreateInstance(transformRotationGUIType);
                if (guiObj != null)
                {
                    wrapper = new TransformRotationGUIWrapper();
                    wrapper.t = tran;
                    wrapper.guiObj = guiObj;
                    CallInternalFunction(transformRotationGUIType, "OnEnable", false, false, true, guiObj, new object[] { new SerializedObject(tran).FindProperty("m_LocalRotation"), EditorGUIUtility.TrTextContent("Rotation", "The local rotation.") });

                    wrapper.mi = transformRotationGUIType.GetMethod("RotationField", BindingFlags.Instance | BindingFlags.Public, null, new Type[] { }, null);
                }
            }
            return wrapper;
        }

        public static void SaveFieldInfo(Type src, Type des, object srcObj, object desObj, bool shaowError = true)
        {
            System.Reflection.BindingFlags flag = System.Reflection.BindingFlags.Default;
            flag |= System.Reflection.BindingFlags.Public;
            flag |= System.Reflection.BindingFlags.Instance;
            FieldInfo[] fields = src.GetFields(flag);
            FieldInfo[] fieldsSave = des.GetFields(flag);
            for (int i = 0; i < fields.Length; ++i)
            {
                FieldInfo fi = fields[i];
                FieldInfo saveFi = Array.Find(fieldsSave, (field) => { return field.Name == fi.Name; });
                if (saveFi == null)
                {
                    if (shaowError)
                    {
                        var attr = fi.GetCustomAttributes(typeof(NonSerializedAttribute), false).FirstOrDefault();
                        if (attr == null)
                        {
                            attr = fi.GetCustomAttributes(typeof(NoSerializedAttribute), false).FirstOrDefault();
                            if (attr == null)
                                Debug.LogError(string.Format("Field {0} not find.", fi.Name));
                        }
                    }
                }
                else
                {
                    object value = fi.GetValue(srcObj);
                    saveFi.SetValue(desObj, value);
                }

            }
        }

        public static string GetSceneObjectPath(Transform trans)
        {
            string sceneObjectPath = trans.name;
            Transform parent = trans.parent;
            while (parent != null)
            {
                sceneObjectPath = parent.name + "/" + sceneObjectPath;
                parent = parent.parent;
            }
            return sceneObjectPath;
        }

        public static void WriteMatrix(BinaryWriter bw, Matrix4x4 mat)
        {
            for (int m = 0; m < 16; ++m)
            {
                bw.Write(mat[m]);
            }
        }
        public static void WriteVector(BinaryWriter bw, Vector4 vec)
        {
            bw.Write(vec.x);
            bw.Write(vec.y);
            bw.Write(vec.z);
            bw.Write(vec.w);
        }

        public static void WriteVector(BinaryWriter bw, Vector3 vec)
        {
            bw.Write(vec.x);
            bw.Write(vec.y);
            bw.Write(vec.z);
        }

        public static void WriteVector(BinaryWriter bw, Vector2 vec)
        {
            bw.Write(vec.x);
            bw.Write(vec.y);
        }

        public static Vector2 ReadVector2(BinaryReader br)
        {
            Vector2 vector;
            vector.x = br.ReadSingle();
            vector.y = br.ReadSingle();
            return vector;
        }

        public static Vector3 ReadVector3(BinaryReader br)
        {
            Vector3 vector;
            vector.x = br.ReadSingle();
            vector.y = br.ReadSingle();
            vector.z = br.ReadSingle();
            return vector;
        }

        public static Vector4 ReadVector4(BinaryReader br)
        {
            Vector4 vector;
            vector.x = br.ReadSingle();
            vector.y = br.ReadSingle();
            vector.z = br.ReadSingle();
            vector.w = br.ReadSingle();
            return vector;
        }

        public static void CreateDir(string dir)
        {
            if (!AssetDatabase.IsValidFolder(dir))
            {
                string name = Path.GetFileNameWithoutExtension(dir);
                dir = Path.GetDirectoryName(dir);
                CreateDir(dir);
                AssetDatabase.CreateFolder(dir, name);
            }
        }

        public static void EnumRootObject(EnumTransform cb, object param = null)
        {
            UnityEngine.SceneManagement.Scene s = SceneManager.GetActiveScene();
            GameObject[] roots = s.GetRootGameObjects();
            for (int i = 0, imax = roots.Length; i < imax; ++i)
            {
                Transform t = roots[i].transform;
                cb(t, param);
            }
        }
        public static void EnumTargetObject(string goPath, EnumTransform cb, object param = null)
        {
            GameObject go = GameObject.Find(goPath);
            if (go != null)
            {
                Transform t = go.transform;
                for (int i = 0, imax = t.childCount; i < imax; ++i)
                {
                    Transform child = t.GetChild(i);
                    cb(child, param);
                }
            }
        }
        public static void EnumChildObject(Transform t, object param, EnumTransform fun)
        {
            for (int i = 0; i < t.childCount; ++i)
            {
                Transform child = t.GetChild(i);
                fun(child, param);
            }
        }

        public static void DestroyChildObjects(GameObject go, string name = "")
        {
            if (go != null)
            {
                Transform t = go.transform;
                if (t.childCount > 0)
                {
                    List<GameObject> children = new List<GameObject>();
                    for (int i = 0; i < t.childCount; ++i)
                    {
                        Transform child = t.GetChild(i);
                        if (string.IsNullOrEmpty(name) || child.name.StartsWith(name))
                            children.Add(child.gameObject);
                    }

                    for (int i = 0; i < children.Count; ++i)
                    {
                        GameObject.DestroyImmediate(children[i]);
                    }
                }
            }
        }

        public static string GetReplaceStr(string str, string repalceStr = ".")
        {
            str = str.Replace(",", repalceStr);
            str = str.Replace("٫", repalceStr);
            return str;
        }

        
        public static int GetSize(this SpriteSize size)
        {
            int powerof2 = (int)size;
            return 1 << powerof2;
        }
        public static bool HasFlag(uint flag,TexFlag f)
        {
            return (flag&(uint)f)!=0;
        }

        public static void SetFlag(ref uint flag, TexFlag f, bool add)
        {
            if (add)
            {
                flag |= (uint) f;
            }
            else
            {
                flag &= ~((uint) f);
            }
        }

    }
}
#endif
