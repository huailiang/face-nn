using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using System.Linq.Expressions;
using System.Reflection;
using System.Text;
using UnityEditor;
using System.IO;

namespace XEngine.Editor
{
    public static class EditorUtility
    {
        static Dictionary<string, GUIContent> s_GUIContentCache;
        static Dictionary<Type, AttributeDecorator> s_AttributeDecorators;
        static IEnumerable<Type> m_AssemblyTypes;

        public static string basepath
        {
            get
            {
                string path = Application.dataPath;
                path = path.Remove(path.IndexOf("/Assets"));
                return path;
            }
        }

        static EditorUtility()
        {
            s_GUIContentCache = new Dictionary<string, GUIContent>();
            s_AttributeDecorators = new Dictionary<Type, AttributeDecorator>();
            ReloadDecoratorTypes();
        }

        [UnityEditor.Callbacks.DidReloadScripts]
        static void OnEditorReload()
        {
            ReloadDecoratorTypes();
        }

        public static IEnumerable<Type> GetAllAssemblyTypes()
        {
            if (m_AssemblyTypes == null)
            {
                m_AssemblyTypes = AppDomain.CurrentDomain.GetAssemblies().SelectMany(t => t.GetTypes());
            }
            return m_AssemblyTypes;
        }

        public static T GetAttribute<T>(this Type type) where T : Attribute
        {
            Assert.IsTrue(type.IsDefined(typeof(T), false), "Attribute not found");
            return (T)type.GetCustomAttributes(typeof(T), false)[0];
        }

        public static Attribute[] GetMemberAttributes<TType, TValue>(Expression<Func<TType, TValue>> expr)
        {
            Expression body = expr;

            if (body is LambdaExpression)
                body = ((LambdaExpression)body).Body;

            switch (body.NodeType)
            {
                case ExpressionType.MemberAccess:
                    var fi = (FieldInfo)((MemberExpression)body).Member;
                    return fi.GetCustomAttributes(false).Cast<Attribute>().ToArray();
                default:
                    throw new InvalidOperationException();
            }
        }

        public static string GetFieldPath<TType, TValue>(Expression<Func<TType, TValue>> expr)
        {
            MemberExpression me;
            switch (expr.Body.NodeType)
            {
                case ExpressionType.MemberAccess:
                    me = expr.Body as MemberExpression;
                    break;
                default:
                    throw new InvalidOperationException();
            }

            var members = new List<string>();
            while (me != null)
            {
                members.Add(me.Member.Name);
                me = me.Expression as MemberExpression;
            }

            var sb = new StringBuilder();
            for (int i = members.Count - 1; i >= 0; i--)
            {
                sb.Append(members[i]);
                if (i > 0) sb.Append('.');
            }

            return sb.ToString();
        }

        public static object GetParentObject(string path, object obj)
        {
            var fields = path.Split('.');

            if (fields.Length == 1)
                return obj;

            var info = obj.GetType().GetField(fields[0], BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            obj = info.GetValue(obj);
            return GetParentObject(string.Join(".", fields, 1, fields.Length - 1), obj);
        }

        static void ReloadDecoratorTypes()
        {
            s_AttributeDecorators.Clear();

            var types = GetAllAssemblyTypes()
                            .Where(
                                t => t.IsSubclassOf(typeof(AttributeDecorator))
                                  && t.IsDefined(typeof(DecoratorAttribute), false)
                                  && !t.IsAbstract
                            );

            foreach (var type in types)
            {
                var attr = type.GetAttribute<DecoratorAttribute>();
                var decorator = (AttributeDecorator)Activator.CreateInstance(type);
                s_AttributeDecorators.Add(attr.attributeType, decorator);
            }
        }

        internal static AttributeDecorator GetDecorator(Type attributeType)
        {
            AttributeDecorator decorator;
            return !s_AttributeDecorators.TryGetValue(attributeType, out decorator)
                ? null : decorator;
        }

        public static GUIContent GetContent(string textAndTooltip)
        {
            if (string.IsNullOrEmpty(textAndTooltip))
                return GUIContent.none;

            GUIContent content;

            if (!s_GUIContentCache.TryGetValue(textAndTooltip, out content))
            {
                var s = textAndTooltip.Split('|');
                content = new GUIContent(s[0]);

                if (s.Length > 1 && !string.IsNullOrEmpty(s[1]))
                    content.tooltip = s[1];

                s_GUIContentCache.Add(textAndTooltip, content);
            }
            return content;
        }

        [MenuItem("Help/IO/OpenCacheDiectory")]
        public static void OpenCacheDiectory()
        {
            Open(Application.temporaryCachePath);
        }

        [MenuItem("Help/IO/OpenPersistDirectory")]
        public static void OpenPersistDirectory()
        {
            Open(Application.persistentDataPath);
        }

        [MenuItem("Help/IO/OpenShellDirectory")]
        public static void OpenAssetbundle()
        {
            Open(basepath + "/Shell");
        }

        [MenuItem("Help/IO/OpenUnityInstallDirectory")]
        public static void OpenUnityDir()
        {
            Open(EditorApplication.applicationContentsPath);
        }

        [MenuItem("Help/RestartUnity")]
        public static void RestartUnity()
        {
#if UNITY_EDITOR_WIN
            string install = Path.GetDirectoryName(EditorApplication.applicationContentsPath);
            string path = Path.Combine(install, "Unity.exe");
            string[] args = path.Split('\\');
            System.Diagnostics.Process po = new System.Diagnostics.Process();
            Debug.Log("install: " + install + " path: " + path);
            po.StartInfo.FileName = path;
            po.Start();

            System.Diagnostics.Process[] pro = System.Diagnostics.Process.GetProcessesByName(args[args.Length - 1].Split('.')[0]);//Unity
            foreach (var item in pro)
            {
                item.Kill();
            }
#endif
        }

        public static void Open(string path)
        {
            if (File.Exists(path))
            {
                path = Path.GetDirectoryName(path);
            }
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
            }
#if UNITY_EDITOR_OSX
            string shell = basepath + "/Shell/open.sh";
            string arg = path;
            string ex = shell + " " + arg;
            System.Diagnostics.Process.Start("/bin/bash", ex);
#elif UNITY_EDITOR_WIN
            path = path.Replace("/", "\\");
            System.Diagnostics.Process.Start("explorer.exe", path);
#endif
        }
    }
}
