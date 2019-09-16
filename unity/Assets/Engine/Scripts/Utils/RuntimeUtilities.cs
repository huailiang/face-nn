using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

#if UNITY_EDITOR
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using UnityEditor;
#endif

namespace XEngine
{
    using UnityEngine.SceneManagement;
    using UnityObject = UnityEngine.Object;

    public static class RuntimeUtilities
    {
        #region Textures        

#if UNITY_EDITOR
        static Texture2D m_TransparentTexture;
        public static Texture2D transparentTexture
        {
            get
            {
                if (m_TransparentTexture == null)
                {
                    m_TransparentTexture = new Texture2D (1, 1, TextureFormat.ARGB32, false);
                    m_TransparentTexture.SetPixel (0, 0, Color.clear);
                    m_TransparentTexture.Apply ();
                }

                return m_TransparentTexture;
            }
        }
#endif

        #endregion

        #region Rendering

        static Mesh s_FullscreenTriangle;
        public static Mesh fullscreenTriangle
        {
            get
            {
                if (s_FullscreenTriangle != null)
                    return s_FullscreenTriangle;

                s_FullscreenTriangle = new Mesh { name = "Fullscreen Triangle" };

                // Because we have to support older platforms (GLES2/3, DX9 etc) we can't do all of
                // this directly in the vertex shader using vertex ids :(
                s_FullscreenTriangle.SetVertices(new List<Vector3>
                {
                    new Vector3 (-1f, -1f, 0f),
                    new Vector3 (-1f, 3f, 0f),
                    new Vector3 (3f, -1f, 0f)
                });
                s_FullscreenTriangle.SetIndices(new[] { 0, 1, 2 }, MeshTopology.Triangles, 0, false);
                s_FullscreenTriangle.UploadMeshData(true);

                return s_FullscreenTriangle;
            }
        }

        static Material s_CopyMaterial;
        public static Material GetCopyMaterial(Shader shader)
        {
            if (s_CopyMaterial != null)
                return s_CopyMaterial;

            s_CopyMaterial = new Material(shader)
            {
                name = "PostProcess - Copy",
                hideFlags = HideFlags.HideAndDontSave
            };

            return s_CopyMaterial;
        }


        // Use a custom blit method to draw a fullscreen triangle instead of a fullscreen quad
        // https://michaldrobot.com/2014/04/01/gcn-execution-patterns-in-full-screen-passes/
        public static void BlitFullscreenTriangle(this CommandBuffer cmd, ref RenderTargetIdentifier source, ref RenderTargetIdentifier destination, Shader shader, bool clear = false)
        {
            cmd.SetGlobalTexture(ShaderIDs.MainTex, source);
            cmd.SetRenderTarget(destination);

            if (clear)
                cmd.ClearRenderTarget(true, true, Color.clear);

            cmd.DrawMesh(fullscreenTriangle, Matrix4x4.identity, GetCopyMaterial(shader), 0, 0);
        }

        public static void BlitFullscreenTriangle(this CommandBuffer cmd, ref RenderTargetIdentifier source, ref RenderTargetIdentifier destination, PropertySheet propertySheet, int pass, bool clear = false)
        {
            cmd.SetGlobalTexture(ShaderIDs.MainTex, source);
            cmd.SetRenderTarget(destination);

            if (clear)
                cmd.ClearRenderTarget(true, true, Color.clear);

            cmd.DrawMesh(fullscreenTriangle, Matrix4x4.identity, propertySheet.material, 0, pass, propertySheet.properties);
        }
        public static void BlitFullscreenTriangle(this CommandBuffer cmd, RenderTexture source, RenderTexture destination, Shader shader, bool clear = false)
        {
            cmd.SetGlobalTexture(ShaderIDs.MainTex, source);
            cmd.SetRenderTarget(destination);

            if (clear)
                cmd.ClearRenderTarget(true, true, Color.clear);

            cmd.DrawMesh(fullscreenTriangle, Matrix4x4.identity, GetCopyMaterial(shader), 0, 0);
        }

        public static void BlitFullscreenTriangle(this CommandBuffer cmd, RenderTexture source, RenderTexture destination, PropertySheet propertySheet, int pass, bool clear = false)
        {
            if (source != null)
                cmd.SetGlobalTexture(ShaderIDs.MainTex, source);
            cmd.SetRenderTarget(destination);

            if (clear)
                cmd.ClearRenderTarget(true, true, Color.clear);

            cmd.DrawMesh(fullscreenTriangle, Matrix4x4.identity, propertySheet.material, 0, pass, propertySheet.properties);
        }

        public static void BlitFullscreenTriangle(this CommandBuffer cmd, ref RenderTargetIdentifier source, RenderTexture destination, PropertySheet propertySheet, int pass, bool clear = false)
        {
            cmd.SetGlobalTexture(ShaderIDs.MainTex, source);
            cmd.SetRenderTarget(destination);

            if (clear)
                cmd.ClearRenderTarget(true, true, Color.clear);

            cmd.DrawMesh(fullscreenTriangle, Matrix4x4.identity, propertySheet.material, 0, pass, propertySheet.properties);
        }

        public static void BlitDepth(this CommandBuffer cmd, RenderTexture source, RenderTexture destination, Shader shader, int pass = 0)
        {
            cmd.SetGlobalTexture(ShaderIDs.MainTex, source);
            cmd.SetRenderTarget(destination, BuiltinRenderTextureType.None);


            cmd.DrawMesh(fullscreenTriangle, Matrix4x4.identity, GetCopyMaterial(shader), 0, pass);
        }
        public static void EnableKeyword(string keyword, bool isEnable)
        {
            if (isEnable)
            {
                Shader.EnableKeyword(keyword);
            }
            else
            {
                Shader.DisableKeyword(keyword);
            }
        }

        #endregion

        #region Unity specifics & misc methods

        public static bool scriptableRenderPipelineActive
        {
            get { return GraphicsSettings.renderPipelineAsset != null; } // 5.6+ only
        }

#if UNITY_EDITOR
        public static bool isSinglePassStereoSelected
        {
            get
            {
                return UnityEditor.PlayerSettings.virtualRealitySupported &&
                    UnityEditor.PlayerSettings.stereoRenderingPath == UnityEditor.StereoRenderingPath.SinglePass;
            }
        }
#endif


        public static RenderTextureFormat defaultHDRRenderTextureFormat
        {
            get
            {
#if UNITY_ANDROID || UNITY_IPHONE || UNITY_TVOS || UNITY_SWITCH || UNITY_EDITOR
                RenderTextureFormat format = RenderTextureFormat.RGB111110Float;
#if UNITY_EDITOR
                var target = EditorUserBuildSettings.activeBuildTarget;
                if (target != BuildTarget.Android && target != BuildTarget.iOS && target != BuildTarget.tvOS && target != BuildTarget.Switch)
                    return RenderTextureFormat.DefaultHDR;
#endif // UNITY_EDITOR
                if (format.IsSupported ())
                    return format;
#endif // UNITY_ANDROID || UNITY_IPHONE || UNITY_TVOS || UNITY_SWITCH || UNITY_EDITOR
                return RenderTextureFormat.DefaultHDR;
            }
        }
        public static bool isLinearColorSpace
        {
            get { return QualitySettings.activeColorSpace == ColorSpace.Linear; }
        }

        public static bool isFloatingPointFormat(RenderTextureFormat format)
        {
            return format == RenderTextureFormat.DefaultHDR || format == RenderTextureFormat.ARGBHalf || format == RenderTextureFormat.ARGBFloat ||
                format == RenderTextureFormat.RGFloat || format == RenderTextureFormat.RGHalf ||
                format == RenderTextureFormat.RFloat || format == RenderTextureFormat.RHalf ||
                format == RenderTextureFormat.RGB111110Float;
        }

        public static void Destroy(UnityObject obj)
        {
            if (obj != null)
            {
#if UNITY_EDITOR
                if (Application.isPlaying)
                    UnityObject.Destroy (obj);
                else
                    UnityObject.DestroyImmediate (obj);
#else
                UnityObject.Destroy(obj);
#endif
            }
        }


#if UNITY_EDITOR
        // Returns ALL scene objects in the hierarchy, included inactive objects
        // Beware, this method will be slow for big scenes
        public static IEnumerable<T> GetAllSceneObjects<T> ()
        where T : Component
        {
            var queue = new Queue<Transform> ();
            var roots = SceneManager.GetActiveScene ().GetRootGameObjects ();

            foreach (var root in roots)
            {
                queue.Enqueue (root.transform);
                var comp = root.GetComponent<T> ();

                if (comp != null)
                    yield return comp;
            }

            while (queue.Count > 0)
            {
                foreach (Transform child in queue.Dequeue ())
                {
                    queue.Enqueue (child);
                    var comp = child.GetComponent<T> ();

                    if (comp != null)
                        yield return comp;
                }
            }
        }
#endif

        public static void CreateIfNull<T>(ref T obj)
        where T : class, new()
        {
            if (obj == null)
                obj = new T();
        }

        #endregion

        #region Reflection

#if UNITY_EDITOR
        static IEnumerable<Type> m_AssemblyTypes;

        public static IEnumerable<Type> GetAllAssemblyTypes ()
        {
            if (m_AssemblyTypes == null)
            {
                m_AssemblyTypes = AppDomain.CurrentDomain.GetAssemblies ()
                    .SelectMany (t =>
                    {
                        // Ugly hack to handle mis-versioned dlls
                        var innerTypes = new Type[0];
                        try
                        {
                            innerTypes = t.GetTypes ();
                        }
                        catch { }
                        return innerTypes;
                    });
            }

            return m_AssemblyTypes;
        }

        // Quick extension method to get the first attribute of type T on a given Type
        public static T GetAttribute<T> (this Type type) where T : Attribute
        {
            Assert.IsTrue (type.IsDefined (typeof (T), false), "Attribute not found");
            return (T) type.GetCustomAttributes (typeof (T), false) [0];
        }

        // Returns all attributes set on a specific member
        // Note: doesn't include inherited attributes, only explicit ones
        public static Attribute[] GetMemberAttributes<TType, TValue> (Expression<Func<TType, TValue>> expr)
        {
            Expression body = expr;

            if (body is LambdaExpression)
                body = ((LambdaExpression) body).Body;

            switch (body.NodeType)
            {
                case ExpressionType.MemberAccess:
                    var fi = (FieldInfo) ((MemberExpression) body).Member;
                    return fi.GetCustomAttributes (false).Cast<Attribute> ().ToArray ();
                default:
                    throw new InvalidOperationException ();
            }
        }

        // Returns a string path from an expression - mostly used to retrieve serialized properties
        // without hardcoding the field path. Safer, and allows for proper refactoring.
        public static string GetFieldPath<TType, TValue> (Expression<Func<TType, TValue>> expr)
        {
            MemberExpression me;
            switch (expr.Body.NodeType)
            {
                case ExpressionType.MemberAccess:
                    me = expr.Body as MemberExpression;
                    break;
                default:
                    throw new InvalidOperationException ();
            }

            var members = new List<string> ();
            while (me != null)
            {
                members.Add (me.Member.Name);
                me = me.Expression as MemberExpression;
            }

            var sb = new StringBuilder ();
            for (int i = members.Count - 1; i >= 0; i--)
            {
                sb.Append (members[i]);
                if (i > 0) sb.Append ('.');
            }

            return sb.ToString ();
        }

        public static object GetParentObject (string path, object obj)
        {
            var fields = path.Split ('.');

            if (fields.Length == 1)
                return obj;

            var info = obj.GetType ().GetField (fields[0], BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            obj = info.GetValue (obj);

            return GetParentObject (string.Join (".", fields, 1, fields.Length - 1), obj);
        }
#endif
        #endregion
    }
}