using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace XEngine.Editor
{
    internal class MeshAssets
    {
        internal static Mesh GetScreenMesh(Rect viewPort)
        {
            Mesh mesh = new Mesh { name = "Screen Triangle" };
            mesh.MarkDynamic();

            mesh.SetVertices(new List<Vector3> {
            new Vector3 (viewPort.xMin, viewPort.yMin, 0f),
            new Vector3 (viewPort.xMin, viewPort.yMax, 0f),
            new Vector3 (viewPort.xMax, viewPort.yMax, 0f),
            new Vector3 (viewPort.xMax, viewPort.yMin, 0f),
        });
            mesh.SetIndices(new[] { 0, 1, 3, 1, 2, 3 }, MeshTopology.Triangles, 0, false);
            mesh.SetUVs(0, new List<Vector2> {
            new Vector2 (0, 0),
            new Vector2 (0, 1),
            new Vector2 (1, 1),
            new Vector2 (1, 0)
        });
            return mesh;
        }

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
                s_FullscreenTriangle.SetVertices(new List<Vector3> {
                new Vector3 (-1f, -1f, 0f),
                new Vector3 (-1f, 3f, 0f),
                new Vector3 (3f, -1f, 0f)
            });
                s_FullscreenTriangle.SetIndices(new[] { 0, 1, 2 }, MeshTopology.Triangles, 0, false);
                s_FullscreenTriangle.UploadMeshData(true);

                return s_FullscreenTriangle;
            }
        }
        static Vector3 Average(Vector3[] array, IEnumerable<int> indices)
        {
            Vector3 avg = Vector3.zero;
            int count = 0;

            foreach (int i in indices)
            {
                avg.x += array[i].x;
                avg.y += array[i].y;
                avg.z += array[i].z;

                count++;
            }

            return avg / count;
        }
        static void Cross(float ax, float ay, float az, float bx, float by, float bz, ref float x, ref float y, ref float z)
        {
            x = ay * bz - az * by;
            y = az * bx - ax * bz;
            z = ax * by - ay * bx;
        }
        static Vector3 Normal(Vector3 p0, Vector3 p1, Vector3 p2)
        {
            float ax = p1.x - p0.x,
                ay = p1.y - p0.y,
                az = p1.z - p0.z,
                bx = p2.x - p0.x,
                by = p2.y - p0.y,
                bz = p2.z - p0.z;

            Vector3 cross = Vector3.zero;
            Cross(ax, ay, az, bx, by, bz, ref cross.x, ref cross.y, ref cross.z);
            cross.Normalize();

            if (cross.magnitude < Mathf.Epsilon)
                return new Vector3(0f, 0f, 0f); // bad triangle
            else
                return cross;
        }
        public static void RecalculateNormals(Vector3[] vertices, int[] triangles, Vector3[] normal)
        {
            List<List<int>> smooth = null;

            if (normal != null)
            {
                List<List<int>> common;

                int[] ttmp = new int[vertices.Length];
                for (int i = 0; i < ttmp.Length; ++i)
                    ttmp[i] = i;
                common = ttmp.ToLookup(x => (RndVec3)vertices[x]).Select(y => y.ToList()).ToList();

                smooth = common
                    .SelectMany(x => x.GroupBy(i => (RndVec3)normal[i]))
                    .Where(n => n.Count() > 1)
                    .Select(t => t.ToList())
                    .ToList();

            }
            //calc normal
            Vector3[] perTriangleNormal = new Vector3[vertices.Length];
            int[] perTriangleAvg = new int[vertices.Length];
            int[] tris = triangles;

            for (int i = 0; i < tris.Length; i += 3)
            {
                int a = tris[i], b = tris[i + 1], c = tris[i + 2];

                Vector3 cross = Normal(vertices[a], vertices[b], vertices[c]);

                perTriangleNormal[a].x += cross.x;
                perTriangleNormal[b].x += cross.x;
                perTriangleNormal[c].x += cross.x;

                perTriangleNormal[a].y += cross.y;
                perTriangleNormal[b].y += cross.y;
                perTriangleNormal[c].y += cross.y;

                perTriangleNormal[a].z += cross.z;
                perTriangleNormal[b].z += cross.z;
                perTriangleNormal[c].z += cross.z;

                perTriangleAvg[a]++;
                perTriangleAvg[b]++;
                perTriangleAvg[c]++;
            }

            for (int i = 0; i < vertices.Length; i++)
            {
                normal[i].x = perTriangleNormal[i].x * (float)perTriangleAvg[i];
                normal[i].y = perTriangleNormal[i].y * (float)perTriangleAvg[i];
                normal[i].z = perTriangleNormal[i].z * (float)perTriangleAvg[i];
            }

            if (smooth != null)
            {
                foreach (List<int> l in smooth)
                {
                    Vector3 n = Average(normal, l);

                    foreach (int i in l)
                        normal[i] = n;
                }
            }
        }

        public static Vector4[] SolveTangent(Vector3[] vertices, int[] triangles, Vector3[] normals)
        {
            int triangleCount = triangles.Length / 3;
            int vertexCount = vertices.Length;

            Vector3[] tan1 = new Vector3[vertexCount];
            Vector3[] tan2 = new Vector3[vertexCount];
            Vector4[] tangents = new Vector4[vertexCount];
            for (long a = 0; a < triangleCount; a += 3)
            {
                long i1 = triangles[a + 0];
                long i2 = triangles[a + 1];
                long i3 = triangles[a + 2];
                Vector3 v1 = vertices[i1];
                Vector3 v2 = vertices[i2];
                Vector3 v3 = vertices[i3];

                Vector2 w1 = new Vector2(v1.x, v1.z);
                Vector2 w2 = new Vector2(v2.x, v2.z);
                Vector2 w3 = new Vector2(v3.x, v3.z);

                float x1 = v2.x - v1.x;
                float x2 = v3.x - v1.x;
                float y1 = v2.y - v1.y;
                float y2 = v3.y - v1.y;
                float z1 = v2.z - v1.z;
                float z2 = v3.z - v1.z;
                float s1 = w2.x - w1.x;
                float s2 = w3.x - w1.x;
                float t1 = w2.y - w1.y;
                float t2 = w3.y - w1.y;
                float r = 1.0f / (s1 * t2 - s2 * t1);
                Vector3 sdir = new Vector3((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
                Vector3 tdir = new Vector3((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);
                tan1[i1] += sdir;
                tan1[i2] += sdir;
                tan1[i3] += sdir;
                tan2[i1] += tdir;
                tan2[i2] += tdir;
                tan2[i3] += tdir;
            }
            for (long a = 0; a < vertexCount; ++a)
            {
                Vector3 n = normals[a];
                Vector3 t = tan1[a];
                Vector3 tmp = (t - n * Vector3.Dot(n, t)).normalized;
                tangents[a] = new Vector4(tmp.x, tmp.y, tmp.z);
                tangents[a].w = (Vector3.Dot(Vector3.Cross(n, t), tan2[a]) < 0.0f) ? -1.0f : 1.0f;
            }
            return tangents;
        }

        internal static void MakeMakeReadable(string path)
        {
            Mesh mesh = AssetDatabase.LoadAssetAtPath<Mesh>(path);
            if (mesh != null)
            {
                mesh.UploadMeshData(false);
                SerializedProperty sp = CommonAssets.GetSerializeProperty(mesh, "m_IsReadable");
                if (sp != null)
                {
                    sp.boolValue = true;
                    sp.serializedObject.ApplyModifiedProperties();
                }
            }
        }

        internal static Bounds CalcBounds(Mesh m, Matrix4x4 matrix, out int vertexCount, out int indexCount)
        {
            vertexCount = m.vertexCount;
            indexCount = (int)m.GetIndexCount(0);
            Vector3[] vertices = m.vertices;
            Bounds bound = new Bounds();
            for (int i = 0; i < vertices.Length; ++i)
            {
                Vector3 vertex = vertices[i];
                Vector3 worldPos = matrix.MultiplyPoint(vertex);
                if (i == 0)
                {
                    bound.center = worldPos;
                }
                else
                {
                    bound.Encapsulate(worldPos);
                }
            }
            return bound;
        }

    }
}