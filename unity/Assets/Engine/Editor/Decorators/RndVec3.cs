using UnityEngine;
namespace XEngine.Editor
{
    public struct RndVec3 : System.IEquatable<RndVec3>
    {
        public float x;
        public float y;
        public float z;

        const float resolution = .0001f;

        public RndVec3(Vector3 vector)
        {
            this.x = vector.x;
            this.y = vector.y;
            this.z = vector.z;
        }

        public bool Equals(RndVec3 p)
        {
            return Mathf.Abs(x - p.x) < resolution &&
                    Mathf.Abs(y - p.y) < resolution &&
                    Mathf.Abs(z - p.z) < resolution;
        }

        public bool Equals(Vector3 p)
        {
            return Mathf.Abs(x - p.x) < resolution &&
                    Mathf.Abs(y - p.y) < resolution &&
                    Mathf.Abs(z - p.z) < resolution;
        }

        public override bool Equals(System.Object b)
        {
            return (b is RndVec3 && (this.Equals((RndVec3)b))) ||
                    (b is Vector3 && this.Equals((Vector3)b));
        }

        public override int GetHashCode()
        {
            int hash = 27;

            unchecked
            {
                hash = hash * 29 + round(x);
                hash = hash * 29 + round(y);
                hash = hash * 29 + round(z);
            }

            return hash;
        }

        public override string ToString()
        {
            return string.Format("{{{0:F2}, {1:F2}, {2:F2}}}",
                x, y, z);
        }

        private int round(float v)
        {
            return (int)(v / resolution);
        }

        public static implicit operator Vector3(RndVec3 p)
        {
            return new Vector3(p.x, p.y, p.z);
        }

        public static implicit operator RndVec3(Vector3 p)
        {
            return new RndVec3(p);
        }
    }
}