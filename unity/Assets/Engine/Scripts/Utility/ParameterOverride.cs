using System;
using CFUtilPoolLib;
using UnityEngine;

namespace XEngine
{
    public abstract class ParameterOverride : IParameterOverride
    {
        public bool overrideState;

        public abstract int GetHash();

        public T GetValue<T>()
        {
            return ((ParameterOverride<T>)this).value;
        }

        protected internal virtual void OnEnable() { }

        protected internal virtual void OnDisable() { }

        internal abstract void SetValue(ParameterOverride parameter);

        public virtual void SetValue(UnityEngine.Vector4 value) { }

        public virtual void Interp(ParameterOverride from, float to, float t) { }

        public virtual void Interp(float from, ParameterOverride to, float t) { }

        public virtual void Interp(ParameterOverride from, Vector4 to, float t) { }

        public virtual void Interp(Vector4 from, ParameterOverride to, float t) { }

        public virtual void Interp(ParameterOverride from, Enum to, float t) { }

        public virtual void Interp(Enum from, ParameterOverride to, float t) { }

        public void SetOverride(bool overrided)
        {
            overrideState = overrided;
        }
    }

    [Serializable]
    public class ParameterOverride<T> : ParameterOverride
    {
        public T value;

        public ParameterOverride() : this(default(T), false) { }

        public ParameterOverride(T value) : this(value, false) { }

        public ParameterOverride(T value, bool overrideState)
        {
            this.value = value;
            this.overrideState = overrideState;
        }

        public void Override(T x)
        {
            overrideState = true;
            value = x;
        }

        internal override void SetValue(ParameterOverride parameter)
        {
            value = parameter.GetValue<T>();
        }

        public override int GetHash()
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 23 + overrideState.GetHashCode();
                hash = hash * 23 + value.GetHashCode();
                return hash;
            }
        }

        public static implicit operator T(ParameterOverride<T> prop)
        {
            return prop.value;
        }
    }

    [Serializable]
    public sealed class FloatParameter : ParameterOverride<float>
    {

        public override void SetValue(UnityEngine.Vector4 value)
        {
            this.value = value.x;
        }

        public override void Interp(ParameterOverride from, float to, float t)
        {
            value = from.GetValue<float>() + (to - from.GetValue<float>()) * t;
        }

        public override void Interp(float from, ParameterOverride to, float t)
        {
            value = from + (to.GetValue<float>() - from) * t;
        }
    }

    [Serializable]
    public sealed class IntParameter : ParameterOverride<int>
    {
        public override void SetValue(UnityEngine.Vector4 value)
        {
            this.value = (int)value.x;
        }
    }

    [Serializable]
    public sealed class BoolParameter : ParameterOverride<bool>
    {
        public override void SetValue(UnityEngine.Vector4 value)
        {
            this.value = value.x > 0.0f;
        }
        public override void Interp(ParameterOverride from, float to, float t)
        {
            value = to > 0;
        }
        public override void Interp(float from, ParameterOverride to, float t)
        {
            value = from > 0;
        }
    }

    [Serializable]
    public sealed class ColorParameter : ParameterOverride<Color>
    {
        public override void SetValue(UnityEngine.Vector4 value)
        {
            this.value = new Color(value.x, value.y, value.z, value.w);
        }

        public override void Interp(ParameterOverride from, Vector4 to, float t)
        {
            value = Color.Lerp(from.GetValue<Color>(), to, t);
        }
        public override void Interp(Vector4 from, ParameterOverride to, float t)
        {
            value = Color.Lerp(from, to.GetValue<Color>(), t);
        }
    }

    [Serializable]
    public sealed class Vector2Parameter : ParameterOverride<Vector2>
    {
        public override void SetValue(UnityEngine.Vector4 value)
        {
            this.value = new Vector2(value.x, value.y);
        }
    }

    [Serializable]
    public sealed class Vector3Parameter : ParameterOverride<Vector3>
    {
        public override void SetValue(UnityEngine.Vector4 value)
        {
            this.value = new Vector3(value.x, value.y, value.w);
        }
    }

    [Serializable]
    public sealed class Vector4Parameter : ParameterOverride<Vector4>
    {
        public override void SetValue(UnityEngine.Vector4 value)
        {
            this.value = value;
        }
        public override void Interp(ParameterOverride from, Vector4 to, float t)
        {
            value = Vector4.Lerp(from.GetValue<Vector4>(), to, t);
        }
        public override void Interp(Vector4 from, ParameterOverride to, float t)
        {
            value = Vector4.Lerp(from, to.GetValue<Vector4>(), t);
        }
    }


    [Serializable]
    public sealed class ResPathParameter : ParameterOverride<string>
    {
        [NonSerialized]
        public UnityEngine.Object asset = null;

        public T GetAsset<T>() where T : UnityEngine.Object
        {
            return (T)asset;
        }
    }
}
