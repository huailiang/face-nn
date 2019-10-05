using System;
using System.Linq;
using UnityEditor;

namespace XEngine.Editor
{
    public class SerializedParameter
    {
        public SerializedProperty value { get; protected set; }
        public Attribute[] attributes { get; protected set; }

        internal SerializedProperty baseProperty;

        public string displayName
        {
            get { return baseProperty.displayName; }
        }
        internal SerializedParameter() { }

        internal SerializedParameter(SerializedProperty property, Attribute[] attributes)
        {
            baseProperty = property.Copy();
            value = baseProperty.Copy();
            this.attributes = attributes;
        }

        public T GetAttribute<T>()
            where T : Attribute
        {
            return (T)attributes.FirstOrDefault(x => x is T);
        }
    }

}