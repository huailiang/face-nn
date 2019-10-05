using System;
using UnityEngine;
using UnityEditor;
using System.Linq;

namespace XEngine.Editor
{

    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
    public sealed class DecoratorAttribute : Attribute
    {
        public readonly Type attributeType;

        public DecoratorAttribute(Type attributeType)
        {
            this.attributeType = attributeType;
        }
    }


    public abstract class AttributeDecorator
    {
        public virtual bool IsAutoProperty()
        {
            return true;
        }

        public abstract bool OnGUI(SerializedParameter serializedParameterOverride, GUIContent title, Attribute attribute);
    }

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