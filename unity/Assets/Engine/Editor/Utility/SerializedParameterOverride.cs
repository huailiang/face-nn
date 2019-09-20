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


    public class SerializedParameterOverride : SerializedParameter
    {
        public SerializedProperty overrideState { get; private set; }

        internal SerializedParameterOverride(SerializedProperty property, Attribute[] attributes)
            : base()
        {
            baseProperty = property.Copy();

            var localCopy = baseProperty.Copy();
            localCopy.Next(true);
            overrideState = localCopy.Copy();
            localCopy.Next(false);
            value = localCopy.Copy();

            this.attributes = attributes;
        }
    }


    public delegate void ButtonCallback();

    public class ResSerializedParameterOverride : SerializedParameterOverride
    {
        public ResPathParameter rawParam { get; set; }

        public ButtonCallback onButton { get; set; }
        public ButtonCallback onButton2 { get; set; }
        public ButtonCallback onButton3 { get; set; }
        internal ResSerializedParameterOverride(SerializedProperty property, Attribute[] attributes)
            : base(property, attributes)
        {

        }
    }
}