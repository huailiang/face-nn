using System;
using UnityEngine;

namespace CFEngine.Editor
{
    public abstract class AttributeDecorator
    {
        // Override this and return false if you want to customize the override checkbox position,
        // else it'll automatically draw it and put the property content in a horizontal scope.
        public virtual bool IsAutoProperty()
        {
            return true;
        }

        public abstract bool OnGUI(SerializedParameter serializedParameterOverride, GUIContent title, Attribute attribute);
    }
}