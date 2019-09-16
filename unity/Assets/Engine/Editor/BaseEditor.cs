using System;
using System.Linq.Expressions;
using UnityEngine;
using UnityEditor;
using UnityEngineEditor = UnityEditor.Editor;

namespace XEngine.Editor
{
    public class BaseEditor<T> : UnityEngineEditor
        where T : UnityEngine.Object
    {
        protected T m_Target
        {
            get { return (T)target; }
        }

        protected SerializedProperty FindProperty<TValue>(Expression<Func<T, TValue>> expr)
        {
            return serializedObject.FindProperty(RuntimeUtilities.GetFieldPath(expr));
        }

        protected SerializedParameter FindParameter<TValue>(Expression<Func<T, TValue>> expr)
        {
            var property = serializedObject.FindProperty(RuntimeUtilities.GetFieldPath(expr));
            var attributes = RuntimeUtilities.GetMemberAttributes(expr);
            return new SerializedParameter(property, attributes);
        }

        protected void PropertyField(SerializedParameter property)
        {
            var title = EditorUtilities.GetContent(property.displayName);
            PropertyField(property, title);
        }



        protected void PropertyField(SerializedParameter property, GUIContent title)
        {
            // Check for DisplayNameAttribute first
            var displayNameAttr = property.GetAttribute<CFDisplayNameAttribute>();
            if (displayNameAttr != null)
                title.text = displayNameAttr.displayName;

            // Add tooltip if it's missing and an attribute is available
            if (string.IsNullOrEmpty(title.tooltip))
            {
                var tooltipAttr = property.GetAttribute<TooltipAttribute>();
                if (tooltipAttr != null)
                    title.tooltip = tooltipAttr.tooltip;
            }

            // Look for a compatible attribute decorator
            AttributeDecorator decorator = null;
            Attribute attribute = null;

            foreach (var attr in property.attributes)
            {
                // Use the first decorator we found
                if (decorator == null)
                {
                    decorator = EditorUtilities.GetDecorator(attr.GetType());
                    attribute = attr;
                }

                // Draw unity built-in Decorators (Space, Header)
                if (attr is PropertyAttribute)
                {
                    if (attr is SpaceAttribute)
                    {
                        EditorGUILayout.GetControlRect(false, (attr as SpaceAttribute).height);
                    }
                    else if (attr is HeaderAttribute)
                    {
                        var rect = EditorGUILayout.GetControlRect(false, 24f);
                        rect.y += 8f;
                        rect = EditorGUI.IndentedRect(rect);
                        EditorGUI.LabelField(rect, (attr as HeaderAttribute).header, Styling.labelHeader);
                    }
                }
            }

            bool invalidProp = false;

            if (decorator != null && !decorator.IsAutoProperty())
            {
                if (decorator.OnGUI(property, title, attribute))
                    return;

                // Attribute is invalid for the specified property; use default unity field instead
                invalidProp = true;
            }

            using (new EditorGUILayout.HorizontalScope())
            {
                // Property
                // using (new EditorGUI.DisabledScope(!property.overrideState.boolValue))
                {
                    if (decorator != null && !invalidProp)
                    {
                        if (decorator.OnGUI(property, title, attribute))
                            return;
                    }

                    // Default unity field
                    if (property.value.hasVisibleChildren
                        && property.value.propertyType != SerializedPropertyType.Vector2
                        && property.value.propertyType != SerializedPropertyType.Vector3)
                    {
                        GUILayout.Space(12f);
                        EditorGUILayout.PropertyField(property.value, title, true);
                    }
                    else
                    {
                        EditorGUILayout.PropertyField(property.value, title);
                    }
                }
            }
        }
    }
}
