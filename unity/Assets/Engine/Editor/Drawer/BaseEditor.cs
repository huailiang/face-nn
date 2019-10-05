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
        protected T m_Target { get { return (T)target; } }

        protected SerializedProperty FindProperty<TValue>(Expression<Func<T, TValue>> expr)
        {
            return serializedObject.FindProperty(EditorUtility.GetFieldPath(expr));
        }

        protected SerializedParameter FindParameter<TValue>(Expression<Func<T, TValue>> expr)
        {
            var property = serializedObject.FindProperty(EditorUtility.GetFieldPath(expr));
            var attributes = EditorUtility.GetMemberAttributes(expr);
            return new SerializedParameter(property, attributes);
        }

        protected void PropertyField(SerializedParameter property)
        {
            var title = EditorUtility.GetContent(property.displayName);
            PropertyField(property, title);
        }

        protected void PropertyField(SerializedParameter property, GUIContent title)
        {
            var displayNameAttr = property.GetAttribute<DisplayNameAttribute>();
            if (displayNameAttr != null)
                title.text = displayNameAttr.displayName;
            if (string.IsNullOrEmpty(title.tooltip))
            {
                var tooltipAttr = property.GetAttribute<TooltipAttribute>();
                if (tooltipAttr != null)
                    title.tooltip = tooltipAttr.tooltip;
            }

            AttributeDecorator decorator = null;
            Attribute attribute = null;
            foreach (var attr in property.attributes)
            {
                if (decorator == null)
                {
                    decorator = EditorUtility.GetDecorator(attr.GetType());
                    attribute = attr;
                }
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
                        EditorGUI.LabelField(rect, (attr as HeaderAttribute).header, EditorStyles.miniLabel);
                    }
                }
            }

            bool invalidProp = false;

            if (decorator != null && !decorator.IsAutoProperty())
            {
                if (decorator.OnGUI(property, title, attribute))
                    return;

                invalidProp = true;
            }

            using (new EditorGUILayout.HorizontalScope())
            {
                {
                    if (decorator != null && !invalidProp)
                    {
                        if (decorator.OnGUI(property, title, attribute))
                            return;
                    }
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
