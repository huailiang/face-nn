using System;
using UnityEditor;
using UnityEngine;

namespace CFEngine.Editor
{
    [CFDecorator (typeof (CFRangeAttribute))]
    public sealed class RangeDecorator : AttributeDecorator
    {
        public override bool OnGUI (SerializedParameter serializedParameterOverride, GUIContent title, Attribute attribute)
        {
            var attr = (CFRangeAttribute) attribute;
            var property = serializedParameterOverride.value;
            if (property.propertyType == SerializedPropertyType.Float)
            {
                property.floatValue = EditorGUILayout.Slider (title, property.floatValue, attr.min, attr.max);
                return true;
            }

            if (property.propertyType == SerializedPropertyType.Integer)
            {
                property.intValue = EditorGUILayout.IntSlider (title, property.intValue, (int) attr.min, (int) attr.max);
                return true;
            }

            return false;
        }
    }

    [CFDecorator (typeof (CFMinAttribute))]
    public sealed class MinDecorator : AttributeDecorator
    {
        public override bool OnGUI (SerializedParameter serializedParameterOverride, GUIContent title, Attribute attribute)
        {
            var attr = (CFMinAttribute) attribute;
            var property = serializedParameterOverride.value;
            if (property.propertyType == SerializedPropertyType.Float)
            {
                float v = EditorGUILayout.FloatField (title, property.floatValue);
                property.floatValue = Mathf.Max (v, attr.min);
                return true;
            }

            if (property.propertyType == SerializedPropertyType.Integer)
            {
                int v = EditorGUILayout.IntField (title, property.intValue);
                property.intValue = Mathf.Max (v, (int) attr.min);
                return true;
            }

            return false;
        }
    }

    [CFDecorator (typeof (CFMaxAttribute))]
    public sealed class MaxDecorator : AttributeDecorator
    {
        public override bool OnGUI (SerializedParameter serializedParameterOverride, GUIContent title, Attribute attribute)
        {
            var attr = (CFMaxAttribute) attribute;
            var property = serializedParameterOverride.value;
            if (property.propertyType == SerializedPropertyType.Float)
            {
                float v = EditorGUILayout.FloatField (title, property.floatValue);
                property.floatValue = Mathf.Min (v, attr.max);
                return true;
            }

            if (property.propertyType == SerializedPropertyType.Integer)
            {
                int v = EditorGUILayout.IntField (title, property.intValue);
                property.intValue = Mathf.Min (v, (int) attr.max);
                return true;
            }

            return false;
        }
    }

    [CFDecorator (typeof (CFMinMaxAttribute))]
    public sealed class MinMaxDecorator : AttributeDecorator
    {
        public override bool OnGUI (SerializedParameter serializedParameterOverride, GUIContent title, Attribute attribute)
        {
            var attr = (CFMinMaxAttribute) attribute;
            var property = serializedParameterOverride.value;
            if (property.propertyType == SerializedPropertyType.Float)
            {
                float v = EditorGUILayout.FloatField (title, property.floatValue);
                property.floatValue = Mathf.Clamp (v, attr.min, attr.max);
                return true;
            }

            if (property.propertyType == SerializedPropertyType.Integer)
            {
                int v = EditorGUILayout.IntField (title, property.intValue);
                property.intValue = Mathf.Clamp (v, (int) attr.min, (int) attr.max);
                return true;
            }

            if (property.propertyType == SerializedPropertyType.Vector2)
            {
                var v = property.vector2Value;
                EditorGUILayout.MinMaxSlider (title, ref v.x, ref v.y, attr.min, attr.max);
                property.vector2Value = v;
                return true;
            }

            return false;
        }
    }

    [CFDecorator (typeof (CFColorUsageAttribute))]
    public sealed class ColorUsageDecorator : AttributeDecorator
    {
        public override bool OnGUI (SerializedParameter serializedParameterOverride, GUIContent title, Attribute attribute)
        {
            var attr = (CFColorUsageAttribute) attribute;
            var property = serializedParameterOverride.value;

            if (property.propertyType != SerializedPropertyType.Color)
                return false;

            property.colorValue = EditorGUILayout.ColorField (title, property.colorValue, true, attr.showAlpha, attr.hdr);

            return true;
        }
    }

    [CFDecorator(typeof(CFResPathAttribute))]
    public sealed class ResPathUsageDecorator : AttributeDecorator
    {
        public override bool OnGUI(SerializedParameter serializedParameterOverride, GUIContent title, Attribute attribute)
        {
            var attr = (CFResPathAttribute)attribute;
            var resSerializedParameterOverride = serializedParameterOverride as ResSerializedParameterOverride;
            var property = resSerializedParameterOverride.value;
            var rawParam = resSerializedParameterOverride.rawParam;

            if (property.propertyType != SerializedPropertyType.String)
                return false;

            int buttonWidth = 80;

            float indentOffset = EditorGUI.indentLevel * 15f;
            var lineRect = GUILayoutUtility.GetRect(1, EditorGUIUtility.singleLineHeight);

            var labelRect = new Rect(lineRect.x, lineRect.y, EditorGUIUtility.labelWidth - indentOffset, lineRect.height);
            var fieldRect = new Rect(labelRect.xMin, lineRect.y, lineRect.width - labelRect.width, lineRect.height);
            var buttonRect = new Rect(lineRect.x, lineRect.y+EditorGUIUtility.singleLineHeight, buttonWidth, lineRect.height);

            EditorGUI.PrefixLabel(labelRect, title);
            EditorGUI.ObjectField(fieldRect, rawParam.asset, attr.type, false);
            if (!string.IsNullOrEmpty(attr.buttonName))
            {
                if (GUI.Button(buttonRect, EditorUtilities.GetContent(attr.buttonName), EditorStyles.miniButton))
                {
                    if (resSerializedParameterOverride.onButton != null)
                    {
                        resSerializedParameterOverride.onButton();
                    }
                }
            }
            if (!string.IsNullOrEmpty(attr.buttonName2))
            {
                buttonRect.x += 80;
                if (GUI.Button(buttonRect, EditorUtilities.GetContent(attr.buttonName2), EditorStyles.miniButton))
                {
                    if (resSerializedParameterOverride.onButton2 != null)
                    {
                        resSerializedParameterOverride.onButton2();
                    }
                }
            }
            if (!string.IsNullOrEmpty(attr.buttonName3))
            {
                buttonRect.x += 80;
                if (GUI.Button(buttonRect, EditorUtilities.GetContent(attr.buttonName3), EditorStyles.miniButton))
                {
                    if (resSerializedParameterOverride.onButton3 != null)
                    {
                        resSerializedParameterOverride.onButton3();
                    }
                }
            }
            return true;
        }
    }
}
