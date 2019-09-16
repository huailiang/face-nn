using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEditor;

namespace XEngine.Editor
{
    public static class EditorUtilities
    {
        static Dictionary<string, GUIContent> s_GUIContentCache;
        static Dictionary<Type, AttributeDecorator> s_AttributeDecorators;

        public static bool isTargetingConsoles
        {
            get
            {
                var t = EditorUserBuildSettings.activeBuildTarget;
                return t == BuildTarget.PS4
                    || t == BuildTarget.XboxOne
                    || t == BuildTarget.Switch;
            }
        }

        public static bool isTargetingMobiles
        {
            get
            {
                var t = EditorUserBuildSettings.activeBuildTarget;
                return t == BuildTarget.Android
                    || t == BuildTarget.iOS;
            }
        }

        public static bool isTargetingConsolesOrMobiles
        {
            get { return isTargetingConsoles || isTargetingMobiles; }
        }

        static EditorUtilities()
        {
            s_GUIContentCache = new Dictionary<string, GUIContent>();
            s_AttributeDecorators = new Dictionary<Type, AttributeDecorator>();
            ReloadDecoratorTypes();
        }

        [UnityEditor.Callbacks.DidReloadScripts]
        static void OnEditorReload()
        {
            ReloadDecoratorTypes();
        }

        static void ReloadDecoratorTypes()
        {
            s_AttributeDecorators.Clear();

            // Look for all the valid attribute decorators
            var types = RuntimeUtilities.GetAllAssemblyTypes()
                            .Where(
                                t => t.IsSubclassOf(typeof(AttributeDecorator))
                                  && t.IsDefined(typeof(CFDecoratorAttribute), false)
                                  && !t.IsAbstract
                            );

            // Store them
            foreach (var type in types)
            {
                var attr = type.GetAttribute<CFDecoratorAttribute>();
                var decorator = (AttributeDecorator)Activator.CreateInstance(type);
                s_AttributeDecorators.Add(attr.attributeType, decorator);
            }
        }

        internal static AttributeDecorator GetDecorator(Type attributeType)
        {
            AttributeDecorator decorator;
            return !s_AttributeDecorators.TryGetValue(attributeType, out decorator)
                ? null
                : decorator;
        }

        public static GUIContent GetContent(string textAndTooltip)
        {
            if (string.IsNullOrEmpty(textAndTooltip))
                return GUIContent.none;

            GUIContent content;

            if (!s_GUIContentCache.TryGetValue(textAndTooltip, out content))
            {
                var s = textAndTooltip.Split('|');
                content = new GUIContent(s[0]);

                if (s.Length > 1 && !string.IsNullOrEmpty(s[1]))
                    content.tooltip = s[1];

                s_GUIContentCache.Add(textAndTooltip, content);
            }

            return content;
        }

        public static void DrawFixMeBox(string text, Action action)
        {
            Assert.IsNotNull(action);

            EditorGUILayout.HelpBox(text, MessageType.Warning);

            GUILayout.Space(-32);
            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.FlexibleSpace();

                if (GUILayout.Button("Fix", GUILayout.Width(60)))
                    action();

                GUILayout.Space(8);
            }
            GUILayout.Space(11);
        }

        public static void DrawSplitter()
        {
            var rect = GUILayoutUtility.GetRect(1f, 1f);

            // Splitter rect should be full-width
            rect.xMin = 0f;
            rect.width += 4f;

            if (Event.current.type != EventType.Repaint)
                return;

            EditorGUI.DrawRect(rect, !EditorGUIUtility.isProSkin
                ? new Color(0.6f, 0.6f, 0.6f, 1.333f)
                : new Color(0.12f, 0.12f, 0.12f, 1.333f));
        }

        public static void DrawOverrideCheckbox(Rect rect, SerializedProperty property)
        {
            var oldColor = GUI.color;
            GUI.color = new Color(0.6f, 0.6f, 0.6f, 0.75f);
            property.boolValue = GUI.Toggle(rect, property.boolValue, GetContent("|Override this setting for this volume."), Styling.smallTickbox);
            GUI.color = oldColor;
        }

        public static void DrawHeaderLabel(string title)
        {
            EditorGUILayout.LabelField(title, Styling.labelHeader);
        }

        public static bool DrawHeader(string title, bool state)
        {
            var backgroundRect = GUILayoutUtility.GetRect(1f, 17f);

            var labelRect = backgroundRect;
            labelRect.xMin += 16f;
            labelRect.xMax -= 20f;

            var foldoutRect = backgroundRect;
            foldoutRect.y += 1f;
            foldoutRect.width = 13f;
            foldoutRect.height = 13f;

            // Background rect should be full-width
            backgroundRect.xMin = 0f;
            backgroundRect.width += 4f;

            // Background
            float backgroundTint = EditorGUIUtility.isProSkin ? 0.1f : 1f;
            EditorGUI.DrawRect(backgroundRect, new Color(backgroundTint, backgroundTint, backgroundTint, 0.2f));

            // Title
            EditorGUI.LabelField(labelRect, GetContent(title), EditorStyles.boldLabel);

            // Active checkbox
            state = GUI.Toggle(foldoutRect, state, GUIContent.none, EditorStyles.foldout);

            var e = Event.current;
            if (e.type == EventType.MouseDown && backgroundRect.Contains(e.mousePosition) && e.button == 0)
            {
                state = !state;
                e.Use();
            }

            return state;
        }

        public static bool DrawHeader(string title, SerializedProperty group, Action resetAction, Action removeAction)
        {
            Assert.IsNotNull(group);

            var backgroundRect = GUILayoutUtility.GetRect(1f, 17f);

            var labelRect = backgroundRect;
            labelRect.xMin += 16f;
            labelRect.xMax -= 20f;

            var toggleRect = backgroundRect;
            toggleRect.y += 2f;
            toggleRect.width = 13f;
            toggleRect.height = 13f;

            var menuIcon = EditorGUIUtility.isProSkin
                ? Styling.paneOptionsIconDark
                : Styling.paneOptionsIconLight;

            var menuRect = new Rect(labelRect.xMax + 4f, labelRect.y + 4f, menuIcon.width, menuIcon.height);

            // Background rect should be full-width
            backgroundRect.xMin = 0f;
            backgroundRect.width += 4f;

            // Background
            float backgroundTint = EditorGUIUtility.isProSkin ? 0.1f : 1f;
            EditorGUI.DrawRect(backgroundRect, new Color(backgroundTint, backgroundTint, backgroundTint, 0.2f));

            // Dropdown menu icon
            GUI.DrawTexture(menuRect, menuIcon);

            return group.isExpanded;
        }

        static void ShowHeaderContextMenu(Vector2 position, Action resetAction, Action removeAction)
        {
            Assert.IsNotNull(resetAction);
            Assert.IsNotNull(removeAction);

            var menu = new GenericMenu();
            menu.AddItem(GetContent("Reset"), false, () => resetAction());
            menu.AddItem(GetContent("Remove"), false, () => removeAction());
            menu.AddSeparator(string.Empty);
            menu.DropDown(new Rect(position, Vector2.zero));
        }


    }
}
