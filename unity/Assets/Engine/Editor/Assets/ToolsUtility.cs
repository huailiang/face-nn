using System;
using System.Collections.Generic;

using CFUtilPoolLib;
using UnityEditor;
using UnityEngine;
namespace CFEngine.Editor
{
    public abstract class CommonToolTemplate : ScriptableObject
    {
        public virtual void OnInit()
        {
        }
        public virtual void OnUninit()
        {
        }

        public virtual void DrawGUI()
        {
        }
        public virtual void DrawSceneGUI()
        {

        }
        public virtual void DrawGizmos()
        {

        }
        public virtual void Update()
        {

        }
    }

    public class ToolsUtility
    {
        public delegate void MouseDownCb(bool left);
        public delegate void MouseDragCb();
        public delegate void MouseUpCb(Vector2Int chunkSrcPoint, Vector2Int chunkEndPoint);
        public delegate void RepaintCb();


        public class GridContext
        {
            public int hLines;
            public int vLines;
            public Rect gridRect = Rect.zero;
            public Rect innerRect = Rect.zero;
            public RectOffset padding = new RectOffset(10, 10, 10, 10);
            public Color bgColor = new Color(0.15f, 0.15f, 0.15f, 1f);
            public Color rectOutlineColor = new Color(0.8f, 0.8f, 0.8f, 0.5f);
            public Color handleColor = new Color(1f, 1f, 1f, 0.05f);
            public Vector2 textSize = new Vector2(20, 20);
            public int gridOffsetH;
            public int gridOffsetV;
            public bool doubleClick = false;
            public Vector2 mousePos;
            public Rect dragRect;
            public Vector2Int clickSrcChunk = new Vector2Int(-1, -1);
            public MouseDownCb mouseDownCb;
            public MouseDragCb mouseDragCb;
            public MouseUpCb mouseUpCb;
            public RepaintCb repaintCb;
            public bool receiveEvent = true;

            public bool drawDragRect = true;
        }
        public static void PrepareGrid(GridContext context, int hLines, int vLines, int hSize = 20, int vSize = 20, int padding = 10)
        {
            context.hLines = hLines;
            context.vLines = vLines;
            context.gridRect = GUILayoutUtility.GetAspectRect(2f);
            context.gridRect.width = hLines * hSize + padding * 2;
            context.gridRect.height = vLines * vSize + padding * 2;
            context.padding.left = padding;
            context.padding.right = padding;
            context.padding.top = padding;
            context.padding.bottom = padding;
            context.innerRect = context.padding.Remove(context.gridRect);
            context.gridOffsetH = Mathf.FloorToInt(context.innerRect.width / hLines);
            context.gridOffsetV = Mathf.FloorToInt(context.innerRect.height / vLines);
        }

        public static void DrawGrid(GridContext context)
        {
            // Background                        
            EditorGUI.DrawRect(context.gridRect, context.bgColor);
            // Bounds
            Handles.color = Color.white * (GUI.enabled ? 1f : 0.5f);
            Handles.DrawSolidRectangleWithOutline(context.innerRect, Color.clear, context.rectOutlineColor);

            Vector2 centerPos = context.innerRect.position;
            centerPos.y += context.innerRect.height;
            // Grid setup
            Handles.color = context.handleColor;
            float halfGridOffset0 = context.gridOffsetH * 0.6f;
            float halfGridOffset1 = context.gridOffsetV * 0.8f;

            int gridPadding = ((int)(context.innerRect.width) % context.hLines) / 2;
            for (int i = 1; i < context.hLines; i++)
            {
                float halfGridOffset = i < 11 ? halfGridOffset0 : halfGridOffset1;
                var offset = i * Vector2.right * context.gridOffsetH;
                offset.x += gridPadding;
                Handles.DrawLine(context.innerRect.position + offset, new Vector2(context.innerRect.x, context.innerRect.yMax - 1) + offset);
                var textoffset = i * Vector2.right * context.gridOffsetH - Vector2.right * halfGridOffset;
                Rect textRect = new Rect(centerPos + textoffset, context.textSize);
                EditorGUI.LabelField(textRect, (i - 1).ToString());
            }
            var lastTextoffset = context.hLines * Vector2.right * context.gridOffsetH - Vector2.right * halfGridOffset1;
            EditorGUI.LabelField(new Rect(centerPos + lastTextoffset, context.textSize), (context.hLines - 1).ToString());

            gridPadding = ((int)(context.innerRect.height) % context.vLines) / 2;
            for (int i = 1; i < context.vLines; i++)
            {
                float halfGridOffset = i < 11 ? halfGridOffset0 : halfGridOffset1;
                var offset = i * Vector2.up * context.gridOffsetV;
                offset.y += gridPadding;
                Handles.DrawLine(context.innerRect.position + offset, new Vector2(context.innerRect.xMax - 1, context.innerRect.y) + offset);
                var textoffset = (i - 1) * Vector2.up * context.gridOffsetV + Vector2.up * halfGridOffset;
                Rect textRect = new Rect(centerPos - textoffset, context.textSize);
                EditorGUI.LabelField(textRect, (i - 1).ToString());
            }
            lastTextoffset = (context.vLines - 1) * Vector2.up * context.gridOffsetV + Vector2.up * halfGridOffset1;
            EditorGUI.LabelField(new Rect(centerPos - lastTextoffset, context.textSize), (context.vLines - 1).ToString());
        }

        public static void DrawBlock(GridContext context, int hIndex, int vIndex, Color color, int padding = 5)
        {
            Rect rect = new Rect();

            rect.xMin = hIndex * context.gridOffsetH + padding;
            rect.yMin = vIndex * context.gridOffsetV + padding;
            rect.width = context.gridOffsetH - padding * 2;
            rect.height = context.gridOffsetV - padding * 2;
            rect.position += context.innerRect.position;
            EditorGUI.DrawRect(rect, color);
        }
        public static Vector2Int CalcGridIndex(GridContext gridContext, Vector2 mousePosition)
        {
            Vector2 pos = mousePosition - gridContext.innerRect.position;
            int xIndex = Mathf.FloorToInt(pos.x / gridContext.gridOffsetH);
            int zIndex = (gridContext.vLines - 1) - Mathf.FloorToInt(pos.y / gridContext.gridOffsetV);
            return new Vector2Int(xIndex, zIndex);
        }

        public static void DrawGrid(GridContext gridContext, int hLines, int vLines, int hSize, int vSize)
        {
            ToolsUtility.PrepareGrid(gridContext, hLines, vLines, hSize, vSize);
            var e = Event.current;
            if (gridContext.receiveEvent)
            {

                if (e.type == EventType.MouseDown)
                {
                    gridContext.doubleClick = e.clickCount == 2;
                    bool leftMouse = e.button == 0;
                    // bool rightMouse = e.button == 1;

                    if (gridContext.innerRect.Contains(e.mousePosition))
                    {
                        gridContext.mousePos = e.mousePosition;
                        gridContext.dragRect = new Rect(0, 0, 0, 0);
                        Vector2Int pos = CalcGridIndex(gridContext, e.mousePosition);
                        if (leftMouse)
                        {
                            gridContext.clickSrcChunk.x = pos.x;
                            gridContext.clickSrcChunk.y = pos.y;
                        }
                        else
                        {
                            gridContext.clickSrcChunk.x = -1;
                            gridContext.clickSrcChunk.y = -1;
                        }
                        if (gridContext.mouseDownCb != null)
                        {
                            gridContext.mouseDownCb(leftMouse);
                        }
                    }
                }
                else if (e.type == EventType.MouseDrag)
                {
                    if (gridContext.innerRect.Contains(e.mousePosition) && gridContext.clickSrcChunk.x >= 0 && gridContext.clickSrcChunk.y >= 0)
                    {
                        Vector2 pos = e.mousePosition;
                        float xMin = pos.x > gridContext.mousePos.x ? gridContext.mousePos.x : pos.x;
                        float yMin = pos.y > gridContext.mousePos.y ? gridContext.mousePos.y : pos.y;
                        float xMax = pos.x < gridContext.mousePos.x ? gridContext.mousePos.x : pos.x;
                        float yMax = pos.y < gridContext.mousePos.y ? gridContext.mousePos.y : pos.y;
                        gridContext.dragRect = new Rect(xMin, yMin, xMax - xMin, yMax - yMin);

                    }
                }
                else if (e.type == EventType.MouseUp)
                {
                    bool leftMouse = e.button == 0;
                    if (leftMouse && gridContext.innerRect.Contains(e.mousePosition) && gridContext.clickSrcChunk.x >= 0 && gridContext.clickSrcChunk.y >= 0)
                    {
                        float dist = Vector2.Distance(e.mousePosition, gridContext.mousePos);
                        if (dist > 0.1f || gridContext.doubleClick)
                        {
                            Vector2 pos = e.mousePosition - gridContext.innerRect.position;
                            int xIndex = Mathf.FloorToInt(pos.x / gridContext.gridOffsetH);
                            int zIndex = (gridContext.vLines - 1) - Mathf.FloorToInt(pos.y / gridContext.gridOffsetV);

                            int srcX = xIndex < gridContext.clickSrcChunk.x ? xIndex : gridContext.clickSrcChunk.x;
                            int srcZ = zIndex < gridContext.clickSrcChunk.y ? zIndex : gridContext.clickSrcChunk.y;
                            int endX = xIndex > gridContext.clickSrcChunk.x ? xIndex : gridContext.clickSrcChunk.x;
                            int endZ = zIndex > gridContext.clickSrcChunk.y ? zIndex : gridContext.clickSrcChunk.y;

                            Vector2Int chunkSrcPoint = new Vector2Int(srcX, srcZ);
                            Vector2Int chunkEndPoint = new Vector2Int(endX, endZ);
                            if (gridContext.mouseUpCb != null)
                            {
                                gridContext.mouseUpCb(chunkSrcPoint, chunkEndPoint);
                            }
                        }

                    }
                    gridContext.clickSrcChunk.x = -1;
                    gridContext.clickSrcChunk.y = -1;
                }
            }
            if (e.type == EventType.Repaint)
            {
                ToolsUtility.DrawGrid(gridContext);
                if (gridContext.repaintCb != null)
                {
                    gridContext.repaintCb();
                }

                //drag rect
                if (gridContext.drawDragRect && gridContext.clickSrcChunk.x >= 0 && gridContext.clickSrcChunk.y >= 0)
                    Handles.DrawSolidRectangleWithOutline(gridContext.dragRect, Color.white, new Color(1f, 1f, 1f, 1f));
            }
        }


        public static void BeginGroup(string name, bool beginHorizontal = true)
        {
            BeginGroup(name, new Vector4Int(0, 0, 1000, 200), beginHorizontal);
        }

        public static void BeginGroup(string name, Vector4Int minMax, bool beginHorizontal)
        {
            if (beginHorizontal)
                GUILayout.BeginHorizontal();
            EditorGUILayout.BeginVertical(GUI.skin.box,
                GUILayout.MinWidth(minMax.x),
                GUILayout.MinHeight(minMax.y),
                GUILayout.MaxWidth(minMax.z),
                GUILayout.MaxHeight(minMax.w));
            if (!string.IsNullOrEmpty(name))
                EditorGUILayout.LabelField(name, EditorStyles.boldLabel);
        }

        public static void EndGroup(bool endHorizontal = true)
        {
            EditorGUILayout.EndVertical();
            if (endHorizontal)
                GUILayout.EndHorizontal();
        }

        public static bool BeginFolderGroup(string name, ref bool folder)
        {
            folder = EditorGUILayout.Foldout(folder, name);
            if (folder)
            {
                ToolsUtility.BeginGroup("");

            }
            return folder;
        }
        public static void EndFolderGroup()
        {
            ToolsUtility.EndGroup();
        }
    }
}