using System;
using System.Collections.Generic;
using UnityEngine;

namespace CFEngine
{
    public sealed class PropertySheetFactory
    {
        readonly Dictionary<Shader, PropertySheet> m_Sheets;

        public PropertySheetFactory()
        {
            m_Sheets = new Dictionary<Shader, PropertySheet>();
        }

        public PropertySheet Get(Shader shader)
        {
            PropertySheet sheet;

            if (m_Sheets.TryGetValue(shader, out sheet))
                return sheet;

            if (shader == null)
                throw new ArgumentException(string.Format("Invalid shader ({0})", shader));

            var shaderName = shader.name;
            var material = new Material(shader)
            {
                name = string.Format("PostProcess - {0}", shaderName.Substring(shaderName.LastIndexOf('/') + 1)),
                hideFlags = HideFlags.DontSave
            };

            sheet = new PropertySheet(material);
            m_Sheets.Add(shader, sheet);
            return sheet;
        }

        public void Release()
        {
            var it = m_Sheets.GetEnumerator();
            while (it.MoveNext())
            {
                var sheet = it.Current.Value;
                sheet.Release();
            }
            m_Sheets.Clear();
        }
    }
}
