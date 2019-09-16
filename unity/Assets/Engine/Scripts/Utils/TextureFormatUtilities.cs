using System;
using System.Collections.Generic;
using UnityEngine.Assertions;
using UnityEngine;

namespace XEngine
{
    public static class TextureFormatUtilities
    {
        static Dictionary<int, bool> s_SupportedRenderTextureFormats;

        static TextureFormatUtilities()
        {
            //s_FormatAliasMap = new Dictionary<int, RenderTextureFormat>
            //{
            //    { (int)TextureFormat.Alpha8, RenderTextureFormat.ARGB32 },
            //    { (int)TextureFormat.ARGB4444, RenderTextureFormat.ARGB4444 },
            //    { (int)TextureFormat.RGB24, RenderTextureFormat.ARGB32 },
            //    { (int)TextureFormat.RGBA32, RenderTextureFormat.ARGB32 },
            //    { (int)TextureFormat.ARGB32, RenderTextureFormat.ARGB32 },
            //    { (int)TextureFormat.RGB565, RenderTextureFormat.RGB565 },
            //    { (int)TextureFormat.R16, RenderTextureFormat.RHalf },
            //    { (int)TextureFormat.DXT1, RenderTextureFormat.ARGB32 },
            //    { (int)TextureFormat.RFloat, RenderTextureFormat.RFloat },
            //    { (int)TextureFormat.RGFloat, RenderTextureFormat.RGFloat },
            //    { (int)TextureFormat.RGBAFloat, RenderTextureFormat.ARGBFloat },
            //    { (int)TextureFormat.RGB9e5Float, RenderTextureFormat.ARGBHalf },
            //    { (int)TextureFormat.BC4, RenderTextureFormat.R8 },
            //    { (int)TextureFormat.BC5, RenderTextureFormat.RGHalf },
            //    { (int)TextureFormat.BC6H, RenderTextureFormat.ARGBHalf },
            //    { (int)TextureFormat.BC7, RenderTextureFormat.ARGB32 },
            //#if !UNITY_IOS && !UNITY_TVOS
            //    { (int)TextureFormat.DXT1Crunched, RenderTextureFormat.ARGB32 },
            //    { (int)TextureFormat.DXT5Crunched, RenderTextureFormat.ARGB32 },
            //#endif
            //};

            // In 2018.1 SystemInfo.SupportsRenderTextureFormat() generates garbage so we need to
            // cache its calls to avoid that...
            s_SupportedRenderTextureFormats = new Dictionary<int, bool>();
            var values = Enum.GetValues(typeof(RenderTextureFormat));

            foreach (var format in values)
            {
                bool supported = SystemInfo.SupportsRenderTextureFormat((RenderTextureFormat)format);
                s_SupportedRenderTextureFormats.Add((int)format, supported);
            }
        }



        internal static bool IsSupported(this RenderTextureFormat format)
        {
            bool supported;
            s_SupportedRenderTextureFormats.TryGetValue((int)format, out supported);
            return supported;
        }
    }
}
