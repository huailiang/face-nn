using System;
using System.Diagnostics;
using UnityEngine;

namespace XEngine
{

    [Conditional("UNITY_EDITOR")]
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
    public sealed class DisplayNameAttribute : Attribute
    {
        public readonly string displayName;

        public DisplayNameAttribute(string displayName)
        {
            this.displayName = displayName;
        }
    }

    [Conditional("UNITY_EDITOR")]
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
    public sealed class MaxAttribute : Attribute
    {
        public readonly float max;

        public MaxAttribute(float max)
        {
            this.max = max;
        }
    }

    [Conditional("UNITY_EDITOR")]
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
    public sealed class MinAttribute : Attribute
    {
        public readonly float min;

        public MinAttribute(float min)
        {
            this.min = min;
        }
    }

    [Conditional("UNITY_EDITOR")]
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
    public sealed class MinMaxAttribute : Attribute
    {
        public readonly float min;
        public readonly float max;

        public MinMaxAttribute(float min, float max)
        {
            this.min = min;
            this.max = max;
        }
    }
    [Conditional("UNITY_EDITOR")]
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
    public sealed class TrackballAttribute : Attribute
    {
        public enum Mode
        {
            None,
            Lift,
            Gamma,
            Gain
        }

        public readonly Mode mode;

        public TrackballAttribute(Mode mode)
        {
            this.mode = mode;
        }
    }



    [Conditional("UNITY_EDITOR")]
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
    public class RangeAttribute : PropertyAttribute
    {
        public readonly float min;
        public readonly float max;

        public RangeAttribute(float min, float max)
        {
            this.min = min;
            this.max = max;
        }
    }


    [Conditional("UNITY_EDITOR")]
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
    public class ResPathAttribute : PropertyAttribute
    {
        public readonly Type type;
        public readonly string buttonName;
        public readonly string buttonName2;
        public readonly string buttonName3;
        public ResPathAttribute(Type type, string buttonName, string buttonName2, string buttonName3)
        {
            this.type = type;
            this.buttonName = buttonName;
            this.buttonName2 = buttonName2;
            this.buttonName3 = buttonName3;
        }
    }

    [Conditional("UNITY_EDITOR")]
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
    public sealed class ColorUsageAttribute : PropertyAttribute
    {
        //
        //     If false then the alpha bar is hidden in the ColorField and the alpha value is
        //     not shown in the Color Picker.
        //     ///
        public readonly bool showAlpha;
        //
        //     If set to true the Color is treated as a HDR color.
        //     ///
        public readonly bool hdr;
        //
        //     Minimum allowed HDR color component value when using the Color Picker.
        //     ///
        public readonly float minBrightness;
        //
        //     Maximum allowed HDR color component value when using the HDR Color Picker.
        //     ///
        public readonly float maxBrightness;
        //
        //     Minimum exposure value allowed in the HDR Color Picker.
        //     ///
        public readonly float minExposureValue;
        //
        //     Maximum exposure value allowed in the HDR Color Picker.
        //     ///
        public readonly float maxExposureValue;

        //
        //     Attribute for Color fields. Used for configuring the GUI for the color.
        //   showAlpha:
        //     If false then the alpha channel info is hidden both in the ColorField and in
        //     the Color Picker.
        //
        //   hdr:
        //     Set to true if the color should be treated as a HDR color (default value: false).
        //
        //   minBrightness:
        //     Minimum allowed HDR color component value when using the HDR Color Picker (default
        //     value: 0).
        //
        //   maxBrightness:
        //     Maximum allowed HDR color component value when using the HDR Color Picker (default
        //     value: 8).
        //
        //   minExposureValue:
        //     Minimum exposure value allowed in the HDR Color Picker (default value: 1/8 =
        //     0.125).
        //
        //   maxExposureValue:
        //     Maximum exposure value allowed in the HDR Color Picker (default value: 3).
        public ColorUsageAttribute(bool showAlpha)
        {
            this.showAlpha = showAlpha;
        }

        //
        //     ///
        //     Attribute for Color fields. Used for configuring the GUI for the color.
        //     ///
        //
        //   showAlpha:
        //     If false then the alpha channel info is hidden both in the ColorField and in
        //     the Color Picker.
        //
        //   hdr:
        //     Set to true if the color should be treated as a HDR color (default value: false).
        //
        //   minBrightness:
        //     Minimum allowed HDR color component value when using the HDR Color Picker (default
        //     value: 0).
        //
        //   maxBrightness:
        //     Maximum allowed HDR color component value when using the HDR Color Picker (default
        //     value: 8).
        //
        //   minExposureValue:
        //     Minimum exposure value allowed in the HDR Color Picker (default value: 1/8 =
        //     0.125).
        //
        //   maxExposureValue:
        //     Maximum exposure value allowed in the HDR Color Picker (default value: 3).
        public ColorUsageAttribute(bool showAlpha, bool hdr, float minBrightness, float maxBrightness, float minExposureValue, float maxExposureValue)
        {
            this.showAlpha = showAlpha;

            this.hdr = showAlpha;

            this.minBrightness = minBrightness;

            this.maxBrightness = maxBrightness;

            this.minExposureValue = minExposureValue;

            this.maxExposureValue = maxExposureValue;
        }
    }

    [Conditional("UNITY_EDITOR")]
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = false)]
    public sealed class NoSerializedAttribute : Attribute
    {
        internal NoSerializedAttribute()
        {
        }
    }
}