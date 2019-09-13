#ifndef UNITY_POSTFX_SAMPLING
#define UNITY_POSTFX_SAMPLING

#include "StdLib.hlsl"

// Better, temporally stable box filtering
// [Jimenez14] http://goo.gl/eomGso
// . . . . . . .
// . A . B . C .
// . . D . E . .
// . F . G . H .
// . . I . J . .
// . K . L . M .
// . . . . . . .
half4 DownsampleBox13Tap(TEXTURE2D_ARGS(tex), float2 uv, float2 texelSize)
{
    half4 A = SAMPLE_TEXTURE2D(tex, (uv + texelSize * float2(-1.0, -1.0)));
    half4 B = SAMPLE_TEXTURE2D(tex, (uv + texelSize * float2( 0.0, -1.0)));
    half4 C = SAMPLE_TEXTURE2D(tex, (uv + texelSize * float2( 1.0, -1.0)));
    half4 D = SAMPLE_TEXTURE2D(tex, (uv + texelSize * float2(-0.5, -0.5)));
    half4 E = SAMPLE_TEXTURE2D(tex, (uv + texelSize * float2( 0.5, -0.5)));
    half4 F = SAMPLE_TEXTURE2D(tex, (uv + texelSize * float2(-1.0,  0.0)));
    half4 G = SAMPLE_TEXTURE2D(tex, (uv                                 ));
    half4 H = SAMPLE_TEXTURE2D(tex, (uv + texelSize * float2( 1.0,  0.0)));
    half4 I = SAMPLE_TEXTURE2D(tex, (uv + texelSize * float2(-0.5,  0.5)));
    half4 J = SAMPLE_TEXTURE2D(tex, (uv + texelSize * float2( 0.5,  0.5)));
    half4 K = SAMPLE_TEXTURE2D(tex, (uv + texelSize * float2(-1.0,  1.0)));
    half4 L = SAMPLE_TEXTURE2D(tex, (uv + texelSize * float2( 0.0,  1.0)));
    half4 M = SAMPLE_TEXTURE2D(tex, (uv + texelSize * float2( 1.0,  1.0)));

    half2 div = (1.0 / 4.0) * half2(0.5, 0.125);

    half4 o = (D + E + I + J) * div.x;
    o += (A + B + G + F) * div.y;
    o += (B + C + H + G) * div.y;
    o += (F + G + L + K) * div.y;
    o += (G + H + M + L) * div.y;

    return o;
}

// Standard box filtering
half4 DownsampleBox4Tap(TEXTURE2D_ARGS(tex), float2 uv, float2 texelSize)
{
    float4 d = texelSize.xyxy * float4(-1.0, -1.0, 1.0, 1.0);

    half4 s;
    s =  (SAMPLE_TEXTURE2D(tex, (uv + d.xy)));
    s += (SAMPLE_TEXTURE2D(tex, (uv + d.zy)));
    s += (SAMPLE_TEXTURE2D(tex, (uv + d.xw)));
    s += (SAMPLE_TEXTURE2D(tex, (uv + d.zw)));

    return s * (1.0 / 4.0);
}

half4 DownsampleBoxMedian(TEXTURE2D_ARGS(tex), float2 uv, float2 texelSize)
{
    float4 d = texelSize.xyxy * float4(-1.0, -1.0, 1.0, 1.0);

    half4 s0,s1,s2,s3,s4;
    s0 = (SAMPLE_TEXTURE2D(tex, (uv)));
    s1 = (SAMPLE_TEXTURE2D(tex, (uv + d.xy)));
    s2 = (SAMPLE_TEXTURE2D(tex, (uv + d.zy)));
    s3 = (SAMPLE_TEXTURE2D(tex, (uv + d.xw)));
    s4 = (SAMPLE_TEXTURE2D(tex, (uv + d.zw)));

    s0 = s0 + s1 + s2 - Max3(s0,s1,s2) - Min3(s0,s1,s2);
    s0 = s0 + s3 + s4 - Max3(s0,s3,s4) - Min3(s0,s3,s4);

    return s0;
}

// 9-tap bilinear upsampler (tent filter)
half4 UpsampleTent(TEXTURE2D_ARGS(tex), float2 uv, float2 texelSize, float sampleScale)
{
    float4 d = texelSize.xyxy * float4(1.0, 1.0, -1.0, 0.0) * sampleScale;

    half4 s;
    s =  SAMPLE_TEXTURE2D(tex, (uv - d.xy));
    s += SAMPLE_TEXTURE2D(tex, (uv - d.wy)) * 2.0;
    s += SAMPLE_TEXTURE2D(tex, (uv - d.zy));

    s += SAMPLE_TEXTURE2D(tex, (uv + d.zw)) * 2.0;
    s += SAMPLE_TEXTURE2D(tex, (uv       )) * 4.0;
    s += SAMPLE_TEXTURE2D(tex, (uv + d.xw)) * 2.0;

    s += SAMPLE_TEXTURE2D(tex, (uv + d.zy));
    s += SAMPLE_TEXTURE2D(tex, (uv + d.wy)) * 2.0;
    s += SAMPLE_TEXTURE2D(tex, (uv + d.xy));

    return s * (1.0 / 16.0);
}

// Standard box filtering
half4 UpsampleBox(TEXTURE2D_ARGS(tex), float2 uv, float2 texelSize, float sampleScale)
{
    float4 d = texelSize.xyxy * float4(-1.0, -1.0, 1.0, 1.0) * (sampleScale * 0.5);

    half4 s;
    s =  (SAMPLE_TEXTURE2D(tex, (uv + d.xy)));
    s += (SAMPLE_TEXTURE2D(tex, (uv + d.zy)));
    s += (SAMPLE_TEXTURE2D(tex, (uv + d.xw)));
    s += (SAMPLE_TEXTURE2D(tex, (uv + d.zw)));

    return s * (1.0 / 4.0);
}

half4 UpsampleBox4(TEXTURE2D_ARGS(tex0), TEXTURE2D_ARGS(tex1), TEXTURE2D_ARGS(tex2), 
		float2 uv, float2 texelSize, float sampleScale)
{
	float4 d0 = texelSize.xyxy * float4(-1.0, -1.0, 1.0, 1.0) * (sampleScale * 0.5);
	float4 d1 = d0* 0.5;
	float4 d2 = d1* 0.5;
	float4 d3 = d2* 0.5;

	half4 s0 = (SAMPLE_TEXTURE2D(tex0, (uv + d0.xy)));
	half4 s1 = (SAMPLE_TEXTURE2D(tex1, (uv + d1.xy)));	
	half4 s2 = (SAMPLE_TEXTURE2D(tex2, (uv + d2.xy)));

	half4 s2sum = s2 + (SAMPLE_TEXTURE2D(tex2, (uv + d2.zy)));
	s2sum += (SAMPLE_TEXTURE2D(tex2, (uv + d2.xw)));
	s2sum += (SAMPLE_TEXTURE2D(tex2, (uv + d2.zw)));
	s2sum *= 0.5f;

	half4 s1sum = s1 + (SAMPLE_TEXTURE2D(tex1, (uv + d1.zy)));
	s1sum += (SAMPLE_TEXTURE2D(tex1, (uv + d1.xw)));
	s1sum += (SAMPLE_TEXTURE2D(tex1, (uv + d1.zw)));
	s1sum *= 0.5f;
	s1sum += s2sum;

	half4 s0sum = s0 + (SAMPLE_TEXTURE2D(tex0, (uv + d0.zy)));
	s0sum += (SAMPLE_TEXTURE2D(tex0, (uv + d0.xw)));
	s0sum += (SAMPLE_TEXTURE2D(tex0, (uv + d0.zw)));
	s0sum *= 0.25f;
	s0sum += s1sum;

	return s0sum;
}

#endif // UNITY_POSTFX_SAMPLING
