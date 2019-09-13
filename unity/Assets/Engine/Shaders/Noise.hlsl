#ifndef UNITY_POSTFX_NOISE
#define UNITY_POSTFX_NOISE

#include "StdLib.hlsl"

FLOAT4 mod289(FLOAT4 x)
{
	return x - floor(x * (1.0 / 289.0)) * 289.0;
}

FLOAT4 permute(FLOAT4 x)
{
	return mod289(((x*34.0)+1.0)*x);
}

FLOAT4 taylorInvSqrt(FLOAT4 r)
{
	return 1.79284291400159 - 0.85373472095314 * r;
}

FLOAT2 fade(FLOAT2 t) 
{
	return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Classic Perlin noise
FLOAT cnoise(FLOAT2 P)
{
	FLOAT4 Pi = floor(P.xyxy) + FLOAT4(0.0, 0.0, 1.0, 1.0);
	FLOAT4 Pf = frac(P.xyxy) - FLOAT4(0.0, 0.0, 1.0, 1.0);
	Pi = mod289(Pi); // To avoid truncation effects in permutation
	FLOAT4 ix = Pi.xzxz;
	FLOAT4 iy = Pi.yyww;
	FLOAT4 fx = Pf.xzxz;
	FLOAT4 fy = Pf.yyww;

	FLOAT4 i = permute(permute(ix) + iy);

	FLOAT4 gx = frac(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
	FLOAT4 gy = abs(gx) - 0.5 ;
	FLOAT4 tx = floor(gx + 0.5);
	gx = gx - tx;

	FLOAT2 g00 = FLOAT2(gx.x,gy.x);
	FLOAT2 g10 = FLOAT2(gx.y,gy.y);
	FLOAT2 g01 = FLOAT2(gx.z,gy.z);
	FLOAT2 g11 = FLOAT2(gx.w,gy.w);

	FLOAT4 norm = taylorInvSqrt(FLOAT4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
	g00 *= norm.x;  
	g01 *= norm.y;  
	g10 *= norm.z;  
	g11 *= norm.w;  

	FLOAT n00 = dot(g00, FLOAT2(fx.x, fy.x));
	FLOAT n10 = dot(g10, FLOAT2(fx.y, fy.y));
	FLOAT n01 = dot(g01, FLOAT2(fx.z, fy.z));
	FLOAT n11 = dot(g11, FLOAT2(fx.w, fy.w));

	FLOAT2 fade_xy = fade(Pf.xy);
	FLOAT2 n_x = lerp(FLOAT2(n00, n01), FLOAT2(n10, n11), fade_xy.x);
	FLOAT n_xy = lerp(n_x.x, n_x.y, fade_xy.y);
	return 2.3 * n_xy;
}

// Classic Perlin noise, periodic variant
FLOAT pnoise(FLOAT2 P, FLOAT2 rep)
{
	FLOAT4 Pi = floor(P.xyxy) + FLOAT4(0.0, 0.0, 1.0, 1.0);
	FLOAT4 Pf = frac(P.xyxy) - FLOAT4(0.0, 0.0, 1.0, 1.0);
	Pi = fmod(Pi, rep.xyxy); // To create noise with explicit period
	Pi = mod289(Pi);        // To avoid truncation effects in permutation
	FLOAT4 ix = Pi.xzxz;
	FLOAT4 iy = Pi.yyww;
	FLOAT4 fx = Pf.xzxz;
	FLOAT4 fy = Pf.yyww;

	FLOAT4 i = permute(permute(ix) + iy);

	FLOAT4 gx = frac(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
	FLOAT4 gy = abs(gx) - 0.5 ;
	FLOAT4 tx = floor(gx + 0.5);
	gx = gx - tx;

	FLOAT2 g00 = FLOAT2(gx.x,gy.x);
	FLOAT2 g10 = FLOAT2(gx.y,gy.y);
	FLOAT2 g01 = FLOAT2(gx.z,gy.z);
	FLOAT2 g11 = FLOAT2(gx.w,gy.w);

	FLOAT4 norm = taylorInvSqrt(FLOAT4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
	g00 *= norm.x;  
	g01 *= norm.y;  
	g10 *= norm.z;  
	g11 *= norm.w;  

	FLOAT n00 = dot(g00, FLOAT2(fx.x, fy.x));
	FLOAT n10 = dot(g10, FLOAT2(fx.y, fy.y));
	FLOAT n01 = dot(g01, FLOAT2(fx.z, fy.z));
	FLOAT n11 = dot(g11, FLOAT2(fx.w, fy.w));

	FLOAT2 fade_xy = fade(Pf.xy);
	FLOAT2 n_x = lerp(FLOAT2(n00, n01), FLOAT2(n10, n11), fade_xy.x);
	FLOAT n_xy = lerp(n_x.x, n_x.y, fade_xy.y);
	return 2.3 * n_xy;
}
FLOAT2 hash22(FLOAT2 p)
{
	p = FLOAT2(dot(p, FLOAT2(127.1, 311.7)),
		dot(p, FLOAT2(269.5, 183.3)));

	return -1.0 + 2.0 * frac(sin(p)*43758.5453123);
}

//float perlin_noise(FLOAT2 p)
//{
//	FLOAT2 pi = floor(p);
//	FLOAT2 pf = p - pi;
//
//	FLOAT2 w = pf * pf * (3.0 - 2.0 * pf);
//
//	return mix(mix(dot(hash22(pi + FLOAT2(0.0, 0.0)), pf - FLOAT2(0.0, 0.0)),
//		dot(hash22(pi + FLOAT2(1.0, 0.0)), pf - FLOAT2(1.0, 0.0)), w.x),
//		mix(dot(hash22(pi + FLOAT2(0.0, 1.0)), pf - FLOAT2(0.0, 1.0)),
//			dot(hash22(pi + FLOAT2(1.0, 1.0)), pf - FLOAT2(1.0, 1.0)), w.x),
//		w.y);
//}

FLOAT simplex_noise(FLOAT2 p)
{
	const FLOAT K1 = 0.366025404; // (sqrt(3)-1)/2;
	const FLOAT K2 = 0.211324865; // (3-sqrt(3))/6;

	FLOAT2 i = floor(p + (p.x + p.y) * K1);

	FLOAT2 a = p - (i - (i.x + i.y) * K2);
	FLOAT2 o = (a.x < a.y) ? FLOAT2(0.0, 1.0) : FLOAT2(1.0, 0.0);
	FLOAT2 b = a - o + K2;
	FLOAT2 c = a - 1.0 + 2.0 * K2;

	FLOAT3 h = max(0.5 - FLOAT3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
	FLOAT3 n = h * h * h * h * FLOAT3(dot(a, hash22(i)), dot(b, hash22(i + o)), dot(c, hash22(i + 1.0)));

	return dot(FLOAT3(70.0, 70.0, 70.0), n);
}

//without using sin cos function
FLOAT rand(FLOAT2 co)
{
	FLOAT x = dot(co.xy ,FLOAT2(12.9898,78.233));
	x = abs(frac(x)-0.5)*2-1;
	return frac(x * 43758.5453);
}


FLOAT Noise(FLOAT2 v)
{
	FLOAT2 i = floor(v);
	FLOAT2 t = frac(v);
	FLOAT2 u  = t*t*(3-2*t);

	return lerp(lerp(rand(i + FLOAT2(0,0)),rand(i + FLOAT2(1,0)),u.x),
				lerp(rand(i + FLOAT2(0,1)),rand(i + FLOAT2(1,1)),u.x),
				u.y);
}

#endif//UNITY_POSTFX_NOISE