#ifndef PBS_EFFECT_INCLUDE
#define PBS_EFFECT_INCLUDE

#ifdef _WIND_EFFECT
FLOAT4 _WindDir;
FLOAT4 _WindPos;
#endif
#ifdef _BLOCK_WIND_EFFECT
// FLOAT4 _WindPlane;//xyz normal w dist
FLOAT4 _WindParam0;//x max range  y range z range scale

#define _WindRotCos _WindParam0.x
#define _WindRotSin _WindParam0.y
#define _WindStrength _WindParam0.z
#define _WindScale _WindParam0.w

#endif
FLOAT4 _Interactive;

FLOAT4 qmul(FLOAT4 q1, FLOAT4 q2)
{
	return FLOAT4(
		q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
		q1.w * q2.w - dot(q1.xyz, q2.xyz)
	);
}

// Rotate a vector with a rotation quaternion.
// http://mathworld.wolfram.com/Quaternion.html
FLOAT3 rotate_vector(FLOAT3 v, FLOAT4 r)
{
	FLOAT4 r_c = r * FLOAT4(-1, -1, -1, 1);
	return qmul(r, qmul(FLOAT4(v, 0), r_c)).xyz;
}
#ifdef _BLOCK_WIND_EFFECT


inline FLOAT2 WindStrength(FLOAT3 pos)
{
	return FLOAT2(pos.x +_Time.w *_WindStrength,pos.z +_Time.w *_WindStrength);
}

inline FLOAT2 Wind(FLOAT3 pos, FLOAT windStrength,FLOAT windForce)
{
	FLOAT3 realPos = FLOAT3(pos.x*_WindRotCos - pos.z * _WindRotSin, pos.y, pos.x*_WindRotSin+pos.z*_WindRotCos);
	FLOAT2 windWaveStrength = FLOAT2(realPos.x +_Time.w *windStrength,realPos.z +_Time.w *windStrength);
	windWaveStrength = windForce* sin(0.7*windWaveStrength)*CosLike(0.15*windWaveStrength);
	return FLOAT2(windWaveStrength.x*_WindRotCos- windWaveStrength.y*_WindRotSin,windWaveStrength.x*_WindRotSin+windWaveStrength.y*_WindRotCos);
}
#endif

FLOAT4 VertexEffect(FVertexInput Input,FLOAT4 WorldPosition)
{
	// #if (defined(_GLOBAL_WIND_EFFECT)||defined(_INTERACTIVE))
	#if (defined(_WIND_EFFECT)||defined(_BLOCK_WIND_EFFECT)||defined(_INTERACTIVE))
		FLOAT fallmask = Input.uv0.y;
		fallmask = saturate(fallmask-0.3f);
	#endif

	#ifdef _WIND_EFFECT
		// #ifdef _GLOBAL_WIND_EFFECT
			FLOAT3 windPos = WorldPosition.xyz - _WindPos.xyz;
			FLOAT  windDist = length(windPos);
			FLOAT3 nWindDir = windPos * rcp(windDist);

			FLOAT x = CosLike(windDist*PI + _Time.y) * FLOAT(0.5) + FLOAT(0.5);
			#ifndef _BLOCK_WIND_EFFECT
				fallmask = Input.Color.x;
			#endif
			WorldPosition.xyz += normalize(-Input.TangentZ.xyz + _WindDir.xyz + nWindDir) * (x * _WindDir.w) * fallmask;
		// #endif
	#endif



	#ifdef _BLOCK_WIND_EFFECT
		// #ifdef _GLOBAL_WIND_EFFECT
			FLOAT3 randPos = WorldPosition.xyz;

			FLOAT2 windDir = Wind(randPos,_WindStrength,_WindScale);

			WorldPosition.xz += windDir.xy*fallmask;
			WorldPosition.y -= length(windDir)*0.5*fallmask;
		// #endif
	#endif

	#ifdef _INTERACTIVE
			FLOAT3 dist2Obj = WorldPosition.xyz - _Interactive.xyz;
			FLOAT dist2Obj2 = dot(dist2Obj, dist2Obj);
			dist2Obj2 = 1 - saturate(dist2Obj2/(_Interactive.w*_Interactive.w));
			FLOAT3 pushOffset = FLOAT3(dist2Obj2*dist2Obj.x*2,-dist2Obj2,dist2Obj2*dist2Obj.z*2)*fallmask;
			WorldPosition.xyz += pushOffset;
	#endif

	return WorldPosition;
}

#endif //PBS_EFFECT_INCLUDE