#ifndef PBS_INSTANCE_INCLUDE
#define PBS_INSTANCE_INCLUDE 


#define SUPPORT_COMPUTERBUFFER (SHADER_TARGET>= 45)

#if defined(_INSTANCE) && SUPPORT_COMPUTERBUFFER
	#define _ROT90_POS 1
	uint instanceOffset;
	FLOAT4x4 obj2World;
	#ifdef _RANDOM_TRANS
	
		#if SUPPORT_COMPUTERBUFFER
			StructuredBuffer<FLOAT4> rotationBuffer;
			StructuredBuffer<FLOAT4> positionBuffer;				

		FLOAT4 GetPosData(uint offset)
		{
			return positionBuffer[offset];
		}

		FLOAT4 GetRotData(uint offset)
		{
			return rotationBuffer[offset];
		}
		
		#endif//SUPPORT_COMPUTERBUFFER

	#else//!_RANDOM_TRANS

		#if SUPPORT_COMPUTERBUFFER
			StructuredBuffer<float4x4> obj2WorldMat;

		FLOAT4 GetMatrix(uint offset)
		{
			return obj2WorldMat[offset];
		}
		#endif//SUPPORT_COMPUTERBUFFER

	#endif//_RANDOM_TRANS

	#define INSTANCE_INPUT ,uint instanceID : SV_InstanceID

	FLOAT4 GetInstancePos(FLOAT4 localpos,uint id)
	{
		#ifdef _RANDOM_TRANS

			FLOAT4 data = GetPosData(id+instanceOffset);

			uint rotIndex = (uint)data.w;
			FLOAT4 rot = GetRotData(rotIndex);
			localpos.xz = FLOAT2(localpos.x * rot.y - localpos.z * rot.x, localpos.x * rot.x + localpos.z * rot.y);
			
			localpos.xyz *= rot.z;			
			FLOAT4 wpos = FLOAT4(data.xyz + localpos.xyz, 1);
		#else//!_RANDOM_TRANS
			FLOAT4 wpos = mul(GetMatrix(id), localpos);
		#endif//_RANDOM_TRANS
		return wpos;
	}	

	#define INSTANCE_WPOS(localpos) GetInstancePos(localpos,instanceID);
#else//!_INSTANCE

#define _ROT90_POS 0
#define INSTANCE_INPUT
#define INSTANCE_WPOS(localpos) mul(_objectToWorld, FLOAT4(localpos.xyz, 1.0));

#endif//_INSTANCE
#endif //PBS_INSTANCE_INCLUDE