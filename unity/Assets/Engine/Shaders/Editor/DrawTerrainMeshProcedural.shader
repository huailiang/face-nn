Shader "Custom/Editor/DrawTerrainMeshProcedural"
{
	Properties
	{
		_Color("Outline Color",Color) = (1,1,1,1)
	}

	SubShader
	{
		ZWrite Off
		// ZTest Off
		Pass
		{
			Tags{ "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" }
			LOD 100

			Blend SrcAlpha OneMinusSrcAlpha
			CGPROGRAM


			#pragma vertex Vert
			#pragma geometry geom
			#pragma fragment Frag
			// #pragma target 4.5

			#include "UnityCG.cginc"

			struct v2g
			{
				float4 projectionSpaceVertex : SV_POSITION;
				float2 uv0 : TEXCOORD0;
				float4 color : COLOR0;
			};
			
			struct g2f
			{
				float4 projectionSpaceVertex : SV_POSITION;
				float2 uv0 : TEXCOORD0;
				float4 dist : TEXCOORD2;
				float4 color : COLOR0;
			};

			float4 _Color;
			float4 _PosOffset;
			float4 _GridSize;
			uint _LineVertexCount;
			uint _LineBlockCount;
			uint _HeightOffset;
			static const uint indexOffsets[6] = 
			{
				0,3,1,
				0,2,3
			};
			// StructuredBuffer<Point> points;
			StructuredBuffer<float> vertexHeight;
			uniform float _WireThickness;
			v2g Vert(uint id : SV_VertexID, 
				uint instanceID : SV_InstanceID)
			{
				bool bottomTriangle = instanceID%2==0;
				uint blockIndex = bottomTriangle?(instanceID/2):((instanceID-1)/2);

				uint xLineIndex = blockIndex%_LineBlockCount;
				uint zLineIndex = blockIndex/_LineBlockCount;
				uint pointIndex[4] = 
				{
					xLineIndex + zLineIndex * _LineVertexCount,
					xLineIndex + zLineIndex * _LineVertexCount + 1,
					xLineIndex + zLineIndex * _LineVertexCount + _LineVertexCount,
					xLineIndex + zLineIndex * _LineVertexCount + _LineVertexCount + 1,
				};
				uint vertexIndex = bottomTriangle?id:(id+3);
				uint indexOffset = indexOffsets[vertexIndex];
				uint index = pointIndex[indexOffset];
				float3 points[4] = 
				{
					float3(0,0,0),
					float3(_PosOffset.w,0,0),
					float3(0,0,_PosOffset.w),				
					float3(_PosOffset.w,0,_PosOffset.w),
				};
				float4 p = float4(points[indexOffset],1);
				p.x += xLineIndex * _PosOffset.w + _PosOffset.x;
				p.z += zLineIndex * _PosOffset.w + _PosOffset.z;

				p.y = vertexHeight[index + _HeightOffset]+0.01f;

				v2g o = (v2g)0;
				o.projectionSpaceVertex = UnityObjectToClipPos(p);
				o.uv0 = float2(p.x*_GridSize.z,p.z*_GridSize.w);
				return o;
			}

			[maxvertexcount(3)]
			void geom(triangle v2g i[3], inout TriangleStream<g2f> triangleStream)
			{
				float2 p0 = i[0].projectionSpaceVertex.xy / i[0].projectionSpaceVertex.w;
				float2 p1 = i[1].projectionSpaceVertex.xy / i[1].projectionSpaceVertex.w;
				float2 p2 = i[2].projectionSpaceVertex.xy / i[2].projectionSpaceVertex.w;

				float2 edge0 = p2 - p1;
				float2 edge1 = p2 - p0;
				float2 edge2 = p1 - p0;

				// To find the distance to the opposite edge, we take the
				// formula for finding the area of a triangle Area = Base/2 * Height, 
				// and solve for the Height = (Area * 2)/Base.
				// We can get the area of a triangle by taking its cross product
				// divided by 2.  However we can avoid dividing our area/base by 2
				// since our cross product will already be double our area.
				float area = abs(edge1.x * edge2.y - edge1.y * edge2.x);
				float wireThickness = 800 - _WireThickness;

				g2f o = (g2f)0;
				
				o.uv0 = i[0].uv0;
				// o.worldSpacePosition = i[0].worldSpacePosition;
				o.projectionSpaceVertex = i[0].projectionSpaceVertex;
				o.dist.xyz = float3( (area / length(edge0)), 0.0, 0.0) * o.projectionSpaceVertex.w * wireThickness;
				o.dist.w = 1.0 / o.projectionSpaceVertex.w;
				triangleStream.Append(o);

				o.uv0 = i[1].uv0;
				// o.worldSpacePosition = i[1].worldSpacePosition;
				o.projectionSpaceVertex = i[1].projectionSpaceVertex;
				o.dist.xyz = float3(0.0, (area / length(edge1)), 0.0) * o.projectionSpaceVertex.w * wireThickness;
				o.dist.w = 1.0 / o.projectionSpaceVertex.w;
				triangleStream.Append(o);

				o.uv0 = i[2].uv0;
				// o.worldSpacePosition = i[2].worldSpacePosition;
				o.projectionSpaceVertex = i[2].projectionSpaceVertex;
				o.dist.xyz = float3(0.0, 0.0, (area / length(edge2))) * o.projectionSpaceVertex.w * wireThickness;
				o.dist.w = 1.0 / o.projectionSpaceVertex.w;
				triangleStream.Append(o);
			}
			half4 Frag(g2f i) : SV_Target
			{
				half4 color = _Color;
				color.a = 0.1;
				return color;
			}
			ENDCG
		}
	}
}
