Shader "Custom/Hair/HairTest01"
{

	Properties
	{
		_MainTex ("Texture", 2D) = "white" {}
		_SpecTex ("SpecTexture", 2D) = "black" {}
		_HairColor("HairColor",Color)=(0.1,0.1,0.1,0.1)
		_HairRootColor("HairRootColor",Color)=(0.1,0.1,0.1,1)
		_SpecColor("xyz:SpecColor  w:SpecIntensity",Color)=(1,1,1,0.5)
		_RimColor("RimColor",Color)=(0.5,0.5,0.5,1)
		_SmoothStep("NdotL SmoothStep x:Min,y:Max,z:Brightness",Vector) = (-0.6,1.6,0.8,0)
		_Parameter("x:SpeShift y:SpeShift1 z:SpecInte w:RimPow",Vector)=(0,1,2,6)
		_Parameter01("specDot",Vector)=(1,0,1,0)
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" "Queue"="Transparent" "IgnoreProjector"="False"}
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			// make fog work
			//#pragma multi_compile_fog
			
			#include "UnityCG.cginc"

			fixed _Cutoff;
			sampler2D _MainTex;
			float4 _MainTex_ST;

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD1;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				//UNITY_FOG_COORDS(1)
				float4 vertex : SV_POSITION;
			};
			
			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				fixed4 col = tex2D(_MainTex, i.uv);
				clip (col.a-0.5);
				return 0;
			}
			ENDCG
		}

		Pass
		{
			Tags{"LightMode"="ForwardBase"}
			Blend   SrcAlpha  OneMinusSrcAlpha
			ZWrite off

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma multi_compile_fog
			//#pragma target 3.0
			
			#include "UnityCG.cginc"
			#include "AutoLight.cginc"
			// #include "../StdLib.hlsl"
			// #include "../Common/LightingHead.hlsl"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
				float2 uv1 : TEXCOORD1;
				float3 normal:NORMAL;
				float4 tangent:TANGENT;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float2 uv1 : TEXCOORD1;
				UNITY_FOG_COORDS(6)
				float4 pos :SV_POSITION;
				float3 tsLightDir : TEXCOORD2;
				float3 tsViewDir  : TEXCOORD3;
			};

			sampler2D _MainTex;
			sampler2D _SpecTex;
			float _TranInte;
			fixed4 _HairColor;
			fixed4 _HairRootColor;
			fixed4 _SpecColor;
			fixed4 _RimColor;
			half4 _Parameter;
			half4 _Parameter01;
			half4 _SmoothStep;

			uniform float4 _LightColor0;
			uniform float4 _SpecTex_ST;
			float4 _DirectionalLightColor0;
			

			#define EPSILON         1.0e-4
			float3 RgbToHsv(float3 c)
			{
				float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
				float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
				float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
				float d = q.x - min(q.w, q.y);
				float e = EPSILON;
				return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
			}

			float3 HsvToRgb(float3 c)
			{
				float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
				float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
				return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
			}
			
			v2f vert (appdata v)
			{
				v2f o;
				o.uv=v.uv;
				o.uv1=v.uv1;

				float3 tangentDir = UnityObjectToWorldDir(v.tangent);
				float3 normalDir = UnityObjectToWorldNormal(v.normal);
				float3 bitangentDir = normalize(cross(normalDir,tangentDir)*v.tangent.w * unity_WorldTransformParams.w);
				float3x3 tangentTransform = float3x3(tangentDir,bitangentDir,normalDir);
				float4 wsPosition = mul(unity_ObjectToWorld,float4(v.vertex.xyz,1));
				o.pos = mul(UNITY_MATRIX_VP,wsPosition);
				o.tsLightDir = mul(tangentTransform,normalize(float3(0,1,0)));
				o.tsViewDir = mul(tangentTransform,normalize(UnityWorldSpaceViewDir(wsPosition)));
				UNITY_TRANSFER_FOG(o,o.pos);
				TRANSFER_VERTEX_TO_FRAGMENT(o);
				return o;
			}
			
			fixed4 frag (v2f i) : SV_Target
			{
				fixed4 texColor=tex2D(_MainTex,i.uv1);
				float3 TSbitangentDir=float3(0,1,0);
			    fixed  ShiftT=tex2D(_SpecTex, TRANSFORM_TEX(i.uv,_SpecTex)).a;
			    float3 TSnormalDir = float3(0,0,1);
			    float3 TSView=i.tsViewDir;
			    float3 TSLightDir0=i.tsLightDir;
				float3 DLight= _LightColor0.rgb;//  _DirectionalLightColor0.rgb;
			    //fixed bug: The specular glow will cross for each other.
				float3 TSHalf=normalize(TSView+TSLightDir0*0.6);
				fixed3 HairC=lerp(_HairRootColor,_HairColor,texColor.g);
				
				//Use HSV algorithm to change the Saturation and Brightness of base color. 
				fixed3 hairHsv = RgbToHsv(HairC);
				HairC = HsvToRgb(fixed3(hairHsv.xy,clamp(hairHsv.z,0,_SmoothStep.z)));
				//The specular color is also brightness more than the base color.
				fixed3 hairSpec = HsvToRgb(hairHsv + fixed3(0,-0.3,0.5));
				//The rim color need to add some colourful things.
				fixed3 hairRim = HsvToRgb(hairHsv + fixed3(0,0.2,0.2));
				
				// The normal formula is max(0,dot(N,L)) , but this is a trick to remapping the cos(N,L) to 0~1.
				// It can improve the drakness range and average the part of 0 to 1.
				float  NdotL = smoothstep(_SmoothStep.x,_SmoothStep.y,dot(TSLightDir0,TSnormalDir));
				float3 diffuse=lerp(HairC*HairC*0.33,HairC,NdotL)*texColor.r*_HairColor.a*10;
				fixed3 specT =tex2D(_SpecTex,float2(1,(1-dot(TSHalf,normalize(TSbitangentDir+fixed3(0,0,1)*((ShiftT.x-0.5)*_Parameter.y-0.5+_Parameter.x)).xyz))*0.5)).rgb;
				//The brigness of base color will effect the strength of the specular color.
				fixed3 specTex=dot(_Parameter01.xyz,specT*specT)*texColor.b*_Parameter.z*saturate(0.1+max(max(HairC.r,HairC.g),HairC.b));
				// The specular effect need depend on NdotL,otherwise it is always lighting in the dark.
				half   NdotL2 = max(1.5*NdotL,0.5);
				NdotL2 *= NdotL2;
				fixed3 specular=specTex*hairSpec*_SpecColor.rgb*_SpecColor.a*NdotL2;
				
				fixed3 fresnel=pow(1-saturate(dot(TSView,TSnormalDir)),_Parameter.w)*hairRim*_RimColor.rgb*_RimColor.a;

				fixed4 finalCol=float4((diffuse+specular+fresnel)*DLight,min(texColor.a*2,1));

				// UNITY_APPLY_FOG(i.fogCoord, finalCol);
				return finalCol;
			}
			ENDCG
		}


	}
}
