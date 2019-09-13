// Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.

/*=============================================================================
	BRDF.usf: Bidirectional reflectance distribution functions.
=============================================================================*/

#ifndef __BRDF_COMMON__
#define __BRDF_COMMON__

// Physically based shading model
// parameterized with the below options

// Microfacet specular = D*G*F / (4*NoL*NoV) = D*Vis*F
// Vis = G / (4*NoL*NoV)


float3 Diffuse_Lambert( float3 DiffuseColor )
{
	return DiffuseColor * (1 / PI);
}

// [Burley 2012, "Physically-Based Shading at Disney"]
float3 Diffuse_Burley( float3 DiffuseColor, float Roughness, float NoV, float NoL, float VoH )
{
	float FD90 = 0.5 + 2 * VoH * VoH * Roughness;
	float FdV = 1 + (FD90 - 1) * Pow5( 1 - NoV );
	float FdL = 1 + (FD90 - 1) * Pow5( 1 - NoL );
	return DiffuseColor * ( /*(1 / PI) **/ FdV * FdL );//hack divide PI
}

// [Gotanda 2012, "Beyond a Simple Physically Based Blinn-Phong Model in Real-Time"]
float3 Diffuse_OrenNayar( float3 DiffuseColor, float Roughness, float NoV, float NoL, float VoH )
{
	float a = Roughness * Roughness;
	float s = a;// / ( 1.29 + 0.5 * a );
	float s2 = s * s;
	float VoL = 2 * VoH * VoH - 1;		// double angle identity
	float Cosri = VoL - NoV * NoL;
	float C1 = 1 - 0.5 * s2 / (s2 + 0.33);
	float C2 = 0.45 * s2 / (s2 + 0.09) * Cosri * ( Cosri >= 0 ? rcp( max( NoL, NoV ) ) : 1 );
	return DiffuseColor / PI * ( C1 + C2 ) * ( 1 + Roughness * 0.5 );
}

// [Gotanda 2014, "Designing Reflectance Models for New Consoles"]
float3 Diffuse_Gotanda( float3 DiffuseColor, float Roughness, float NoV, float NoL, float VoH )
{
	float a = Roughness * Roughness;
	float a2 = a * a;
	float F0 = 0.04;
	float VoL = 2 * VoH * VoH - 1;		// double angle identity
	float Cosri = VoL - NoV * NoL;
#if 1
	float a2_13 = a2 + 1.36053;
	float Fr = ( 1 - ( 0.542026*a2 + 0.303573*a ) / a2_13 ) * ( 1 - pow( 1 - NoV, 5 - 4*a2 ) / a2_13 ) * ( ( -0.733996*a2*a + 1.50912*a2 - 1.16402*a ) * pow( 1 - NoV, 1 + rcp(39*a2*a2+1) ) + 1 );
	//float Fr = ( 1 - 0.36 * a ) * ( 1 - pow( 1 - NoV, 5 - 4*a2 ) / a2_13 ) * ( -2.5 * Roughness * ( 1 - NoV ) + 1 );
	float Lm = ( max( 1 - 2*a, 0 ) * ( 1 - Pow5( 1 - NoL ) ) + min( 2*a, 1 ) ) * ( 1 - 0.5*a * (NoL - 1) ) * NoL;
	float Vd = ( a2 / ( (a2 + 0.09) * (1.31072 + 0.995584 * NoV) ) ) * ( 1 - pow( 1 - NoL, ( 1 - 0.3726732 * NoV * NoV ) / ( 0.188566 + 0.38841 * NoV ) ) );
	float Bp = Cosri < 0 ? 1.4 * NoV * NoL * Cosri : Cosri;
	float Lr = (21.0 / 20.0) * (1 - F0) * ( Fr * Lm + Vd + Bp );
	return DiffuseColor / PI * Lr;
#else
	float a2_13 = a2 + 1.36053;
	float Fr = ( 1 - ( 0.542026*a2 + 0.303573*a ) / a2_13 ) * ( 1 - pow( 1 - NoV, 5 - 4*a2 ) / a2_13 ) * ( ( -0.733996*a2*a + 1.50912*a2 - 1.16402*a ) * pow( 1 - NoV, 1 + rcp(39*a2*a2+1) ) + 1 );
	float Lm = ( max( 1 - 2*a, 0 ) * ( 1 - Pow5( 1 - NoL ) ) + min( 2*a, 1 ) ) * ( 1 - 0.5*a + 0.5*a * NoL );
	float Vd = ( a2 / ( (a2 + 0.09) * (1.31072 + 0.995584 * NoV) ) ) * ( 1 - pow( 1 - NoL, ( 1 - 0.3726732 * NoV * NoV ) / ( 0.188566 + 0.38841 * NoV ) ) );
	float Bp = Cosri < 0 ? 1.4 * NoV * Cosri : Cosri / max( NoL, 1e-8 );
	float Lr = (21.0 / 20.0) * (1 - F0) * ( Fr * Lm + Vd + Bp );
	return DiffuseColor / PI * Lr;
#endif
}

// [Blinn 1977, "Models of light reflection for computer synthesized pictures"]
float D_Blinn( float Roughness, float NoH )
{
	float a = Roughness * Roughness;
	float a2 = a * a;
	float n = 2 / a2 - 2;
	return (n+2) / (2*PI) * PhongShadingPow( NoH, n );		// 1 mad, 1 exp, 1 mul, 1 log
}

// [Beckmann 1963, "The scattering of electromagnetic waves from rough surfaces"]
float D_Beckmann( float Roughness, float NoH )
{
	float a = Roughness * Roughness;
	float a2 = a * a;
	float NoH2 = NoH * NoH;
	return exp( (NoH2 - 1) / (a2 * NoH2) ) / ( PI * a2 * NoH2 * NoH2 );
}

// GGX / Trowbridge-Reitz
// [Walter et al. 2007, "Microfacet models for refraction through rough surfaces"]
float D_GGX( float Roughness, float NoH )
{
	float a2 = Roughness * Roughness;
	// float a2 = a * a;
	float d = ( NoH * a2 - NoH ) * NoH + 1;	// 2 mad
	return /*INV_PI */ a2 / (d * d + 1e-7f);
	//return a2 / ( PI*d*d );					// 4 mul, 1 rcp
}

// Anisotropic GGX
// [Burley 2012, "Physically-Based Shading at Disney"]
float D_GGXaniso( float RoughnessX, float RoughnessY, float NoH, float3 H, float3 X, float3 Y )
{
	float ax = RoughnessX * RoughnessX;
	float ay = RoughnessY * RoughnessY;
	float XoH = dot( X, H );
	float YoH = dot( Y, H );
	float d = XoH*XoH / (ax*ax) + YoH*YoH / (ay*ay) + NoH*NoH;
	return 1 / ( PI * ax*ay * d*d );
}

float Vis_Implicit()
{
	return 0.25;
}

// [Neumann et al. 1999, "Compact metallic reflectance models"]
float Vis_Neumann( float NoV, float NoL )
{
	return 1 / ( 4 * max( NoL, NoV ) );
}

// [Kelemen 2001, "A microfacet based coupled specular-matte brdf model with importance sampling"]
float Vis_Kelemen( float VoH )
{
	// constant to prevent NaN
	return rcp( 4 * VoH * VoH + 1e-5);
}

// Tuned to match behavior of Vis_Smith
// [Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"]
float Vis_Schlick( float Roughness, float NoV, float NoL )
{
	float k = Square( Roughness ) * 0.5;
	float Vis_SchlickV = NoV * (1 - k) + k;
	float Vis_SchlickL = NoL * (1 - k) + k;
	return 0.25 / ( Vis_SchlickV * Vis_SchlickL );
}

// Smith term for GGX
// [Smith 1967, "Geometrical shadowing of a random rough surface"]
float Vis_Smith( float Roughness, float NoV, float NoL )
{
	float a = Square( Roughness );
	float a2 = a*a;

	float Vis_SmithV = NoV + sqrt( NoV * (NoV - NoV * a2) + a2 );
	float Vis_SmithL = NoL + sqrt( NoL * (NoL - NoL * a2) + a2 );
	return rcp( Vis_SmithV * Vis_SmithL );
}

// Appoximation of joint Smith term for GGX
// [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
float Vis_SmithJointApprox( float Roughness, float NoV, float NoL )
{
	float a = Roughness;//Square( Roughness );
	float Vis_SmithV = NoL * ( NoV * ( 1 - a ) + a );
	float Vis_SmithL = NoV * ( NoL * ( 1 - a ) + a );
	// Note: will generate NaNs with Roughness = 0.  MinRoughness is used to prevent this
	return 0.5 * rcp( Vis_SmithV + Vis_SmithL + 1e-5f );
}

float3 F_None( float3 SpecularColor )
{
	return SpecularColor;
}

// [Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"]
float3 F_Schlick( float3 SpecularColor, float VoH )
{
	float Fc = Pow5( 1 - VoH );					// 1 sub, 3 mul
	//return Fc + (1 - Fc) * SpecularColor;		// 1 add, 3 mad
	
	// Anything less than 2% is physically impossible and is instead considered to be shadowing
	return saturate( 50.0 * SpecularColor.g ) * Fc + (1 - Fc) * SpecularColor;
	
}

float3 F_Fresnel( float3 SpecularColor, float VoH )
{
	float3 SpecularColorSqrt = sqrt( clamp( float3(0, 0, 0), float3(0.99, 0.99, 0.99), SpecularColor ) );
	float3 n = ( 1 + SpecularColorSqrt ) / ( 1 - SpecularColorSqrt );
	float3 g = sqrt( n*n + VoH*VoH - 1 );
	return 0.5 * Square( (g - VoH) / (g + VoH) ) * ( 1 + Square( ((g+VoH)*VoH - 1) / ((g-VoH)*VoH + 1) ) );
}

inline half3 FresnelLerp (half3 F0, half3 F90, half cosA)
{
    half t = Pow5 (1 - cosA);   // ala Schlick interpoliation
    return lerp (F0, F90, t);
}

//---------------
// EnvBRDF
//---------------

//Texture2D		PreIntegratedGF;
//SamplerState	PreIntegratedGFSampler;
//
//half3 EnvBRDF( half3 SpecularColor, half Roughness, half NoV )
//{
//	// Importance sampled preintegrated G * F
//	float2 AB = Texture2DSampleLevel( PreIntegratedGF, PreIntegratedGFSampler, float2( NoV, Roughness ), 0 ).rg;
//
//	// Anything less than 2% is physically impossible and is instead considered to be shadowing 
//	float3 GF = SpecularColor * AB.x + saturate( 50.0 * SpecularColor.g ) * AB.y;
//	return GF;
//}

half3 EnvBRDFApprox( half3 SpecularColor, half Roughness, half NoV )
{
	// [ Lazarov 2013, "Getting More Physical in Call of Duty: Black Ops II" ]
	// Adaptation to fit our G term.
	const half4 c0 = { -1, -0.0275, -0.572, 0.022 };
	const half4 c1 = { 1, 0.0425, 1.04, -0.04 };
	half4 r = Roughness * c0 + c1;
	half a004 = min( r.x * r.x, exp2( -9.28 * NoV ) ) * r.x + r.y;
	half2 AB = half2( -1.04, 1.04 ) * a004 + r.zw;

	// Anything less than 2% is physically impossible and is instead considered to be shadowing
	// Note: this is needed for the 'specular' show flag to work, since it uses a SpecularColor of 0
	AB.y *= saturate( 50.0 * SpecularColor.g );

	return SpecularColor * AB.x + AB.y;
}

half EnvBRDFApproxNonmetal( half Roughness, half NoV )
{
	// Same as EnvBRDFApprox( 0.04, Roughness, NoV )
	const half2 c0 = { -1, -0.0275 };
	const half2 c1 = { 1, 0.0425 };
	half2 r = Roughness * c0 + c1;
	return min( r.x * r.x, exp2( -9.28 * NoV ) ) * r.x + r.y;
}


float D_InvBlinn( float Roughness, float NoH )
{
	float m = Roughness * Roughness;
	float m2 = m * m;
	float A = 4;
	float Cos2h = NoH * NoH;
	float Sin2h = 1 - Cos2h;
	//return rcp( PI * (1 + A*m2) ) * ( 1 + A * ClampedPow( Sin2h, 1 / m2 - 1 ) );
	return rcp( PI * (1 + A*m2) ) * ( 1 + A * exp( -Cos2h / m2 ) );
}

float D_InvBeckmann( float Roughness, float NoH )
{
	float m = Roughness * Roughness;
	float m2 = m * m;
	float A = 4;
	float Cos2h = NoH * NoH;
	float Sin2h = 1 - Cos2h;
	float Sin4h = Sin2h * Sin2h;
	return rcp( PI * (1 + A*m2) * Sin4h ) * ( Sin4h + A * exp( -Cos2h / (m2 * Sin2h) ) );
}

float D_InvGGX( float Roughness, float NoH )
{
	float a = Roughness * Roughness;
	float a2 = a * a;
	float A = 4;
	float d = ( NoH - a2 * NoH ) * NoH + a2;
	return rcp( PI * (1 + A*a2) ) * ( 1 + 4 * a2*a2 / ( d*d ) );
}

float Vis_Cloth( float NoV, float NoL )
{
	return rcp( 4 * ( NoL + NoV - NoL * NoV ) );
}

#define MEDIUMP_FLT_MAX    65504.0
#define MEDIUMP_FLT_MIN    0.00006103515625

#define saturateMediump(x) min(x, MEDIUMP_FLT_MAX)

FLOAT CalcSpecular(FLOAT Roughness, FLOAT NoH, FLOAT3 H, FLOAT3 N)
{
	//GGX_Mobile
	FLOAT3 NxH = cross(N, H);
	FLOAT OneMinusNoHSqr = dot(NxH, NxH);

	FLOAT a = max(0.01,Roughness * Roughness);
	FLOAT n = NoH * a;
	FLOAT p = a / (OneMinusNoHSqr + n * n);
	FLOAT d = p * p;

	return (Roughness*_MetallicScale + _MetallicScale) * saturateMediump(d);
}

FLOAT CalcSpecular(FLOAT a, FLOAT NoH, FLOAT3 H, FLOAT3 N,FLOAT param)
{
	//GGX_Mobile
	FLOAT3 NxH = cross(N, H);
	FLOAT OneMinusNoHSqr = dot(NxH, NxH);

	// FLOAT a = max(0.01,Roughness * Roughness);
	FLOAT n = NoH * a;
	FLOAT p = a / (OneMinusNoHSqr + n * n);
	FLOAT d = p * p;

	return param * saturateMediump(d);
}

FLOAT3 Diffuse_Burley(FLOAT Roughness, FLOAT NoL, FLOAT VoH,FLOAT forawrdCalc )
{
	FLOAT FD90 = 0.5 + 2 * VoH * VoH * Roughness;
	FLOAT FdV = 1 + (FD90 - 1) * forawrdCalc;
	FLOAT FdL = 1 + (FD90 - 1) * Pow5( 1 - NoL );
	return FdV * FdL;//hack divide PI
}

FLOAT3 SpecularGGX( FLOAT Roughness, FLightingData LightingData, FLOAT NoL)
{
	FLOAT a2 = Square(Roughness);//Pow4( Roughness );
	// FLOAT Energy = EnergyNormalization( a2, Context.VoH, AreaLight );
	a2 = max(a2, FLOAT(0.002));
	// Generalized microfacet specular
	FLOAT D = D_GGX( a2, LightingData.NdotH );
	FLOAT Vis = Vis_SmithJointApprox( a2, LightingData.NdotC, NoL );
	FLOAT3 F = F_Schlick( LightingData.SpecularColor, LightingData.VdotH );
	FLOAT specularTerm = Vis* D;
	specularTerm = min(5, specularTerm);
	return specularTerm*F;
}
#endif // __BRDF_COMMON__