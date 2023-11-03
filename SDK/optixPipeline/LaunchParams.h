#pragma once
#include <vector_types.h>

struct LaunchParams {
	unsigned int frameID{ 0 };
	float4* accum_image;
	uchar4* image;
	unsigned int width, height;
	unsigned int samples_per_launch;

	float eye;
	float3 U;
	float3 V;
	float3 W;

	//Make Light List
	OptixTraversableHandle handle;
};