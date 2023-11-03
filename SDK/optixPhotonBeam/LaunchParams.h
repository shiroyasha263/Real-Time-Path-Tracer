#pragma once
#include <vector_types.h>

struct QuadSBTData {
	float transmittance;
	float3* vertex;
	int3* index;
	float3 start;
};

struct PhotonBeam {
	float3 start;
	float3 end;
	float transmittance;
};

struct PhotonBeamParams {
	PhotonBeam* beams;
	int maxBeams;
	int maxBounce;
	float materialProp;
};

struct LaunchParams {
	unsigned int frameID{ 0 };
	float mediumProp;
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