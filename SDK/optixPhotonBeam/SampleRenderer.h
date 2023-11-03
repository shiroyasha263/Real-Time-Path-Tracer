#pragma once
#include "CUDABuffer.h"
#include "LaunchParams.h"

#define TERMINAL_RED "\033[1;31m"
#define TERMINAL_GREEN "\033[1;32m"
#define TERMINAL_DEFAULT "\033[0m"

#define PRINT(var) std::cout << #var << " = " << var << std::endl;

struct Quad {
	std::vector<float3> vertex;
	std::vector<int3> index;
	float transmittance;
	float3 start;
};

class SampleRenderer {
public:
	SampleRenderer(const std::vector<Quad> &quads);

	void Render();

	void Resize(const unsigned int width, const unsigned int height);

	void downloadPixels(uchar4* h_pixels);

	CUstream getStream() {
		return stream;
	}

	void updateParams(LaunchParams launchParams);

protected:
	void InitOptix();

	void CreateContext();

	void CreateModule();

	void CreateRayGenPrograms();

	void CreateMissPrograms();

	void CreateHitgroupPrograms();

	void CreatePipeline();

	void buildSBT();

	OptixTraversableHandle buildAccel();

protected:
	CUcontext cudaContext;
	CUstream stream;
	cudaDeviceProp deviceProps;

	OptixDeviceContext optixContext;

	OptixPipeline pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions pipelineLinkOptions = {};

	OptixModule module;
	OptixModuleCompileOptions moduleCompileOptions = {};

	std::vector<OptixProgramGroup> raygenPGs;
	CUDABuffer raygenRecordsBuffer;
	std::vector<OptixProgramGroup> missPGs;
	CUDABuffer missRecordsBuffer;
	std::vector<OptixProgramGroup> hitgroupPGs;
	CUDABuffer hitgroupRecordsBuffer;
	OptixShaderBindingTable sbt = {};

	CUDABuffer launchParamsBuffer;
	LaunchParams launchParams;

	CUDABuffer colorBuffer;

	std::vector<Quad> quads;
	std::vector<CUDABuffer> vertexBuffer;
	std::vector<CUDABuffer> indexBuffer;
	CUDABuffer asBuffer;
};