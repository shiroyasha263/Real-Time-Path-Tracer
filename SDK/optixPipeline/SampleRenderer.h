#pragma once
#include "CUDABuffer.h"
#include "LaunchParams.h"

#define TERMINAL_RED "\033[1;31m"
#define TERMINAL_GREEN "\033[1;32m"
#define TERMINAL_DEFAULT "\033[0m"

#define PRINT(var) std::cout << #var << " = " << var << std::endl;

class SampleRenderer {
public:
	SampleRenderer();

	void Render();

	void Resize(const unsigned int width, const unsigned int height);

	void downloadPixels(uchar4* h_pixels);

	unsigned int getWidth() {
		return launchParams.width;
	}

	unsigned int getHeight() {
		return launchParams.height;
	}

	CUstream getStream() {
		return stream;
	}

	void updateParams();

protected:
	void InitOptix();

	void CreateContext();

	void CreateModule();

	void CreateRayGenPrograms();

	void CreateMissPrograms();

	void CreateHitgroupPrograms();

	void CreatePipeline();

	void buildSBT();

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

	CUDABuffer colorBuffer;
public:
	LaunchParams launchParams;
};