#include "PhotonTracer.h"

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	void* data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	void* data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	int objectID;
};

PhotonTracer::PhotonTracer() {
	InitOptix();

	std::cout << "#RTPT: Creating Optix Context ... " << std::endl;
	CreateContext();

	std::cout << "#RTPT: Creating Optix Module ..." << std::endl;
	CreateModule();

	std::cout << "#RTPT: Creating raygen programs ..." << std::endl;
	CreateRayGenPrograms();

	std::cout << "#RTPT: Creating miss programs ..." << std::endl;
	CreateMissPrograms();

	std::cout << "#RTPT: Creating hitgroup programs ..." << std::endl;
	CreateHitgroupPrograms();

	std::cout << "#RTPT: Creating Optix Pipeline ..." << std::endl;
	CreatePipeline();

	std::cout << "#RTPT: Building SBT ..." << std::endl;
	buildSBT();

	photonBeamParamsBuffer.alloc(sizeof(photonBeamParams));
	std::cout << "#RTPT: All basic things setup!" << std::endl;

	std::cout << TERMINAL_GREEN;
	std::cout << "#RTPT: Optix Successfully set up ..." << std::endl;
	std::cout << TERMINAL_DEFAULT;
}

void PhotonTracer::InitOptix() {
	std::cout << "#RTPT: Initializing optix ..." << std::endl;

	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
		throw std::runtime_error("#RTPT: no cuda capable device found");
	std::cout << "#RTPT: found " << numDevices << " Cuda devices" << std::endl;

	OPTIX_CHECK(optixInit());
	std::cout << TERMINAL_GREEN;
	std::cout << "#RTPT: Optix successfully initialized";
	std::cout << TERMINAL_DEFAULT;
}

static void context_log_cb(unsigned int level,
	const char* tag,
	const char* message,
	void*)
{
	fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

void PhotonTracer::CreateContext() {
	const int deviceID = 0;
	CUDA_CHECK(cudaSetDevice(deviceID));
	CUDA_CHECK(cudaStreamCreate(&stream));

	cudaGetDeviceProperties(&deviceProps, deviceID);
	std::cout << "#RTPT: running on device: " << deviceProps.name << std::endl;

	CUresult cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS)
		fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

void PhotonTracer::CreateModule() {
	moduleCompileOptions.maxRegisterCount = 50;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

	pipelineCompileOptions = {};
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

	pipelineLinkOptions.maxTraceDepth = 2;

	size_t      inputSize = 0;
	const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "PhotonTracer.cu", inputSize);

	char log[2048];
	size_t sizeof_log = sizeof(log);

	OPTIX_CHECK(optixModuleCreate(optixContext,
		&moduleCompileOptions,
		&pipelineCompileOptions,
		input,
		inputSize,
		log, &sizeof_log,
		&module
	));
	if (sizeof_log > 1) PRINT(log);
}

void PhotonTracer::CreateRayGenPrograms() {
	raygenPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = module;
	pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

	// OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
		&pgDesc,
		1,
		&pgOptions,
		log, &sizeof_log,
		&raygenPGs[0]
	));
	if (sizeof_log > 1) PRINT(log);
}

void PhotonTracer::CreateMissPrograms() {
	missPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDesc.miss.module = module;
	pgDesc.miss.entryFunctionName = "__miss__radiance";

	// OptixProgramGroup raypg;
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
		&pgDesc,
		1,
		&pgOptions,
		log, &sizeof_log,
		&missPGs[0]
	));
	if (sizeof_log > 1) PRINT(log);
}

void PhotonTracer::CreateHitgroupPrograms() {
	hitgroupPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleCH = module;
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDesc.hitgroup.moduleAH = module;
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext,
		&pgDesc,
		1,
		&pgOptions,
		log, &sizeof_log,
		&hitgroupPGs[0]
	));
	if (sizeof_log > 1) PRINT(log);
}

void PhotonTracer::CreatePipeline() {
	std::vector<OptixProgramGroup> programGroups;
	for (auto pg : raygenPGs)
		programGroups.push_back(pg);
	for (auto pg : missPGs)
		programGroups.push_back(pg);
	for (auto pg : hitgroupPGs)
		programGroups.push_back(pg);

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(optixContext,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		programGroups.data(),
		(int)programGroups.size(),
		log, &sizeof_log,
		&pipeline
	));
	if (sizeof_log > 1) PRINT(log);

	OPTIX_CHECK(optixPipelineSetStackSize
	(/* [in] The pipeline to configure the stack size for */
		pipeline,
		/* [in] The direct stack size requirement for direct
		   callables invoked from IS or AH. */
		2 * 1024,
		/* [in] The direct stack size requirement for direct
		   callables invoked from RG, MS, or CH.  */
		2 * 1024,
		/* [in] The continuation stack requirement. */
		2 * 1024,
		/* [in] The maximum depth of a traversable graph
		   passed to trace. */
		1));
	if (sizeof_log > 1) PRINT(log);
}

void PhotonTracer::buildSBT() {
	std::vector<RaygenRecord> raygenRecords;
	for (int i = 0; i < raygenPGs.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
		rec.data = nullptr;
		raygenRecords.push_back(rec);
	}
	raygenRecordsBuffer.alloc_and_upload(raygenRecords);
	sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

	std::vector<MissRecord> missRecords;
	for (int i = 0; i < missPGs.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
		rec.data = nullptr;
		missRecords.push_back(rec);
	}
	missRecordsBuffer.alloc_and_upload(missRecords);
	sbt.missRecordBase = missRecordsBuffer.d_pointer();
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount = (int)missRecords.size();

	int numObjects = 1;
	std::vector<HitgroupRecord> hitgroupRecords;
	for (int i = 0; i < numObjects; i++) {
		int objectType = 0;
		HitgroupRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
		rec.objectID = i;
		hitgroupRecords.push_back(rec);
	}
	hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void PhotonTracer::Render() {
	if (photonBeamParams.maxBeams == 0) return;

	photonBeamParamsBuffer.upload(&photonBeamParams, 1);

	OPTIX_CHECK(optixLaunch(
		pipeline, stream,
		photonBeamParamsBuffer.d_pointer(),
		photonBeamParamsBuffer.sizeInBytes,
		&sbt,
		photonBeamParams.maxBeams,
		1,
		1));
	CUDA_SYNC_CHECK();
}

void PhotonTracer::Resize(int beamCount, int bounceCount, float mediumProp) {
	if (beamCount == 0 || bounceCount == 0) return;
	photonBeamParams.materialProp = mediumProp;
	beams.resize(beamCount * bounceCount * sizeof(PhotonBeam));
	photonBeamParams.maxBeams = beamCount;
	photonBeamParams.maxBounce = bounceCount;
	photonBeamParams.beams = (PhotonBeam*)beams.d_pointer();
}

void PhotonTracer::GetBeams(PhotonBeam h_beams[]) {
	beams.download(h_beams, photonBeamParams.maxBeams * photonBeamParams.maxBounce);
}