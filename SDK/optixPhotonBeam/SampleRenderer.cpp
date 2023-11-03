#include "SampleRenderer.h"

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
	QuadSBTData data;
};

SampleRenderer::SampleRenderer(const std::vector<Quad> &quads)
	:quads(quads) {
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
	
	launchParams.handle = buildAccel();

	std::cout << "#RTPT: Creating Optix Pipeline ..." << std::endl;
	CreatePipeline();
	
	std::cout << "#RTPT: Building SBT ..." << std::endl;
	buildSBT();
	
	launchParamsBuffer.alloc(sizeof(launchParams));
	std::cout << "#RTPT: All basic things setup!" << std::endl;
	
	std::cout << TERMINAL_GREEN;
	std::cout << "#RTPT: Optix Successfully set up ..." << std::endl;
	std::cout << TERMINAL_DEFAULT;
}

OptixTraversableHandle SampleRenderer::buildAccel() {
	vertexBuffer.resize(quads.size());
	indexBuffer.resize(quads.size());

	OptixTraversableHandle asHandle{ 0 };

	// ==================================================================
	// triangle inputs
	// ==================================================================
	std::vector<OptixBuildInput> triangleInput(quads.size());
	std::vector<CUdeviceptr> d_vertices(quads.size());
	std::vector<CUdeviceptr> d_indices(quads.size());
	std::vector<uint32_t> triangleInputFlags(quads.size());

	for (int meshID = 0; meshID < quads.size(); meshID++) {
		// upload the model to the device: the builder
		Quad& model = quads[meshID];
		vertexBuffer[meshID].alloc_and_upload(model.vertex);
		indexBuffer[meshID].alloc_and_upload(model.index);

		triangleInput[meshID] = {};
		triangleInput[meshID].type
			= OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// create local variables, because we need a *pointer* to the
		// device pointers
		d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
		d_indices[meshID] = indexBuffer[meshID].d_pointer();

		triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(float3);
		triangleInput[meshID].triangleArray.numVertices = (int)model.vertex.size();
		triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

		triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(int3);
		triangleInput[meshID].triangleArray.numIndexTriplets = (int)model.index.size();
		triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

		triangleInputFlags[meshID] = 0;

		// in this example we have one SBT entry, and no per-primitive
		// materials:
		triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
		triangleInput[meshID].triangleArray.numSbtRecords = 1;
		triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}
	// ==================================================================
	// BLAS setup
	// ==================================================================

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
		| OPTIX_BUILD_FLAG_ALLOW_COMPACTION
		;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage
	(optixContext,
		&accelOptions,
		triangleInput.data(),
		(int)quads.size(),  // num_build_inputs
		&blasBufferSizes
	));

	// ==================================================================
	// prepare compaction
	// ==================================================================

	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.d_pointer();

	// ==================================================================
	// execute build (main stage)
	// ==================================================================

	CUDABuffer tempBuffer;
	tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

	CUDABuffer outputBuffer;
	outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

	OPTIX_CHECK(optixAccelBuild(optixContext,
		/* stream */0,
		&accelOptions,
		triangleInput.data(),
		(int)quads.size(),
		tempBuffer.d_pointer(),
		tempBuffer.sizeInBytes,

		outputBuffer.d_pointer(),
		outputBuffer.sizeInBytes,

		&asHandle,

		&emitDesc, 1
	));
	CUDA_SYNC_CHECK();

	// ==================================================================
	// perform compaction
	// ==================================================================
	uint64_t compactedSize;
	compactedSizeBuffer.download(&compactedSize, 1);

	asBuffer.alloc(compactedSize);
	OPTIX_CHECK(optixAccelCompact(optixContext,
		/*stream:*/0,
		asHandle,
		asBuffer.d_pointer(),
		asBuffer.sizeInBytes,
		&asHandle));
	CUDA_SYNC_CHECK();

	// ==================================================================
	// aaaaaand .... clean up
	// ==================================================================
	outputBuffer.free(); // << the UNcompacted, temporary output buffer
	tempBuffer.free();
	compactedSizeBuffer.free();

	return asHandle;
}

void SampleRenderer::InitOptix() {
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

void SampleRenderer::CreateContext() {
	const int deviceID = 0;
	CUDA_CHECK(cudaSetDevice(deviceID));
	CUDA_CHECK(cudaStreamCreate(&stream));

	cudaGetDeviceProperties(&deviceProps, deviceID);
	std::cout << "#RTPT: running on device: " << deviceProps.name << std::endl;

	CUresult cuRes = cuCtxGetCurrent(&cudaContext);
	if(cuRes != CUDA_SUCCESS)
		fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

void SampleRenderer::CreateModule() {
	moduleCompileOptions.maxRegisterCount = 50;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

	pipelineCompileOptions = {};
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

	pipelineLinkOptions.maxTraceDepth = 2;
	
	size_t      inputSize = 0;
	const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "devicePrograms.cu", inputSize);

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

void SampleRenderer::CreateRayGenPrograms() {
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

void SampleRenderer::CreateMissPrograms() {
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

void SampleRenderer::CreateHitgroupPrograms() {
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

void SampleRenderer::CreatePipeline() {
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

void SampleRenderer::buildSBT() {
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

	int numObjects = (int)quads.size();
	std::vector<HitgroupRecord> hitgroupRecords;
	for (int meshID = 0; meshID < numObjects; meshID++) {
		HitgroupRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));
		rec.data.transmittance = quads[meshID].transmittance;
		rec.data.vertex = (float3*)vertexBuffer[meshID].d_pointer();
		rec.data.index = (int3*)indexBuffer[meshID].d_pointer();
		rec.data.start = quads[meshID].start;
		hitgroupRecords.push_back(rec);
	}
	hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

void SampleRenderer::Render() {
	if (launchParams.width == 0 || launchParams.height == 0) return;

	launchParamsBuffer.upload(&launchParams, 1);
	launchParams.frameID++;

	OPTIX_CHECK(optixLaunch(
							pipeline, stream,
							launchParamsBuffer.d_pointer(),
							launchParamsBuffer.sizeInBytes,
							&sbt,
							launchParams.width,
							launchParams.height,
							1));
	CUDA_SYNC_CHECK();
}

void SampleRenderer::updateParams(LaunchParams params) {
	if (params.width == 0 || params.height == 0) return;
	launchParams.mediumProp = params.mediumProp;
	colorBuffer.resize(params.width * params.height * sizeof(uchar4));
	launchParams.width = params.width;
	launchParams.height = params.height;
	launchParams.image = (uchar4*)colorBuffer.d_pointer();
}

void SampleRenderer::downloadPixels(uchar4* h_pixels) {
	colorBuffer.download(h_pixels, launchParams.width * launchParams.height);
}