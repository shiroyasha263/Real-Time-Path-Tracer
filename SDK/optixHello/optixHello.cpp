#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sampleConfig.h>

#include "optixHello.h"

#include <iomanip>
#include <iostream>
#include <string>

#define TERMINAL_RED "\033[1;31m"
#define TERMINAL_GREEN "\033[1;32m"
#define TERMINAL_DEFAULT "\033[0m"

void OptiXInit() {
	// Clears the devices
	cudaFree(0);
	int numDevices;
	// Gets the number of available devices
	cudaGetDeviceCount(&numDevices);
	// Shows the number of devices
	if (numDevices == 0)
		throw std::runtime_error("#RTPT: no CUDA capable devices found!");
	std::cout << "#RTPT: Found " << numDevices << " CUDA Devices" << std::endl;

	// Initialize optix
	OPTIX_CHECK(optixInit());
}

int main(int argc, char** argv) {
	try {
		std::cout << "#RTPT: Initializing OptiX" << std::endl;

		OptiXInit();

		std::cout << TERMINAL_GREEN 
			      << "#RTPT: Successfully initialized optix"
			      << TERMINAL_DEFAULT << std::endl;

		std::cout << "#RTPT: Clean exit" << std::endl;
	}
	catch (std::runtime_error& e) {
		std::cout << TERMINAL_RED
			      << "FATAL ERROR: " << e.what() 
				  << TERMINAL_DEFAULT << std::endl;

		exit(1);
	}
	return 0;
}