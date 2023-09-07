#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sampleConfig.h>

#include <iomanip>
#include <iostream>
#include <string>

#include "SampleRenderer.h"

int main(int argc, char** argv) {
	try {
		SampleRenderer sample;
	}
	catch (std::runtime_error& e) {
		std::cout << TERMINAL_RED
			<< "FATAL ERROR: " << e.what()
			<< TERMINAL_DEFAULT << std::endl;

		exit(1);
	}
	return 0;
}