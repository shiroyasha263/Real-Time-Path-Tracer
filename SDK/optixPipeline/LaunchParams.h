#pragma once
#include <vector_types.h>

struct LaunchParams {
	int frameID{ 0 };
	uchar4* image;
	unsigned int width, height;
};