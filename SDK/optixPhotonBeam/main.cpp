#include "PhotonTracer.h"
#include "DisplayWindow.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sampleConfig.h>

#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        PhotonTracer sample;

        int maxBeams = 10000;
        int maxBounce = 1;
        float mediumProp = 2.f;
        sample.Resize(maxBeams, maxBounce, mediumProp);
        sample.Render();

        std::vector<PhotonBeam> pBeams(maxBeams * maxBounce);
        sample.GetBeams(pBeams.data());

        float thickness = 0.15f;
        size_t breakSize = 1;
        float3 eye = make_float3(0, 0, -4);
        std::vector<Quad> quads(pBeams.size() * breakSize);

        DisplayWindow* window = new DisplayWindow(pBeams, mediumProp);
        window->run();
    }
    catch (std::runtime_error& e) {
        std::cout << TERMINAL_RED << "FATAL ERROR: " << e.what()
            << TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}