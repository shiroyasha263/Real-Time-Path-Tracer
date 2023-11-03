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

        int maxBeams = 20000;
        int maxBounce = 3;
        float mediumProp = .8f;
        sample.Resize(maxBeams, maxBounce, mediumProp);
        sample.Render();

        std::vector<PhotonBeam> pBeams(maxBeams * maxBounce);
        sample.GetBeams(pBeams.data());

        float thickness = 0.15f;
        float3 eye = make_float3(0, 0, -5);
        std::vector<Quad> quads(pBeams.size());

        for (int i = 0; i < pBeams.size(); i++) {
            float3 dir = normalize(cross(pBeams[i].start - eye, pBeams[i].end - pBeams[i].start));
            quads[i].vertex.push_back(pBeams[i].start + thickness * dir / 2.0f);
            quads[i].vertex.push_back(pBeams[i].start - thickness * dir / 2.0f);
            quads[i].vertex.push_back(pBeams[i].end   + thickness * dir / 2.0f);
            quads[i].vertex.push_back(pBeams[i].end   - thickness * dir / 2.0f);
            quads[i].index.push_back(make_int3(0, 1, 2));
            quads[i].index.push_back(make_int3(3, 2, 1));
            quads[i].transmittance = pBeams[i].transmittance;
            quads[i].start = pBeams[i].start;
        }

        DisplayWindow* window = new DisplayWindow(quads, mediumProp);
        window->run();

        /*for (int i = 0; i < pBeams.size(); i++)
        {
            std::cout << TERMINAL_GREEN
                << std::endl
                << "Start of beam " << i << ": X: " << pBeams[i].start.x << ", Y : " << pBeams[i].start.y << ", Z : " << pBeams[i].start.z
                << std::endl
                << "End of beam " << i << ": X: " << pBeams[i].end.x << ", Y: " << pBeams[i].end.y << ", Z: " << pBeams[i].end.z
                << TERMINAL_DEFAULT
                << std::endl;
        }*/
    }
    catch (std::runtime_error& e) {
        std::cout << TERMINAL_RED << "FATAL ERROR: " << e.what()
            << TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}