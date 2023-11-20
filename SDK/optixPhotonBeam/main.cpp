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

        int maxBeams = 100000;
        int maxBounce = 1;
        float mediumProp = 1.2f;
        sample.Resize(maxBeams, maxBounce, mediumProp);
        sample.Render();

        std::vector<PhotonBeam> pBeams(maxBeams * maxBounce);
        sample.GetBeams(pBeams.data());

        float thickness = 0.15f;
        size_t breakSize = 1;
        float3 eye = make_float3(0, 0, -4);
        std::vector<Quad> quads(pBeams.size() * breakSize);

        for (int i = 0; i < pBeams.size(); i++) {
            float3 widthDir = normalize(cross(pBeams[i].start - eye, pBeams[i].end - pBeams[i].start));
            float3 lengthDir = pBeams[i].end - pBeams[i].start;
            
            for (int j = 0; j < breakSize; j++) {
                quads[i * breakSize + j].vertex.push_back(pBeams[i].start + j * lengthDir / breakSize + thickness * widthDir / 2.0f);
                quads[i * breakSize + j].vertex.push_back(pBeams[i].start + j * lengthDir / breakSize - thickness * widthDir / 2.0f);
                quads[i * breakSize + j].vertex.push_back(pBeams[i].start + (j + 1) * lengthDir / breakSize + thickness * widthDir / 2.0f);
                quads[i * breakSize + j].vertex.push_back(pBeams[i].start + (j + 1) * lengthDir / breakSize - thickness * widthDir / 2.0f);
                quads[i * breakSize + j].index.push_back(make_int3(0, 1, 2));
                quads[i * breakSize + j].index.push_back(make_int3(3, 2, 1));
                quads[i * breakSize + j].transmittance = pBeams[i].transmittance;
                quads[i * breakSize + j].start = pBeams[i].start;
            }
        }

        DisplayWindow* window = new DisplayWindow(quads, mediumProp);
        window->run();
    }
    catch (std::runtime_error& e) {
        std::cout << TERMINAL_RED << "FATAL ERROR: " << e.what()
            << TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}