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

        std::vector<PhotonBeam> allBeams;

        float mediumProp = 2.f;
        int maxBeams = 20;
        int maxBounce = 2;
        int maxPass = 25000;

        for (int i = 0; i < maxPass; i++) {
            sample.Resize(maxBeams, maxBounce, mediumProp, i);
            sample.Render();

            std::vector<PhotonBeam> pBeams(maxBeams * maxBounce);
            sample.GetBeams(pBeams.data());
            allBeams.insert(allBeams.end(), pBeams.begin(), pBeams.end());
        }

        DisplayWindow* window = new DisplayWindow(allBeams, mediumProp, maxBeams * maxPass);
        window->run();
    }
    catch (std::runtime_error& e) {
        std::cout << TERMINAL_RED << "FATAL ERROR: " << e.what()
            << TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}