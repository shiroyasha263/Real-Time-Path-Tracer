#include "DisplayWindow.h"

int main(int argc, char** argv) {
    try {
        DisplayWindow* window = new DisplayWindow();
        window->run();

    }
    catch (std::runtime_error& e) {
        std::cout << TERMINAL_RED << "FATAL ERROR: " << e.what()
            << TERMINAL_DEFAULT << std::endl;
        exit(1);
    }
    return 0;
}