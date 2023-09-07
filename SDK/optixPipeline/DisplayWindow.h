#pragma once

#include <glad/glad.h>  // Needs to be included before gl_interop


#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>

#include "SampleRenderer.h"

struct DisplayWindow {
	DisplayWindow();
	~DisplayWindow();


    /*! put pixels on the screen ... */
    void draw(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display);

    /*! callback that window got resized */
    void resize(unsigned int width, unsigned int height) { 
        /* empty - to be subclassed by user */
        sample.Resize(width, height);
    }

    void key(int key, int mods)
    {}

    /*! callback that window got resized */
    void mouseMotion(const int2& newPos)
    {}

    /*! callback that window got resized */
    void mouseButton(int button, int action, int mods)
    {}

    inline int2 getMousePos() const
    {
        double x, y;
        glfwGetCursorPos(handle, &x, &y);
        return make_int2((int)x, (int)y);
    }

    /*! re-render the frame - typically part of draw(), but we keep
      this a separate function so render() can focus on optix
      rendering, and now have to deal with opengl pixel copies
      etc */
    void render();

    /*! opens the actual window, and runs the window's events to
      completion. This function will only return once the window
      gets closed */
    void run();

    /*! the glfw window handle */
    GLFWwindow* handle{ nullptr };
    SampleRenderer sample;
};