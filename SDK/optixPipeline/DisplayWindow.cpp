#include "DisplayWindow.h"
#include <vector_types.h>
#include <sutil/vec_math.h>

bool minimized = false;

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    DisplayWindow* display = static_cast<DisplayWindow*>(glfwGetWindowUserPointer(window));
    display->mouseButtonCB(button, action, xpos, ypos);
}

void DisplayWindow::mouseButtonCB(int button, int action, double xpos, double ypos) {
    if (action == GLFW_PRESS)
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    DisplayWindow* display = static_cast<DisplayWindow*>(glfwGetWindowUserPointer(window));
    display->cursorPosCB(xpos, ypos);
}

void DisplayWindow::cursorPosCB(double xpos, double ypos) {
    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), launchParams.width, launchParams.height);
        camera_changed = true;
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), launchParams.width, launchParams.height);
        camera_changed = true;
    }
}


static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    // Keep rendering at the current resolution when the window is minimized.
    if (minimized)
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize(res_x, res_y);

    DisplayWindow* display = static_cast<DisplayWindow*>(glfwGetWindowUserPointer(window));
    display->windowSizeCB(res_x, res_y);
}

void DisplayWindow::windowSizeCB(unsigned int res_x, unsigned int res_y) {
    resize(res_x, res_y);
    camera_changed = true;
    resize_dirty = true;
}


static void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
    minimized = (iconified > 0);
}


static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, true);
        }
    }
    else if (key == GLFW_KEY_G)
    {
        // toggle UI draw
    }
}


static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    DisplayWindow* display = static_cast<DisplayWindow*>(glfwGetWindowUserPointer(window));
    display->scrollCB(yscroll);
}

void DisplayWindow::scrollCB(double yscroll) {
    if (trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}

void DisplayWindow::initCameraState()
{
    camera.setEye(make_float3(278.0f, 273.0f, -900.0f));
    camera.setLookat(make_float3(278.0f, 273.0f, 330.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFovY(35.0f);
    camera_changed = true;

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 1.0f, 0.0f)
    );
    trackball.setGimbalLock(true);
}

DisplayWindow::~DisplayWindow() {
    glfwDestroyWindow(handle);
    glfwTerminate();
}


DisplayWindow::DisplayWindow() {
    handle = sutil::initUI("optixPathTracer", 1200, 800);
    launchParams.width = 1200;
    launchParams.height = 800;
    initWinParams();
    initCameraState();
    glfwSetWindowUserPointer(handle, this);
    glfwMakeContextCurrent(handle);
    sample.updateParams(launchParams);
}


void DisplayWindow::run() {
    int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    resize(width, height);


    glfwSetMouseButtonCallback(handle, mouseButtonCallback);
    glfwSetCursorPosCallback(handle, cursorPosCallback);
    glfwSetWindowSizeCallback(handle, windowSizeCallback);
    glfwSetWindowIconifyCallback(handle, windowIconifyCallback);
    glfwSetKeyCallback(handle, keyCallback);
    glfwSetScrollCallback(handle, scrollCallback);

    sutil::CUDAOutputBuffer<uchar4> output_buffer(
        sutil::CUDAOutputBufferType::GL_INTEROP,
        launchParams.width,
        launchParams.height
    );

    output_buffer.setStream(sample.getStream());
    sutil::GLDisplay gl_display;

    std::chrono::duration<double> state_update_time(0.0);
    std::chrono::duration<double> render_time(0.0);
    std::chrono::duration<double> display_time(0.0);

    do
    {
        auto t0 = std::chrono::steady_clock::now();
        glfwPollEvents();

        output_buffer.resize(launchParams.width, launchParams.height);
        auto t1 = std::chrono::steady_clock::now();
        state_update_time += t1 - t0;
        t0 = t1;

        render();
        t1 = std::chrono::steady_clock::now();
        render_time += t1 - t0;
        t0 = t1;

        draw(output_buffer, gl_display);
        t1 = std::chrono::steady_clock::now();
        display_time += t1 - t0;

        sutil::displayStats(state_update_time, render_time, display_time);
        glfwSwapBuffers(handle);
    } while (!glfwWindowShouldClose(handle));

    CUDA_SYNC_CHECK();
    sutil::cleanupUI(handle);
}

void DisplayWindow::render() {
    sample.Render();
}

void DisplayWindow::draw(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display) {
    uchar4* result_buffer_data = output_buffer.map();
    sample.downloadPixels(result_buffer_data);
    output_buffer.unmap();
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize(handle, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}