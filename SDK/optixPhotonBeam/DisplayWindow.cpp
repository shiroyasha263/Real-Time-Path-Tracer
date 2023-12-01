#include "DisplayWindow.h"
#include <vector_types.h>
#include <sutil/vec_math.h>
#include <glm/glm/glm.hpp>
#include<glm/glm/gtc/matrix_transform.hpp>
#include<glm/glm/gtc/type_ptr.hpp>

#include"shaderClass.h"
#include"VAO.h"
#include"VBO.h"
#include"EBO.h"
#include"Camera.h"

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

DisplayWindow::DisplayWindow(const std::vector<PhotonBeam> &pBeams, float mediumProp, int perPassSample) {
    handle = sutil::initUI("optixPathTracer", 1200, 800);
    launchParams.width = 1200;
    launchParams.height = 800;
    launchParams.mediumProp = mediumProp;
    fluxDivider = perPassSample;
    initWinParams();
    initCameraState();
    glfwSetWindowUserPointer(handle, this);
    glfwMakeContextCurrent(handle);

    int perQuadVertex = 4;
    // vPos.x, vPos.y, vPos.z, pos/neg, dir.x, dir.y, dir.z, thickness, transmittance
    int perVertexFloat = 3 + 1 + 1 + 3 + 1 + 1;
    int perQuadFloat = perQuadVertex * perVertexFloat;
    int perQuadIndices = 6;
    const int vertexCount = pBeams.size() * perQuadFloat;
    vertices = std::vector<GLfloat>(vertexCount);
    indices = std::vector<GLuint>(pBeams.size() * 6);

    for (int i = 0; i < pBeams.size(); i++) {
        for (int j = 0; j < 4; j++) {
            //Start point
            vertices[i * perQuadFloat + j * perVertexFloat + 0] = (pBeams[i].start.x);
            vertices[i * perQuadFloat + j * perVertexFloat + 1] = (pBeams[i].start.y);
            vertices[i * perQuadFloat + j * perVertexFloat + 2] = (pBeams[i].start.z);

            //Beamside Mult
            if (j < 2)
                vertices[i * perQuadFloat + j * perVertexFloat + 3] = 0.f;
            else 
                vertices[i * perQuadFloat + j * perVertexFloat + 3] = 1.f;

            //vertices[i * perQuadFloat + j * perVertexFloat + 3] = (float)0.f;

            //Direction plus or negative
            if (j % 2 == 1)
                vertices[i * perQuadFloat + j * perVertexFloat + 4] = 1.f;
            else
                vertices[i * perQuadFloat + j * perVertexFloat + 4] = -1.f;

            //Direction of the beam
            vertices[i * perQuadFloat + j * perVertexFloat + 5] = pBeams[i].end.x;
            vertices[i * perQuadFloat + j * perVertexFloat + 6] = pBeams[i].end.y;
            vertices[i * perQuadFloat + j * perVertexFloat + 7] = pBeams[i].end.z;

            //Thickness of the beam
            float thickness = 0.1f;
            vertices[i * perQuadFloat + j * perVertexFloat + 8] = pBeams[i].thickness;

            //Transmittance of the beam
            vertices[i * perQuadFloat + j * perVertexFloat + 9] = pBeams[i].transmittance;
        }
        indices[i * perQuadIndices + 0] = (i * 4 + 0);
        indices[i * perQuadIndices + 1] = (i * 4 + 1);
        indices[i * perQuadIndices + 2] = (i * 4 + 2);
        indices[i * perQuadIndices + 3] = (i * 4 + 3);
        indices[i * perQuadIndices + 4] = (i * 4 + 2);
        indices[i * perQuadIndices + 5] = (i * 4 + 1);
    }
}

void DisplayWindow::run() {
    int width, height;
    glfwGetFramebufferSize(handle, &width, &height);
    //resize(width, height);
    //glfwSetMouseButtonCallback(handle, mouseButtonCallback);
    //glfwSetCursorPosCallback(handle, cursorPosCallback);
    //glfwSetWindowSizeCallback(handle, windowSizeCallback);
    //glfwSetWindowIconifyCallback(handle, windowIconifyCallback);
    //glfwSetKeyCallback(handle, keyCallback);
    //glfwSetScrollCallback(handle, scrollCallback);

    gladLoadGL();
    glViewport(0, 0, launchParams.width, launchParams.height);


    Shader shaderProgram("default.vert", "default.frag");
    VAO VAO1;
    VAO1.Bind();

    VBO VBO1(vertices.data(), vertices.size() * sizeof(GLfloat));
    EBO EBO1(indices.data(), indices.size() * sizeof(GLuint));

    VAO1.LinkAttrib(VBO1, 0, 3, GL_FLOAT, 10 * sizeof(float), (void*)0);
    VAO1.LinkAttrib(VBO1, 1, 1, GL_FLOAT, 10 * sizeof(float), (void*)(3 * sizeof(float)));
    VAO1.LinkAttrib(VBO1, 2, 1, GL_FLOAT, 10 * sizeof(float), (void*)(4 * sizeof(float)));
    VAO1.LinkAttrib(VBO1, 3, 3, GL_FLOAT, 10 * sizeof(float), (void*)(5 * sizeof(float)));
    VAO1.LinkAttrib(VBO1, 4, 1, GL_FLOAT, 10 * sizeof(float), (void*)(8 * sizeof(float)));
    VAO1.LinkAttrib(VBO1, 5, 1, GL_FLOAT, 10 * sizeof(float), (void*)(9 * sizeof(float)));

    VAO1.Unbind();
    VBO1.Unbind();
    EBO1.Unbind();

    glm::vec3 beamsPos = glm::vec3(0.f, 0.f, 0.f);
    glm::mat4 beamModel = glm::mat4(1.0f);
    beamModel = glm::translate(beamModel, beamsPos);

    shaderProgram.Activate();
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(beamModel));
    glUniform1i(glGetUniformLocation(shaderProgram.ID, "Count"), fluxDivider);
    //glUniform4f(glGetUniformLocation(shaderProgram.ID, "lightColor"), 1.f, 1.f, 1.f, 1.f);

    //glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    Camera camera(width, height, glm::vec3(0.f, 0.f, 1.f));

    //sutil::CUDAOutputBuffer<uchar4> output_buffer(
    //    sutil::CUDAOutputBufferType::GL_INTEROP,
    //    launchParams.width,
    //    launchParams.height
    //);

    //output_buffer.setStream(sample.getStream());
    //sutil::GLDisplay gl_display;

    std::chrono::duration<double> state_update_time(0.0);
    std::chrono::duration<double> render_time(0.0);
    std::chrono::duration<double> display_time(0.0);

    do
    {
        auto t0 = std::chrono::steady_clock::now();
        glfwPollEvents();

        //output_buffer.resize(launchParams.width, launchParams.height);
        auto t1 = std::chrono::steady_clock::now();
        state_update_time += t1 - t0;
        t0 = t1;

        //render();

        glClearColor(0.f, 0.f, 0.f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        camera.Inputs(handle);
        camera.updateMatrix(45.0f, 0.1f, 100.0f);

        shaderProgram.Activate();
        // Export the camMatrix to the Vertex Shader of the pyramid
        camera.Matrix(shaderProgram, "camMatrix");
        glUniform3f(glGetUniformLocation(shaderProgram.ID, "camPos"), camera.Position.x, camera.Position.y, camera.Position.z);
        // Bind the VAO so OpenGL knows to use it
        VAO1.Bind();
        // Draw primitives, number of indices, datatype of indices, index of indices
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);


        t1 = std::chrono::steady_clock::now();
        render_time += t1 - t0;
        t0 = t1;

        //draw(output_buffer, gl_display);
        t1 = std::chrono::steady_clock::now();
        display_time += t1 - t0;

        sutil::displayStats(state_update_time, render_time, display_time);
        glfwSwapBuffers(handle);
    } while (!glfwWindowShouldClose(handle));

    CUDA_SYNC_CHECK();
    sutil::cleanupUI(handle);
}

void DisplayWindow::render() {
    glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void DisplayWindow::draw(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display) {
    uchar4* result_buffer_data = output_buffer.map();
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