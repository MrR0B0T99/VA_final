// glx/cleanup.hpp
#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "glx/mesh.hpp"

namespace glx {

inline void cleanup(GLuint bgProgram, GLuint lineProgram,
                    GLuint bgTex,
                    const glx::Mesh& bg,
                    const glx::Mesh& cube,
                    const glx::Axes& axes,
                    GLFWwindow* window) {
    glDeleteProgram(bgProgram);
    glDeleteProgram(lineProgram);
    glDeleteTextures(1, &bgTex);

    glDeleteVertexArrays(1, &bg.vao);
    glDeleteBuffers(1, &bg.vbo);

    glDeleteVertexArrays(1, &cube.vao);
    glDeleteBuffers(1, &cube.vbo);
    glDeleteBuffers(1, &cube.ebo);

    glDeleteVertexArrays(1, &axes.x.vao);
    glDeleteBuffers(1, &axes.x.vbo);
    glDeleteVertexArrays(1, &axes.y.vao);
    glDeleteBuffers(1, &axes.y.vbo);
    glDeleteVertexArrays(1, &axes.z.vao);
    glDeleteBuffers(1, &axes.z.vbo);

    glfwDestroyWindow(window);
    glfwTerminate();
}
}
