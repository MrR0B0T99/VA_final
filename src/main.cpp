#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "ar/calib.hpp"
#include "ar/pose.hpp"
#include "detect/a4.hpp"
#include "glx/mesh.hpp"
#include "glx/shaders.hpp"
#include "glx/texture.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <utility>

/**
 * @brief Démo AR : détection d'une feuille A4 par frame (vidéo), estimateur de pose (solvePnP), rendu OpenGL (cube + axes).
 */

enum class CaptureMode { Video, Webcam };
int main(){
  try {
    // --- Chargement calibration ---
    const ar::Calibration calib = ar::loadCalibration("../data/camera.yaml");

    const std::string VIDEO_PATH = "../data/Video_AR_1.mp4";
    cv::VideoCapture cap;
    cv::Mat frameBGR;
    CaptureMode mode = CaptureMode::Video;

    auto openCapture = [&](CaptureMode requested, cv::Mat& firstFrame) -> bool {
      cv::VideoCapture newCap;
      if (requested == CaptureMode::Video) {
        newCap.open(VIDEO_PATH);
        if (!newCap.isOpened()) {
          std::cerr << "Video not found!\n";
          return false;
        }
      } else {
        newCap.open(0);
        if (!newCap.isOpened()) {
          std::cerr << "Webcam non détectée !\n";
          return false;
        }
      }

      cv::Mat tmp;
      if (!newCap.read(tmp) || tmp.empty()) {
        if (requested == CaptureMode::Video) {
          std::cerr << "Empty first frame from video!\n";
        } else {
          std::cerr << "Impossible de lire depuis la webcam !\n";
        }
        return false;
      }

      cap = std::move(newCap);
      firstFrame = tmp;
      return true;
    };

    if (!openCapture(mode, frameBGR)) {
      return -1;
    }

    int vw = frameBGR.cols, vh = frameBGR.rows;

    // --- Init GLFW/GL ---
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  #endif
    const char* initialTitle = (mode == CaptureMode::Video) ? "ARCube (Video)" : "ARCube (Webcam)";
    GLFWwindow* window = glfwCreateWindow(vw, vh, initialTitle, nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window); glfwSwapInterval(1);

    glewExperimental = GL_TRUE; if (glewInit()!=GLEW_OK) { std::cerr << "GLEW init failed\n"; return -1; }
    glGetError(); // clear spurious err

    // --- Shaders ---
    GLuint bgVS = glx::compile(GL_VERTEX_SHADER,   glx::BG_VS);
    GLuint bgFS = glx::compile(GL_FRAGMENT_SHADER, glx::BG_FS);
    GLuint bgProgram = glx::link({bgVS,bgFS});
    glDeleteShader(bgVS); glDeleteShader(bgFS);

    GLuint lineVS = glx::compile(GL_VERTEX_SHADER,   glx::LINE_VS);
    GLuint lineGS = glx::compile(GL_GEOMETRY_SHADER, glx::LINE_GS);
    GLuint lineFS = glx::compile(GL_FRAGMENT_SHADER, glx::LINE_FS);
    GLuint lineProgram = glx::link({lineVS,lineGS,lineFS});
    glDeleteShader(lineVS); glDeleteShader(lineGS); glDeleteShader(lineFS);

    // --- Géométries ---
    glx::Mesh bg   = glx::createBackgroundQuad();
    glx::Mesh cube = glx::createCubeWireframe(30.0f);
    glx::Axes axes = glx::createAxes(210.0f);

    // --- Texture du fond ---
    cv::Mat frameRGBA; cv::cvtColor(frameBGR, frameRGBA, cv::COLOR_BGR2RGBA);
    GLuint bgTex = glx::createTextureRGBA(frameRGBA.cols, frameRGBA.rows);

    glEnable(GL_DEPTH_TEST); glClearColor(0.05f, 0.05f, 0.06f, 1.0f);

    // Uniform locations
    const GLint bg_uTex         = glGetUniformLocation(bgProgram,   "uTex");
    const GLint line_uMVP       = glGetUniformLocation(lineProgram, "uMVP");
    const GLint line_uColor     = glGetUniformLocation(lineProgram, "uColor");
    const GLint line_uThickness = glGetUniformLocation(lineProgram, "uThicknessPx");
    const GLint line_uViewport  = glGetUniformLocation(lineProgram, "uViewport");
    const float THICKNESS_PX = 3.0f;

    // Objet réel A4 (mm) : TL, BL, BR, TR (centré sur (0,0,0))
    const float W = 210.f, H = 297.f;
    std::vector<cv::Point3f> objectPts = {
      {-W*0.5f, -H*0.5f, 0.0f}, // TL
      {-W*0.5f, +H*0.5f, 0.0f}, // BL
      {+W*0.5f, +H*0.5f, 0.0f}, // BR
      {+W*0.5f, -H*0.5f, 0.0f}  // TR
    };

    cv::Mat rvec, tvec;

    bool usePendingFrame = true;
    bool prevWPressed = false;
    bool prevVPressed = false;

    while (!glfwWindowShouldClose(window)) {
      if (!usePendingFrame) {
        if (!cap.read(frameBGR) || frameBGR.empty()) {
          if (mode == CaptureMode::Video) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!cap.read(frameBGR) || frameBGR.empty()) {
              std::cerr << "Impossible de lire la vidéo.\n";
              break;
            }
          } else {
            std::cerr << "Lecture webcam échouée.\n";
            break;
          }
        }
      }
      usePendingFrame = false;

      // Détection coins
      std::vector<cv::Point2f> imagePts; bool okDetect = detect::detectA4Corners(frameBGR, imagePts);

      // Estimateur de pose (si détecté)
      if (okDetect) {
        cv::solvePnP(objectPts, imagePts, calib.cameraMatrix, calib.distCoeffs,
                     rvec, tvec, !rvec.empty(), cv::SOLVEPNP_ITERATIVE);
      }

      // BG texture update
      cv::cvtColor(frameBGR, frameRGBA, cv::COLOR_BGR2RGBA);
      cv::flip(frameRGBA, frameRGBA, 0); // OpenGL expects (0,0) at bottom-left
      static int texW = frameRGBA.cols, texH = frameRGBA.rows;
      if (frameRGBA.cols != texW || frameRGBA.rows != texH) {
        glDeleteTextures(1, &bgTex);
        bgTex = glx::createTextureRGBA(frameRGBA.cols, frameRGBA.rows);
        texW = frameRGBA.cols; texH = frameRGBA.rows;
      }
      glx::updateTextureRGBA(bgTex, frameRGBA);

      // Rendu
      glfwPollEvents(); int fbw, fbh; glfwGetFramebufferSize(window, &fbw, &fbh);
      glViewport(0,0,fbw,fbh); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      // 1) Fond vidéo
      glDisable(GL_DEPTH_TEST);
      glUseProgram(bgProgram);
      glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, bgTex);
      glUniform1i(bg_uTex, 0);
      glBindVertexArray(bg.vao); glDrawArrays(GL_TRIANGLES, 0, bg.count);
      glBindVertexArray(0); glBindTexture(GL_TEXTURE_2D, 0);

      // 2) Cube + axes
      glEnable(GL_DEPTH_TEST);
      glm::mat4 P = ar::projectionFromCV(calib.cameraMatrix, (float)fbw, (float)fbh, 0.1f, 2000.0f);
      glm::mat4 V = ar::viewFromRvecTvec(rvec, tvec);

      // Cube posé sur la feuille (base à z=0)
      const float s = 30.0f;
      glm::mat4 M_axes = glm::mat4(1.0f);
      glm::mat4 M_cube = glm::translate(glm::mat4(1.0f), glm::vec3(0.f,0.f,s));

      glUseProgram(lineProgram);
      glUniform2f(line_uViewport, (float)fbw, (float)fbh);
      glUniform1f(line_uThickness, THICKNESS_PX);

      // Axes
      glm::mat4 MVP_axes = P * V * M_axes;
      glUniformMatrix4fv(line_uMVP, 1, GL_FALSE, glm::value_ptr(MVP_axes));
      glBindVertexArray(axes.x.vao); glUniform3f(line_uColor, 1.f,0.f,0.f); glDrawArrays(GL_LINES, 0, axes.x.count);
      glBindVertexArray(axes.y.vao); glUniform3f(line_uColor, 0.f,1.f,0.f); glDrawArrays(GL_LINES, 0, axes.y.count);
      glBindVertexArray(axes.z.vao); glUniform3f(line_uColor, 0.f,0.f,1.f); glDrawArrays(GL_LINES, 0, axes.z.count);
      glBindVertexArray(0);

      // Cube
      glm::mat4 MVP_cube = P * V * M_cube;
      glUniformMatrix4fv(line_uMVP, 1, GL_FALSE, glm::value_ptr(MVP_cube));
      glUniform3f(line_uColor, 0.f,0.f,0.f);
      glBindVertexArray(cube.vao);
      glDrawElements(GL_LINES, cube.count, GL_UNSIGNED_INT, 0);
      glBindVertexArray(0);

      glfwSwapBuffers(window);

      bool wPressed = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
      bool vPressed = glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS;

      CaptureMode requested = mode;
      if (wPressed && !prevWPressed) requested = CaptureMode::Webcam;
      if (vPressed && !prevVPressed) requested = CaptureMode::Video;
      prevWPressed = wPressed;
      prevVPressed = vPressed;

      if (requested != mode) {
        cv::Mat newFrame;
        if (openCapture(requested, newFrame)) {
          mode = requested;
          frameBGR = newFrame;
          usePendingFrame = true;
          glfwSetWindowTitle(window, (mode == CaptureMode::Video) ? "ARCube (Video)" : "ARCube (Webcam)");
          glfwSetWindowSize(window, frameBGR.cols, frameBGR.rows);
          std::cout << ((mode == CaptureMode::Video) ? "Lecture MP4 activée." : "Webcam activée.") << std::endl;
        } else {
          std::cerr << "Changement de source impossible, conservation de la source actuelle.\n";
        }
      }
    }

    cap.release();

    // Cleanup
    glDeleteProgram(bgProgram); glDeleteProgram(lineProgram);
    glDeleteTextures(1, &bgTex);
    glDeleteVertexArrays(1, &bg.vao); glDeleteBuffers(1, &bg.vbo);
    glDeleteVertexArrays(1, &cube.vao); glDeleteBuffers(1, &cube.vbo); glDeleteBuffers(1, &cube.ebo);
    glDeleteVertexArrays(1, &axes.x.vao); glDeleteBuffers(1, &axes.x.vbo);
    glDeleteVertexArrays(1, &axes.y.vao); glDeleteBuffers(1, &axes.y.vbo);
    glDeleteVertexArrays(1, &axes.z.vao); glDeleteBuffers(1, &axes.z.vbo);
    glfwDestroyWindow(window); glfwTerminate();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Fatal: " << e.what() << std::endl; return -1;
  }
}
