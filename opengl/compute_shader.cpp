#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>

const char* computeShaderSource = R"(
#version 430
layout (local_size_x = 1) in;
layout (std430, binding = 0) buffer Data {
    int data[];
};
void main() {
    uint id = gl_GlobalInvocationID.x;
    data[id] *= 2;
}
)";

GLuint createComputeShader() {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &computeShaderSource, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(shader, 512, nullptr, info);
        std::cerr << "Compute shader compilation failed:\n" << info << std::endl;
    }
    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);
    return program;
}

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(1, 1, "", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    glewInit();

    std::vector<int> input = { 1, 2, 3, 4 };
    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, input.size() * sizeof(int), input.data(), GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    GLuint program = createComputeShader();
    glUseProgram(program);
    glDispatchCompute(input.size(), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    int* ptr = (int*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    std::cout << "Result: ";
    for (size_t i = 0; i < input.size(); ++i) {
        std::cout << ptr[i] << " ";
    }
    std::cout << std::endl;
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    glDeleteProgram(program);
    glDeleteBuffers(1, &ssbo);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

