#define DATA(vec) &vec.front()
#define SIZE_F sizeof(float)
#define SIZE_VF(vec) (vec.size() * SIZE_F)
#define SIZE_U_INT sizeof(unsigned int)
#define SIZE_V_UINT(vec) (vec.size() * SIZE_U_INT)
#define PRINT(x) std::cout << x << std::endl


// OpenGL-related includes
#include <glad/glad.h>
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.h"

// STL
#include <iostream>
#include <vector>
#include <string>

// debugging
#include <filesystem>

// other
#include <gridgl.h>

bool fileExists(const char* fileName)
{
    std::ifstream test(fileName);
    if (test) {
        return true;
    } else {
        return false;
    }
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0,0,width,height);
}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }
}

constexpr int width{800}, height{600}, grid_width{100}, grid_height{100}, stride{3};

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    GLFWwindow  *window = glfwCreateWindow(width,height, "grid", NULL,NULL);

    if (window == NULL)
    {
        std::cout << "Failed to create new window" << std::endl;
        glfwTerminate();
        exit(1);
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glViewport(0,0,width,height);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Setup grid
    Grid grid = Grid(grid_width,grid_height,width,height);
    std::vector<float> vertices = grid.getVertices();
    std::vector<unsigned int> ebo_indices = grid.getEBO();

    if (vertices.empty())
    {
        std::cout << "ERROR WITH EMPTY VERTICES" << std::endl;
        exit(1);
    }

    PRINT("VERTICES SIZE: " << vertices.size());

    // -- shaders --
    PRINT(std::filesystem::current_path());
    PRINT("Vertex path check: " << std::boolalpha << fileExists("../src/Shaders/vs1.vert"));
    PRINT("Frag path check: " << fileExists("../src/Shaders/borders.frag"));
    Shader boxShader = Shader("../src/Shaders/vs1.vert", "../src/Shaders/borders.frag");
    boxShader.use();

    glDisable(GL_DEPTH_TEST);

    // -- VBO, VAO, EBO --
    unsigned int VBO, VAO, EBO;

    PRINT("VBO OK");
    //VAO
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // VBO
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, SIZE_VF(vertices), DATA(vertices), GL_STATIC_DRAW);

    PRINT("VAO OK");
    //EBO
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, SIZE_V_UINT(ebo_indices), DATA(ebo_indices),GL_STATIC_DRAW);

    PRINT("EBO OK");

    // VAO: enable vertex attributes

    // position attributes
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,stride*SIZE_F,(void*) 0);
    glEnableVertexAttribArray(0);

    // set cell size uniform in fragment shader
    PRINT("norm height::" << grid.getNormHeight());
    PRINT("norm widgth::" << grid.getNormWidth());
    glm::vec2 cell = glm::vec2(grid.getNormWidth(), grid.getNormHeight());
    GLuint cell_size_unif = glGetUniformLocation(boxShader.ID, "cell_size");
    glUniform2fv(cell_size_unif, 1, glm::value_ptr(cell));

    while (!glfwWindowShouldClose(window))
    {
        processInput(window);
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cout << "OpenGL error: " << err << std::endl;
        }

        // background color
        glClearColor(0.2f,0.3f,0.3f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawElements(GL_TRIANGLES,SIZE_V_UINT(ebo_indices), GL_UNSIGNED_INT, 0);


        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    return 0;
}