cmake_minimum_required(VERSION 3.27)
project(Grid)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

find_package(OpenGL REQUIRED)
# below packages installed via vcpkg
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

add_executable(Grid
        src/main.cpp
        include/shader.h
        src/main.cpp
        include/gridgl.h
)


# link libraries after add_executable
target_link_libraries(Grid PRIVATE OpenGL::GL)
target_link_libraries(Grid PRIVATE glm::glm)
target_link_libraries(Grid PRIVATE glfw)
target_link_libraries(Grid PRIVATE glad::glad)