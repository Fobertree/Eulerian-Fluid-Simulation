cmake_minimum_required(VERSION 3.27)
project(FluidSim)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

find_package(OpenGL REQUIRED)
# below packages installed via vcpkg
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_executable(FluidSim
        src/main.cpp
        include/shader.h
        src/main.cpp
        include/gridgl.h
        include/fluid_math.h
)


# link libraries after add_executable
target_link_libraries(FluidSim PRIVATE OpenGL::GL)
target_link_libraries(FluidSim PRIVATE glm::glm)
target_link_libraries(FluidSim PRIVATE glfw)
target_link_libraries(FluidSim PRIVATE glad::glad)
target_link_libraries(FluidSim PRIVATE Eigen3::Eigen)