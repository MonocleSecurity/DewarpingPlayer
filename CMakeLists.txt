cmake_minimum_required(VERSION 3.21)

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")

project(DewarpingPlayer C CXX)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv4")

find_package(FFMPEG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(DewarpingPlayer
main.cpp)

include_directories(DewarpingPlayer ${FFMPEG_INCLUDE_DIRS})

target_link_libraries(DewarpingPlayer PRIVATE ${FFMPEG_LIBRARIES})
target_link_libraries(DewarpingPlayer PRIVATE GLEW::GLEW)
target_link_libraries(DewarpingPlayer PRIVATE glfw)
target_link_libraries(DewarpingPlayer PRIVATE imgui::imgui)
target_link_libraries(DewarpingPlayer PRIVATE ${OpenCV_LIBS})

if(WIN32)
  target_link_libraries(DewarpingPlayer PRIVATE opengl32)
endif()
