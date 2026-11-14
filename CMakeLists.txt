cmake_minimum_required(VERSION 3.20)
project(VulkanEngine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED COMPONENTS glslc)

include(FetchContent)

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)

# GLM (математика: векторы и матрицы)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
)

FetchContent_MakeAvailable(glfw glm)

add_executable(VulkanEngine src/main.cpp)
target_link_libraries(VulkanEngine PRIVATE Vulkan::Vulkan glfw glm::glm)

if(MSVC)
    target_compile_options(VulkanEngine PRIVATE /utf-8)
endif()

# ---- Шейдеры: GLSL -> SPIR-V ----
set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

set(SHADER_FILES triangle.vert triangle.frag)
set(SPV_FILES "")

foreach(shader ${SHADER_FILES})
    add_custom_command(
        OUTPUT ${SHADER_OUT_DIR}/${shader}.spv
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${SHADER_SRC_DIR}/${shader} -o ${SHADER_OUT_DIR}/${shader}.spv
        DEPENDS ${SHADER_SRC_DIR}/${shader}
        COMMENT "Compiling ${shader}"
    )
    list(APPEND SPV_FILES ${SHADER_OUT_DIR}/${shader}.spv)
endforeach()

add_custom_target(Shaders DEPENDS ${SPV_FILES})
add_dependencies(VulkanEngine Shaders)

# Путь к скомпилированным шейдерам, чтобы программа находила их откуда бы ни запускалась
target_compile_definitions(VulkanEngine PRIVATE SHADER_DIR="${SHADER_OUT_DIR}")
