cmake_minimum_required(VERSION 3.29)
project(VulkanTesting)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable C++ module dependency scanning
#set(CMAKE_CXX_SCAN_FOR_MODULES ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

# Vulkan SDK
find_package(Vulkan REQUIRED)
if (NOT Vulkan_FOUND)
    message(STATUS "Vulkan SDK not found. Installing...")
    include(FetchContent)
    FetchContent_Declare(
            vulkan
            URL https://vulkan.lunarg.com/sdk/home
    )
    FetchContent_MakeAvailable(vulkan)
endif ()

# TinyOBJLoader
include(FetchContent)
FetchContent_Declare(
        tinyobjloader
        GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
        GIT_TAG v1.0.6
)
FetchContent_MakeAvailable(tinyobjloader)

# GLFW3
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)
add_executable(VulkanTesting main.cpp)
target_link_libraries(VulkanTesting PRIVATE glfw Vulkan::cppm tinyobjloader)

# set up Vulkan C++ module
add_library(VulkanCppModule)
add_library(Vulkan::cppm ALIAS VulkanCppModule)

target_compile_definitions(VulkanCppModule
        PRIVATE
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VK_USE_PLATFORM_XLIB_KHR=1
)
target_include_directories(VulkanCppModule
        PRIVATE
        "${Vulkan_INCLUDE_DIR}"
)
target_link_libraries(VulkanCppModule
        PUBLIC
        Vulkan::Vulkan
)

set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 23)

target_sources(VulkanCppModule
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES
        BASE_DIRS
        "${Vulkan_INCLUDE_DIR}"
        FILES
        "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
)

## Add the vulkan.cppm file directly as a source file
target_sources(VulkanCppModule
        PRIVATE
        "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
)

add_executable(glslang::validator IMPORTED)
find_program(GLSLANG_VALIDATOR "glslangValidator" HINTS $ENV{VULKAN_SDK_PATH}/bin REQUIRED)
set_property(TARGET glslang::validator PROPERTY IMPORTED_LOCATION "${GLSLANG_VALIDATOR}")

function(add_slang_shader_target TARGET)
    cmake_parse_arguments(SHADER "" "SOURCES" ${ARGN})
    set(SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
    set(ENTRY_POINTS -entry vertMain -entry fragMain)
    set(SLANGC_EXECUTABLE ${VULKAN_SDK_PATH}/bin/slangc)

    message(STATUS "Slang compiler: ${SLANGC_EXECUTABLE}")
    message(STATUS "Shader sources: ${SHADER_SOURCES}")
    message(STATUS "Shaders dir: ${SHADERS_DIR}")

    add_custom_command(
            OUTPUT ${SHADERS_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    )
    add_custom_command(
            OUTPUT ${SHADERS_DIR}/slang.spv
            COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o ${SHADERS_DIR}/slang.spv
            WORKING_DIRECTORY ${SHADERS_DIR}
            DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
            COMMENT "Compiling Slang Shaders"
            VERBATIM
    )
    add_custom_target(${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

set(SHADER_SLANG_SOURCES "${CMAKE_CURRENT_LIST_DIR}/shaders/shader.slang")
add_slang_shader_target(shader PUBLIC SOURCES ${SHADER_SLANG_SOURCES})