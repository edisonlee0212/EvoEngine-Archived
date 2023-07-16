cmake_minimum_required(VERSION 3.17)

project(
	EvoEngine 
	VERSION 0.1 
	DESCRIPTION "An interactive application framework"
)

include(GenerateExportHeader)
set(CMAKE_CXX_STANDARD 17)

# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------
option(EVOENGINE_BUILD_EXAMPLES "Build EvoEngine Examples" OFF)

# ------------------------------------------------------------------
# Build types
# ------------------------------------------------------------------

# Set a default build type if none was specified
set(default_build_type "Release")
# Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
endif ()

if (CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
    set(EVOENGINE_DEBUG_STATUS NDEBUG)
elseif (CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
    set(EVOENGINE_DEBUG_STATUS _DEBUG)
endif ()

# ------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------
# We need a CMAKE_DIR with some code to find external dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(3RD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty)
add_subdirectory(${3RD_PARTY_DIR})

# ------------------------------------------------------------------
# EvoEngine - sources
# ------------------------------------------------------------------

file(GLOB LOCAL_EVOENGINE_SOURCES "src/lib/*.cpp")
add_library(evoengine
        STATIC
        ${IMGUI_SOURCES}
        ${LOCAL_EVOENGINE_SOURCES}
        )

# ------------------------------------------------------------------
# EvoEngine - headers
# ------------------------------------------------------------------
set(LOCAL_EVOENGINE_PCH ${CMAKE_CURRENT_SOURCE_DIR}/include/lib/EvoEngine-pch.hpp)
target_precompile_headers(evoengine
        PRIVATE
        ${LOCAL_EVOENGINE_PCH}
        )

set(LOCAL_EVOENGINE_INCLUDES
        ${3RD_PARTY_INCLUDES}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/include/lib/Engine
        ${CMAKE_CURRENT_SOURCE_DIR}/include/lib/Engine/Core
        ${CMAKE_CURRENT_SOURCE_DIR}/include/lib/Engine/ECS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/lib/Engine/Layers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/lib/Engine/Assets
        ${CMAKE_CURRENT_SOURCE_DIR}/include/lib/Engine/Physics
        ${CMAKE_CURRENT_SOURCE_DIR}/include/lib/Engine/Rendering
        ${CMAKE_CURRENT_SOURCE_DIR}/include/lib/Engine/Editor
        ${CMAKE_CURRENT_SOURCE_DIR}/include/lib/Engine/Utilities
        ${CMAKE_CURRENT_SOURCE_DIR}/include/lib/External

        ${CMAKE_CURRENT_BINARY_DIR}
        )

target_include_directories(evoengine
        PRIVATE
        ${LOCAL_EVOENGINE_INCLUDES}
        )


# ------------------------------------------------------------------
# EvoEngine - linkings
# ------------------------------------------------------------------
target_link_libraries(evoengine
        volk
        yaml-cpp
        GPUOpen::VulkanMemoryAllocator
        )


target_compile_definitions(evoengine
        PRIVATE
        ${EVOENGINE_INTERNALS_DEF}
        ${EVOENGINE_DEFAULT_PROJECT_DEF}
        ${EVOENGINE_DEBUG_STATUS}
        )
if (WIN32)
    if (CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
        file(GLOB PHYSX_LIBS "3rdParty/physx/win.x86_64.vc143.mt/release/*.lib")
        set(SHADERC_LIB ${SHADERC_DIR}lib/shaderc_combined.lib)
        set(ASSIMP_LIBS
                ${ASSIMP_DIR}assimp-vc143-mt.lib
                ${ASSIMP_DIR}zlibstatic.lib
                )
    else()
        file(GLOB PHYSX_LIBS "3rdParty/physx/win.x86_64.vc143.mt/debug/*.lib")
        set(SHADERC_LIB ${SHADERC_DIR}lib/shaderc_combined_d.lib)
        set(ASSIMP_LIBS
                ${ASSIMP_DIR}assimp-vc143-mtd.lib
                ${ASSIMP_DIR}zlibstaticd.lib
                )
    endif ()
    target_link_libraries(evoengine
            ${GLFW_DIR}glfw3.lib
            ${PHYSX_LIBS}
            ${SHADERC_LIB}
            ${ASSIMP_LIBS}
            )
    # Compiler settings for Windows platform
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    target_compile_definitions(evoengine
            PRIVATE
            NOMINMAX
            USE_ASSIMP
            )
elseif (UNIX AND NOT APPLE)
    # Linker and compiler settings for Linux platform
    if (CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
        file(GLOB PHYSX_LIBS "3rdParty/physx/linux.clang/release/*.a")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    elseif (CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
        file(GLOB PHYSX_LIBS "3rdParty/physx/linux.clang/debug/*.a")
        set(CMAKE_CXX_FLAGS_DEBUG "-g")
    endif ()
    target_link_libraries(evoengine
            glfw
            dl
            X11
            pthread
            Xrandr
            Xi
            assimp
            cuda
            ${PHYSX_LIBS}
            )
elseif (APPLE)
endif ()

# ------------------------------------------------------------------
# EvoEngine - compiler
# ------------------------------------------------------------------
if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    message(STATUS "Setting G++ flags")
    # G++
    target_compile_options(evoengine PRIVATE -Wall -Wextra)
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    message(STATUS "Setting MSVC flags")
    # MSVC
    target_compile_options(evoengine PRIVATE /EHsc /W2 /c)
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    message(STATUS "Setting Clang flags")
    # Activate all clang warnings except those that don't really affect much
    target_compile_options(evoengine PRIVATE -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-newline-eof -Wno-padded -Wno-exit-time-destructors -Wno-global-constructors -Wno-constant-conversion)
endif ()

# ------------------------------------------------------------------
# Examples
# ------------------------------------------------------------------
if (${CMAKE_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR} OR EVOENGINE_BUILD_EXAMPLES)
    add_executable(Project
            src/app/Project.cpp
            )
    target_include_directories(Project
            PRIVATE
            ${LOCAL_EVOENGINE_INCLUDES}
            )
    target_precompile_headers(Project
            PRIVATE
            ${LOCAL_EVOENGINE_PCH}
            )
    target_link_libraries(Project
            evoengine
            )

    file(COPY src/app/imgui.ini DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif ()

# ------------------------------------------------------------------
# Copy Internal resources
# ------------------------------------------------------------------

set(LOCAL_EVOENGINE_INTERNALS_DIR ${CMAKE_CURRENT_BINARY_DIR}/Internals/)
set(EVOENGINE_INTERNALS_DEF EVOENGINE_INTERNALS_FOLDER=\"${LOCAL_EVOENGINE_INTERNALS_DIR}\")

set(LOCAL_EVOENGINE_DEFAULT_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Resources/TestingProject/)
set(EVOENGINE_DEFAULT_PROJECT_DEF EVOENGINE_DEFAULT_PROJECT_FOLDER=\"${LOCAL_EVOENGINE_DEFAULT_PROJECT_DIR}\")

add_custom_command(TARGET evoengine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Internals
        ${CMAKE_BINARY_DIR})

# ------------------------------------------------------------------
# Copy DLLs
# ------------------------------------------------------------------
if (WIN32)
    if (CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
        file(GLOB PHYSX_DLLS "3rdParty/physx/win.x86_64.vc143.mt/release/*.dll")
        set(ASSIMP_DLL ${3RD_PARTY_DIR}/assimp/assimp-vc143-mt.dll)
    elseif (CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
        file(GLOB PHYSX_DLLS "3rdParty/physx/win.x86_64.vc143.mt/debug/*.dll")
        set(ASSIMP_DLL ${3RD_PARTY_DIR}/assimp/assimp-vc143-mtd.dll)
    endif ()
    add_custom_command(
            TARGET evoengine
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${ASSIMP_DLL}
            ${CMAKE_BINARY_DIR}
    )
elseif (UNIX AND NOT APPLE)
    if (CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
        file(GLOB PHYSX_DLLS "3rdParty/physx/linux.clang/release/*.so")
    elseif (CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
        file(GLOB PHYSX_DLLS "3rdParty/physx/linux.clang/debug/*.so")
    endif ()
endif ()

foreach (PHYSX_DLL ${PHYSX_DLLS})
    add_custom_command(
            TARGET evoengine
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${PHYSX_DLL}
            ${CMAKE_BINARY_DIR}
    )
endforeach ()

if (NOT ${CMAKE_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    if (WIN32)
        add_custom_command(
                TARGET evoengine
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${PROJECT_BINARY_DIR}/evoengine.dll
                ${CMAKE_BINARY_DIR}
        )
    endif ()
    set(EVOENGINE_INCLUDES ${LOCAL_EVOENGINE_INCLUDES} PARENT_SCOPE)
    set(EVOENGINE_PCH ${LOCAL_EVOENGINE_PCH} PARENT_SCOPE)
endif ()