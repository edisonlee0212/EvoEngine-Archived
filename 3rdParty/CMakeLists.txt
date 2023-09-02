# ------------------------------------------------------------------
# assimp
# ------------------------------------------------------------------
set(LOCAL_ASSIMP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assimp/include)
if (WIN32)
else ()
    find_package(ASSIMP REQUIRED)
    message(STATUS "Found ASSIMP in ${ASSIMP_INCLUDE_DIR}")
endif ()
set(ASSIMP_DIR ${3RD_PARTY_DIR}/assimp/ PARENT_SCOPE)

# ------------------------------------------------------------------
# glfw
# ------------------------------------------------------------------
set(LOCAL_GLFW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include)
if (WIN32)
else ()
    find_package(GLFW3 REQUIRED)
    message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")
endif ()
set(GLFW_DIR ${3RD_PARTY_DIR}/glfw/ PARENT_SCOPE)

# ------------------------------------------------------------------
# yaml
# ------------------------------------------------------------------
set(YAML_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp/)
include_directories(${YAML_CPP_DIR}/include)
add_subdirectory(${YAML_CPP_DIR} EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# pybind11
# ------------------------------------------------------------------
set(PYBIND11_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/)
include_directories(${PYBIND11_DIR}/include)
add_subdirectory(${PYBIND11_DIR} EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# glm
# ------------------------------------------------------------------
if (WIN32)
    set(GLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glm/)
    include_directories(${GLM_DIR})
    add_subdirectory(${GLM_DIR} EXCLUDE_FROM_ALL)
elseif (UNIX AND NOT APPLE)
    find_package(GLM REQUIRED)
    message(STATUS "GLM included at ${GLM_INCLUDE_DIR}")
endif ()

# ------------------------------------------------------------------
# volk
# ------------------------------------------------------------------
if (WIN32)
    set(VOLK_STATIC_DEFINES VK_NO_PROTOTYPES VK_USE_PLATFORM_WIN32_KHR)
elseif (UNIX AND NOT APPLE)
    find_package(Vulkan REQUIRED)
	message(STATUS "Vulkan included at ${VULKAN_INCLUDE_DIR}")
    set(VOLK_STATIC_DEFINES VK_NO_PROTOTYPES VK_USE_PLATFORM_XLIB_KHR)
elseif (APPLE)
    set(VOLK_STATIC_DEFINES VK_NO_PROTOTYPES VK_USE_PLATFORM_MACOS_MVK)
endif ()
set(VOLK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/volk/)
add_subdirectory(${VOLK_DIR} EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# shaderc
# ------------------------------------------------------------------
set(SHADERC_DIR 
    ${CMAKE_CURRENT_SOURCE_DIR}/shaderc/
    PARENT_SCOPE)

# ------------------------------------------------------------------
# VulkanMemoryAllocator
# ------------------------------------------------------------------
set(VULKAN_MEMORY_ALLOCATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VulkanMemoryAllocator/)
add_subdirectory(${VULKAN_MEMORY_ALLOCATOR_DIR} EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# stb
# ------------------------------------------------------------------
set(STB_DIR ${3RD_PARTY_DIR}/stb/ PARENT_SCOPE)

# ------------------------------------------------------------------
# ImGui & ImGuizmo
# ------------------------------------------------------------------
set(LOCAL_IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ImGui)
set(LOCAL_IMGUIZMO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo)

if (WIN32)
    set(IMGUI_BACKENDS ${LOCAL_IMGUI_DIR}/backends/imgui_impl_win32.cpp ${LOCAL_IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${LOCAL_IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)
elseif (UNIX AND NOT APPLE)
    set(IMGUI_BACKENDS ${LOCAL_IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${LOCAL_IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)
elseif (APPLE)
endif ()
file(GLOB LOCAL_IMGUI_SOURCES ${LOCAL_IMGUI_DIR}/*.cpp)
file(GLOB LOCAL_IMGUIZMO_SOURCES ${LOCAL_IMGUIZMO_DIR}/*.cpp)
set(IMGUI_SOURCES
        ${LOCAL_IMGUI_SOURCES}
        ${IMGUI_BACKENDS}
        ${LOCAL_IMGUIZMO_SOURCES}
        PARENT_SCOPE
        )
set(IMGUI_DIR
        ${3RD_PARTY_DIR}/ImGui
        ${3RD_PARTY_DIR}/ImGui/backends
        ${3RD_PARTY_DIR}/ImGuizmo
        PARENT_SCOPE)

# ------------------------------------------------------------------
# physx
# ------------------------------------------------------------------
set(PHYSX_DIR
        ${3RD_PARTY_DIR}/physx
        PARENT_SCOPE)




# ------------------------------------------------------------------
# Includes
# ------------------------------------------------------------------
set(3RD_PARTY_INCLUDES
	${VULKAN_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include
        ${CMAKE_CURRENT_SOURCE_DIR}/assimp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/stb
        ${CMAKE_CURRENT_SOURCE_DIR}/ImGui
        ${CMAKE_CURRENT_SOURCE_DIR}/ImGui/backends
        ${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo
        ${CMAKE_CURRENT_SOURCE_DIR}/glad/include
        ${CMAKE_CURRENT_SOURCE_DIR}/glad/include/glad
        ${CMAKE_CURRENT_SOURCE_DIR}/glad/include/KHR
        ${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader
        ${CMAKE_CURRENT_SOURCE_DIR}/physx/include
        ${CMAKE_CURRENT_SOURCE_DIR}/shaderc/include
        ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include
        PARENT_SCOPE
        )