# - Try to find Assimp
# Once done, this will define
#
# VULKAN_FOUND - system has Assimp
# VULKAN_INCLUDE_DIR - the Assimp include directories
# VULKAN_LIBRARIES - link these to use Assimp
FIND_PATH( VULKAN_INCLUDE_DIR vulkan/mesh.h
	"${CONDA_PREFIX}/include"
	"/usr/include"
	"/usr/local/include"
	"/opt/local/include"
	"${CMAKE_SOURCE_DIR}/include"
)
FIND_LIBRARY( VULKAN_LIBRARY vulkan
	"${CONDA_PREFIX}/lib"
	"/usr/lib64"
	"/usr/lib"
	"/usr/local/lib"
	"/opt/local/lib"
	"${CMAKE_SOURCE_DIR}/lib"
)
IF(VULKAN_INCLUDE_DIR AND VULKAN_LIBRARY)
	SET( VULKAN_FOUND TRUE )
	SET( VULKAN_LIBRARIES ${VULKAN_LIBRARY} )
ENDIF(VULKAN_INCLUDE_DIR AND VULKAN_LIBRARY)
IF(VULKAN_FOUND)
	IF(NOT VULKAN_FIND_QUIETLY)
	MESSAGE(STATUS "Found VULKAN: ${VULKAN_LIBRARY}")
	ENDIF(NOT VULKAN_FIND_QUIETLY)
ELSE()
	IF(VULKAN_FIND_REQUIRED)
	MESSAGE(FATAL_ERROR "Could not find libVULKAN")
	ENDIF(VULKAN_FIND_REQUIRED)
ENDIF()
