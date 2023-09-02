# Locate the vulkan library
#
# This module defines the following variables:
#
# VULKAN_LIBRARY the name of the library;
# VULKAN_INCLUDE_DIR where to find glfw include files.
# VULKAN_FOUND true if both the VULKAN_LIBRARY and VULKAN_INCLUDE_DIR have been found.
#
# To help locate the library and include file, you can define a
# variable called VULKAN_ROOT which points to the root of the glfw library
# installation.
#
# default search dirs
# 
# Cmake file from: https://github.com/daw42/glslcookbook

set( _vulkan_HEADER_SEARCH_DIRS
	"${CONDA_PREFIX}/include"
	"/usr/include"
	"/usr/local/include"
	"${CMAKE_SOURCE_DIR}/include"
	"C:/Program Files (x86)/glfw/include" )
set( _vulkan_LIB_SEARCH_DIRS
	"${CONDA_PREFIX}/lib"
	"/usr/lib"
	"/usr/lib/x86-64-linux-gnu"
	"/usr/local/lib"
	"${CMAKE_SOURCE_DIR}/lib"
	"C:/Program Files (x86)/glfw/lib-msvc110" )

# Check environment for root search directory
set( _vulkan_ENV_ROOT $ENV{VULKAN_ROOT} )
if( NOT VULKAN_ROOT AND _vulkan_ENV_ROOT )
	set(VULKAN_ROOT ${_vulkan_ENV_ROOT} )
endif()

# Put user specified location at beginning of search
if( VULKAN_ROOT )
	list( INSERT _vulkan_HEADER_SEARCH_DIRS 0 "${VULKAN_ROOT}/include" )
	list( INSERT _vulkan_LIB_SEARCH_DIRS 0 "${VULKAN_ROOT}/lib" )
endif()

# Search for the header
FIND_PATH(VULKAN_INCLUDE_DIR "vulkan/vulkan.h"
PATHS ${_vulkan_HEADER_SEARCH_DIRS} )

# Search for the library
FIND_LIBRARY(VULKAN_LIBRARY NAMES vulkan Vulkan Vulkan::Vulkan Vulkan::vulkan
PATHS ${_vulkan_LIB_SEARCH_DIRS} )
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(VULKAN DEFAULT_MSG
VULKAN_LIBRARY VULKAN_INCLUDE_DIR)
