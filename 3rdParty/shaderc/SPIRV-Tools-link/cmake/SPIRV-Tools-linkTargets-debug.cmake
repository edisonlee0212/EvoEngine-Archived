#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SPIRV-Tools-link" for configuration "Debug"
set_property(TARGET SPIRV-Tools-link APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SPIRV-Tools-link PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/SPIRV-Tools-link.lib"
  )

list(APPEND _cmake_import_check_targets SPIRV-Tools-link )
list(APPEND _cmake_import_check_files_for_SPIRV-Tools-link "${_IMPORT_PREFIX}/lib/SPIRV-Tools-link.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
