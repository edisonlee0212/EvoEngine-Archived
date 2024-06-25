if (${CMAKE_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR} OR EVOENGINE_BUILD_APP_DEMO)
	set(DEFAULT_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Resources/TestingProject/)
	set(DEFAULT_PROJECT_DEF EVOENGINE_DEFAULT_PROJECT_FOLDER=\"${DEFAULT_PROJECT_DIR}\")
	
	add_executable(Demo
		src/Demo.cpp
		)
	target_compile_definitions(Demo
		PRIVATE
		${DEFAULT_PROJECT_DEF}
		${EVOENGINE_DEFS}
		)
	target_include_directories(Demo
		PRIVATE
		${EVOENGINE_INCLUDES}
		)
	target_precompile_headers(Demo
		PRIVATE
		${EVOENGINE_PCHS}
		)
	target_link_libraries(Demo
		EvoEngine_SDK
		${EvoEngine_Plugins}
		)
	file(COPY imgui.ini DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
	evoengine_copy_resources(Demo ${CMAKE_CURRENT_BINARY_DIR})
endif ()