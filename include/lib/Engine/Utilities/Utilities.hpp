#pragma once
#include <shaderc/shaderc.hpp>

namespace EvoEngine
{
	class FileUtils
	{
	public:
		static std::string LoadFileAsString(const std::filesystem::path& path = "");
		static void OpenFolder(
			const std::string& dialogTitle,
			const std::function<void(const std::filesystem::path& path)>& func, bool projectDirCheck = true);

		static void OpenFile(
			const std::string& dialogTitle,
			const std::string& fileType,
			const std::vector<std::string>& extensions,
			const std::function<void(const std::filesystem::path& path)>& func, bool projectDirCheck = true);
		static void SaveFile(
			const std::string& dialogTitle,
			const std::string& fileType,
			const std::vector<std::string>& extensions,
			const std::function<void(const std::filesystem::path& path)>& func, bool projectDirCheck = true);
	};

	class ShaderUtils
	{
	public:
		/**
		 * \brief Preprocess Shader
		 * \param sourceName 
		 * \param kind 
		 * \param source 
		 * \return Returns GLSL shader source text after preprocessing.
		 */
		static std::string PreprocessShader(const std::string& sourceName,
		                                    shaderc_shader_kind kind,
		                                    const std::string& source);

		/**
		 * \brief Compiles a shader to SPIR-V assembly.
		 * \param sourceName 
		 * \param kind 
		 * \param source 
		 * \param optimize 
		 * \return Returns the assembly text as a string.
		 */
		static std::string CompileFileToAssembly(const std::string& sourceName,
		                                         shaderc_shader_kind kind,
		                                         const std::string& source,
		                                         bool optimize = false);
		/**
		 * \brief Compiles a shader to a SPIR-V binary.
		 * \param sourceName 
		 * \param kind 
		 * \param source 
		 * \param optimize 
		 * \return Returns the binary as a vector of 32-bit words.
		 */
		static std::vector<uint32_t> CompileFile(const std::string& sourceName,
		                                  shaderc_shader_kind kind,
		                                  const std::string& source,
		                                  bool optimize = false);

		static std::vector<uint32_t> Get(const std::string& sourceName,
			shaderc_shader_kind kind,
			const std::string& source,
			bool optimize = false);
	};

	class SphereMeshGenerator {
	public:
		static void Icosahedron(std::vector<glm::vec3>& vertices, std::vector<glm::uvec3>& triangles);
	};
}

namespace ImGui
{
	IMGUI_API bool Splitter(
		bool split_vertically,
		float thickness,
		float& size1,
		float& size2,
		float min_size1,
		float min_size2,
		float splitter_long_axis_size = -1.0f);

	IMGUI_API bool Combo(const std::string& label, const std::vector<std::string>& items, unsigned& currentSelection, ImGuiComboFlags flags = 0);
	IMGUI_API bool Combo(const std::string& label, const std::vector<std::string>& items, int& currentSelection, ImGuiComboFlags flags = 0);
}
