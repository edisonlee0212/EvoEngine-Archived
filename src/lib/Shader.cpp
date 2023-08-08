#include "Shader.hpp"
#include "Utilities.hpp"

using namespace EvoEngine;
bool Shader::Compiled() const
{
	return m_shaderModule != nullptr;
}

void Shader::Set(const ShaderType shaderType, const std::string& shaderCode)
{
	m_shaderType = shaderType;
	m_code = shaderCode;
	shaderc_shader_kind shaderKind;
	switch (shaderType)
	{
	case ShaderType::Task:
		shaderKind = shaderc_task_shader; break;
	case ShaderType::Mesh:
		shaderKind = shaderc_mesh_shader; break;
	case ShaderType::Vertex:
		shaderKind = shaderc_vertex_shader; break;
	case ShaderType::TessellationControl:
		shaderKind = shaderc_tess_control_shader; break;
	case ShaderType::TessellationEvaluation:
		shaderKind = shaderc_tess_evaluation_shader; break;
	case ShaderType::Geometry:
		shaderKind = shaderc_geometry_shader; break;
	case ShaderType::Fragment:
		shaderKind = shaderc_fragment_shader; break;
	case ShaderType::Compute:
		shaderKind = shaderc_compute_shader; break;
	case ShaderType::Unknown:
		shaderKind = shaderc_glsl_infer_from_source; break;
	}

	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	const auto binary = ShaderUtils::CompileFile("Shader", shaderKind, m_code);
	createInfo.pCode = binary.data();
	createInfo.codeSize = binary.size() * sizeof(uint32_t);
	m_shaderModule = std::make_unique<ShaderModule>(createInfo);
}

const std::unique_ptr<ShaderModule>& Shader::GetShaderModule() const
{
	return m_shaderModule;
}