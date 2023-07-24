#include "Shader.hpp"

#include "Utilities.hpp"

using namespace EvoEngine;

void Shader::Set(const ShaderType shaderType, const std::string& shaderCode)
{
	shaderc_shader_kind shaderKind = shaderc_glsl_infer_from_source;
	switch (shaderType)
	{
	case ShaderType::Vertex: shaderKind = shaderc_vertex_shader; break;
	case ShaderType::TessellationControl: shaderKind = shaderc_tess_control_shader; break;
	case ShaderType::TessellationEvaluation: shaderKind = shaderc_tess_evaluation_shader; break;
	case ShaderType::Geometry: shaderKind = shaderc_geometry_shader; break;
	case ShaderType::Fragment: shaderKind = shaderc_fragment_shader; break;
	case ShaderType::Compute: shaderKind = shaderc_compute_shader; break;
	case ShaderType::Unknown: shaderKind = shaderc_glsl_infer_from_source; break;

	}
	const auto fragShaderBinary = ShaderUtils::CompileFile("Shader", shaderKind, shaderCode);
	VkShaderCreateInfoEXT fragShaderCreateInfo{};
	fragShaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
	fragShaderCreateInfo.pNext = nullptr;
	fragShaderCreateInfo.flags = 0;
	fragShaderCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderCreateInfo.nextStage = 0;
	fragShaderCreateInfo.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
	fragShaderCreateInfo.codeSize = fragShaderBinary.size() * sizeof(uint32_t);
	fragShaderCreateInfo.pCode = fragShaderBinary.data();
	fragShaderCreateInfo.pName = "main";
	fragShaderCreateInfo.setLayoutCount = 0;
	fragShaderCreateInfo.pSetLayouts = nullptr;
	fragShaderCreateInfo.pushConstantRangeCount = 0;
	fragShaderCreateInfo.pPushConstantRanges = nullptr;
	fragShaderCreateInfo.pSpecializationInfo = nullptr;

	m_shaderEXT.reset();
	m_shaderEXT = std::make_unique<ShaderEXT>(fragShaderCreateInfo);

	m_shaderType = shaderType;
	m_code = shaderCode;
}

const std::unique_ptr<ShaderEXT>& Shader::GetShaderEXT() const
{
	return m_shaderEXT;
}
