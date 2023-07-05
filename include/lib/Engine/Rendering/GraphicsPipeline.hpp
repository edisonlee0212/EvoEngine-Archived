#pragma once
#include "shaderc/shaderc.h"
namespace EvoEngine
{
	class ShaderModule
	{
		shaderc_shader_kind m_shaderKind = shaderc_glsl_infer_from_source;
		VkShaderModule m_vkShaderModule = VK_NULL_HANDLE;
	public:
		void Create(shaderc_shader_kind shaderKind, const std::vector<char>& code);
		void Destroy();

		void Create(shaderc_shader_kind shaderKind, const std::string& code);

		VkShaderModule GetVkShaderModule() const;
	};

	class RenderPass
	{
		VkRenderPass m_vkRenderPass = VK_NULL_HANDLE;
	public:
		void Create(const VkRenderPassCreateInfo& renderPassCreateInfo);
		void Destroy();

		VkRenderPass GetVkRenderPass() const;
	};

	class PipelineLayout
	{
		VkPipelineLayout m_vkPipelineLayout = VK_NULL_HANDLE;
	public:
		void Create(const VkPipelineLayoutCreateInfo& pipelineLayoutCreateInfo);
		void Destroy();

		VkPipelineLayout GetVkPipelineLayout() const;
	};

	class GraphicsPipeline
	{
		VkPipeline m_vkGraphicsPipeline = VK_NULL_HANDLE;
	public:
		void Create(const VkGraphicsPipelineCreateInfo& graphicsPipelineCreateInfo);

		void Destroy();

		VkPipeline GetVkPipeline() const;
	};
}
