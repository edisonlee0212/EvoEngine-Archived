#pragma once
#include "GraphicsResources.hpp"
#include "IGeometry.hpp"
namespace EvoEngine
{
	class Shader;

	struct DescriptorSetLayoutBinding
	{
		VkDescriptorSetLayoutBinding m_binding = {};
		bool m_ready = false;
	};

#pragma region Pipeline Data

	struct PipelineShaderStage
	{
		VkPipelineShaderStageCreateFlags    m_flags;
		VkShaderStageFlagBits               m_stage;
		VkShaderModule                      m_module;
		std::string							m_name;
		std::optional<VkSpecializationInfo> m_specializationInfo;
		void Apply(const VkPipelineShaderStageCreateInfo& vkPipelineShaderStageCreateInfo);
	};

	struct PipelineVertexInputState
	{
		VkPipelineVertexInputStateCreateFlags m_flags;
		std::vector<VkVertexInputBindingDescription> m_vertexBindingDescriptions;
		std::vector<VkVertexInputAttributeDescription> m_vertexAttributeDescriptions;
		void Apply(const VkPipelineVertexInputStateCreateInfo& vkPipelineShaderStageCreateInfo);
	};
	struct PipelineInputAssemblyState
	{
		VkPipelineInputAssemblyStateCreateFlags    m_flags;
		VkPrimitiveTopology                        m_topology;
		VkBool32                                   m_primitiveRestartEnable;
		void Apply(const VkPipelineInputAssemblyStateCreateInfo& vkPipelineInputAssemblyStateCreateInfo);

	};

	struct PipelineTessellationState
	{
		VkPipelineTessellationStateCreateFlags    m_flags;
		uint32_t                                  m_patchControlPoints;
		void Apply(const VkPipelineTessellationStateCreateInfo& vkPipelineTessellationStateCreateInfo);

	};

	struct PipelineViewportState
	{
		VkPipelineViewportStateCreateFlags    m_flags;
		std::vector<VkViewport> m_viewports;
		std::vector<VkRect2D> m_scissors;
		void Apply(const VkPipelineViewportStateCreateInfo& vkPipelineViewportStateCreateInfo);
	};

	struct PipelineRasterizationState
	{
		VkPipelineRasterizationStateCreateFlags    m_flags;
		VkBool32                                   m_depthClampEnable;
		VkBool32                                   m_rasterizerDiscardEnable;
		VkPolygonMode                              m_polygonMode;
		VkCullModeFlags                            m_cullMode;
		VkFrontFace                                m_frontFace;
		VkBool32                                   m_depthBiasEnable;
		float                                      m_depthBiasConstantFactor;
		float                                      m_depthBiasClamp;
		float                                      m_depthBiasSlopeFactor;
		float                                      m_lineWidth;
		void Apply(const VkPipelineRasterizationStateCreateInfo& vkPipelineRasterizationStateCreateInfo);
	};

	struct PipelineMultisampleState
	{
		VkPipelineMultisampleStateCreateFlags    m_flags;
		VkSampleCountFlagBits                    m_rasterizationSamples;
		VkBool32                                 m_sampleShadingEnable;
		float                                    m_minSampleShading;
		std::optional<VkSampleMask>	m_sampleMask;
		VkBool32                                 m_alphaToCoverageEnable;
		VkBool32                                 m_alphaToOneEnable;
		void Apply(const VkPipelineMultisampleStateCreateInfo& vkPipelineMultisampleStateCreateInfo);
	};
	struct PipelineDepthStencilState
	{
		VkPipelineDepthStencilStateCreateFlags    m_flags;
		VkBool32                                  m_depthTestEnable;
		VkBool32                                  m_depthWriteEnable;
		VkCompareOp                               m_depthCompareOp;
		VkBool32                                  m_depthBoundsTestEnable;
		VkBool32                                  m_stencilTestEnable;
		VkStencilOpState                          m_front;
		VkStencilOpState                          m_back;
		float                                     m_minDepthBounds;
		float                                     m_maxDepthBounds;
		void Apply(const VkPipelineDepthStencilStateCreateInfo& vkPipelineDepthStencilStateCreateInfo);
	};
	struct PipelineColorBlendState
	{
		VkPipelineColorBlendStateCreateFlags          m_flags;
		VkBool32                                      m_logicOpEnable;
		VkLogicOp                                     m_logicOp;
		std::vector<VkPipelineColorBlendAttachmentState> m_attachments;
		float                                         m_blendConstants[4];
		void Apply(const VkPipelineColorBlendStateCreateInfo& vkPipelineColorBlendStateCreateInfo);
	};

	struct PipelineDynamicState
	{
		VkPipelineDynamicStateCreateFlags    m_flags;
		std::vector<VkDynamicState> m_dynamicStates;
		void Apply(const VkPipelineDynamicStateCreateInfo& vkPipelineDynamicStateCreateInfo);
	};
#pragma endregion

	class GraphicsPipeline
	{
		friend class Graphics;
		friend class RenderLayer;
		friend class GraphicsGlobalStates;

		std::unique_ptr<PipelineLayout> m_pipelineLayout = {};

		VkPipeline m_vkGraphicsPipeline = VK_NULL_HANDLE;
	public:
		std::vector<std::shared_ptr<DescriptorSetLayout>> m_descriptorSetLayouts;

		std::shared_ptr<Shader> m_vertexShader;
		std::shared_ptr<Shader> m_tessellationControlShader;
		std::shared_ptr<Shader> m_tessellationEvaluationShader;
		std::shared_ptr<Shader> m_geometryShader;
		std::shared_ptr<Shader> m_fragmentShader;
		GeometryType m_geometryType = GeometryType::Mesh;

		uint32_t m_viewMask;
		std::vector<VkFormat> m_colorAttachmentFormats;
		VkFormat m_depthAttachmentFormat;
		VkFormat m_stencilAttachmentFormat;

		std::vector<VkPushConstantRange> m_pushConstantRanges;


		void PreparePipeline();
		[[nodiscard]] bool PipelineReady() const;
		void Bind(VkCommandBuffer commandBuffer) const;
		void BindDescriptorSet(VkCommandBuffer commandBuffer, uint32_t firstSet, VkDescriptorSet descriptorSet) const;
		template<typename T>
		void PushConstant(VkCommandBuffer commandBuffer, size_t rangeIndex, const T& data);
	};

	template <typename T>
	void GraphicsPipeline::PushConstant(const VkCommandBuffer commandBuffer, const size_t rangeIndex, const T& data)
	{
		vkCmdPushConstants(commandBuffer, m_pipelineLayout->GetVkPipelineLayout(), 
			m_pushConstantRanges[rangeIndex].stageFlags, m_pushConstantRanges[rangeIndex].offset, m_pushConstantRanges[rangeIndex].size, &data);
	}
}
