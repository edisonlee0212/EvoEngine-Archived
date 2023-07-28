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
		std::unordered_map<uint32_t, DescriptorSetLayoutBinding> m_descriptorSetLayoutBindings;

		std::unique_ptr<DescriptorSetLayout> m_perPassLayout = {};

		std::unique_ptr<PipelineLayout> m_pipelineLayout = {};

		/**
		 * \brief You can link the shader with the layouts ready.
		 * The actual descriptor set can be set later.
		 */
		bool m_layoutReady = false;

		std::vector<VkDescriptorSet> m_perPassDescriptorSets = {};

		/**
		 * \brief You can only bind shaders after the descriptor sets are ready.
		 */
		bool m_descriptorSetsReady = false;
		void CheckDescriptorSetsReady();


		VkPipeline m_vkGraphicsPipeline = VK_NULL_HANDLE;
	public:
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
		void ClearDescriptorSets();

		[[maybe_unused]] void PushDescriptorBinding(uint32_t binding, VkDescriptorType type, VkShaderStageFlags stageFlags);

		void PreparePerPassLayouts();
		[[nodiscard]] bool LayoutSetsReady() const;

		
		void UpdateImageDescriptorBinding(uint32_t binding, const std::vector<VkDescriptorImageInfo>& imageInfos);
		void UpdateBufferDescriptorBinding(uint32_t binding, const std::vector<VkDescriptorBufferInfo>& bufferInfos);
		[[nodiscard]] bool DescriptorSetsReady() const;

		void PreparePipeline();
		[[nodiscard]] bool PipelineReady() const;
		void Bind(VkCommandBuffer commandBuffer) const;
		template<typename T>
		void PushConstant(VkCommandBuffer commandBuffer, size_t rangeIndex, const T& data);
	};

	template <typename T>
	void GraphicsPipeline::PushConstant(const VkCommandBuffer commandBuffer, const size_t rangeIndex, const T& data)
	{
		vkCmdPushConstants(commandBuffer, m_pipelineLayout->GetVkPipelineLayout(), 
			VK_SHADER_STAGE_ALL, m_pushConstantRanges[rangeIndex].offset, m_pushConstantRanges[rangeIndex].size, &data);
	}
}
