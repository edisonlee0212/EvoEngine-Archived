#pragma once
#include "shaderc/shaderc.h"
namespace EvoEngine
{
	class IGraphicsResource
	{
	protected:
		IGraphicsResource() = default;
	public:
		IGraphicsResource& operator=(IGraphicsResource&) = delete;
		IGraphicsResource& operator=(const IGraphicsResource&) = delete;
		virtual ~IGraphicsResource() = default;
	};
	class Fence final : public IGraphicsResource
	{
		VkFence m_vkFence = VK_NULL_HANDLE;
		VkFenceCreateFlags m_flags = {};
	public:
		explicit Fence(const VkFenceCreateInfo& vkFenceCreateInfo);
		~Fence() override;

		[[nodiscard]] VkFence GetVkFence() const;
	};

	class Semaphore final : public IGraphicsResource
	{
		VkSemaphore m_vkSemaphore = VK_NULL_HANDLE;
		VkSemaphoreCreateFlags m_flags = {};
	public:
		explicit Semaphore(const VkSemaphoreCreateInfo& semaphoreCreateInfo);
		~Semaphore() override;
		[[nodiscard]] VkSemaphore GetVkSemaphore() const;
	};

	class ImageView final : public IGraphicsResource
	{
		VkImageView m_vkImageView = VK_NULL_HANDLE;

		VkImageViewCreateFlags     m_flags;
		VkImage                    m_image;
		VkImageViewType            m_viewType;
		VkFormat                   m_format;
		VkComponentMapping         m_components;
		VkImageSubresourceRange    m_subresourceRange;
	public:
		explicit ImageView(const VkImageViewCreateInfo& imageViewCreateInfo);
		~ImageView() override;

		[[nodiscard]] VkImageView GetVkImageView() const;
	};

	class Framebuffer final : public IGraphicsResource
	{
		VkFramebuffer m_vkFramebuffer = VK_NULL_HANDLE;

		VkFramebufferCreateFlags m_flags;
		VkRenderPass m_renderPass;
		std::vector<VkImageView> m_attachments;
		uint32_t m_width;
		uint32_t m_height;
		uint32_t m_layers;

	public:
		explicit Framebuffer(const VkFramebufferCreateInfo& framebufferCreateInfo);
		~Framebuffer() override;

		[[nodiscard]] VkFramebuffer GetVkFrameBuffer() const;
	};

	class Swapchain final : public IGraphicsResource
	{
		VkSwapchainKHR m_vkSwapchain = VK_NULL_HANDLE;
		std::vector<VkImage> m_vkImages;

		VkSwapchainCreateFlagsKHR m_flags;
		VkSurfaceKHR m_surface;
		uint32_t m_minImageCount;
		VkFormat                         m_imageFormat;
		VkColorSpaceKHR                  m_imageColorSpace;
		VkExtent2D                       m_imageExtent;
		uint32_t                         m_imageArrayLayers;
		VkImageUsageFlags                m_imageUsage;
		VkSharingMode                    m_imageSharingMode;
		std::vector<uint32_t>			m_queueFamilyIndices;
		VkSurfaceTransformFlagBitsKHR    m_preTransform;
		VkCompositeAlphaFlagBitsKHR      m_compositeAlpha;
		VkPresentModeKHR                 m_presentMode;
		VkBool32                         m_clipped;

		std::vector<VkImageView> m_vkImageViews;
	public:
		explicit Swapchain(const VkSwapchainCreateInfoKHR& swapchainCreateInfo);
		~Swapchain() override;

		[[nodiscard]] VkSwapchainKHR GetVkSwapchain() const;

		[[nodiscard]] const std::vector<VkImage>& GetVkImages() const;
		[[nodiscard]] const std::vector<VkImageView>& GetVkImageViews() const;
		[[nodiscard]] VkFormat GetImageFormat() const;

		[[nodiscard]] VkExtent2D GetImageExtent() const;
	};

	class ShaderModule final : public IGraphicsResource
	{
		shaderc_shader_kind m_shaderKind = shaderc_glsl_infer_from_source;
		VkShaderModule m_vkShaderModule = VK_NULL_HANDLE;

		std::string m_code;
	public:
		~ShaderModule() override;

		ShaderModule(shaderc_shader_kind shaderKind, const std::string& code);

		[[nodiscard]] VkShaderModule GetVkShaderModule() const;
	};
	struct SubpassDescription
	{
		VkSubpassDescriptionFlags m_flags;
		VkPipelineBindPoint m_pipelineBindPoint;
		std::vector<VkAttachmentReference> m_inputAttachments;
		std::vector<VkAttachmentReference> m_colorAttachments;
		std::vector<VkAttachmentReference> m_resolveAttachments;
		std::optional<VkAttachmentReference> m_depthStencilAttachment;
		std::vector<uint32_t> m_preserveAttachment;
	};

	class RenderPass final : public IGraphicsResource
	{
		VkRenderPass m_vkRenderPass = VK_NULL_HANDLE;

		VkRenderPassCreateFlags m_flags;
		std::vector<VkAttachmentDescription> m_attachments;
		std::vector<SubpassDescription> m_subpasses;
		std::vector<VkSubpassDependency> m_dependencies;
	public:
		RenderPass(const VkRenderPassCreateInfo& renderPassCreateInfo);
		~RenderPass() override;

		[[nodiscard]] VkRenderPass GetVkRenderPass() const;
	};

	class PipelineLayout final : public IGraphicsResource
	{
		VkPipelineLayout m_vkPipelineLayout = VK_NULL_HANDLE;

		VkPipelineLayoutCreateFlags     m_flags;
		std::vector<VkDescriptorSetLayout> m_setLayouts;
		std::vector<VkPushConstantRange> m_pushConstantRanges;
	public:
		PipelineLayout(const VkPipelineLayoutCreateInfo& pipelineLayoutCreateInfo);
		~PipelineLayout() override;

		[[nodiscard]] VkPipelineLayout GetVkPipelineLayout() const;
	};

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

	class GraphicsPipeline final : public IGraphicsResource
	{
		VkPipeline m_vkGraphicsPipeline = VK_NULL_HANDLE;

		VkPipelineCreateFlags                            m_flags;
		std::vector<PipelineShaderStage> m_stages;
		std::optional<PipelineVertexInputState> m_vertexInputState;
		std::optional<PipelineInputAssemblyState> m_inputAssemblyState;
		std::optional<PipelineTessellationState> m_tessellationState;
		std::optional<PipelineViewportState> m_viewportState;
		std::optional<PipelineRasterizationState> m_rasterizationState;
		std::optional<PipelineMultisampleState> m_multisampleState;
		std::optional<PipelineDepthStencilState> m_depthStencilState;
		std::optional<PipelineColorBlendState> m_colorBlendState;
		std::optional<PipelineDynamicState> m_dynamicState;

		VkPipelineLayout                                 m_layout;
		VkRenderPass                                     m_renderPass;
		uint32_t                                         m_subpass;
		VkPipeline                                       m_basePipelineHandle;
		int32_t                                          m_basePipelineIndex;
	public:

		explicit GraphicsPipeline(const VkGraphicsPipelineCreateInfo& graphicsPipelineCreateInfo);

		~GraphicsPipeline() override;

		[[nodiscard]] VkPipeline GetVkPipeline() const;
	};

	class CommandPool final : public IGraphicsResource
	{
		VkCommandPool m_vkCommandPool = VK_NULL_HANDLE;
	public:
		explicit CommandPool(const VkCommandPoolCreateInfo& commandPoolCreateInfo);

		~CommandPool() override;

		[[nodiscard]] VkCommandPool GetVkCommandPool() const;
	};

	class Buffer final : public IGraphicsResource
	{
		VkBuffer m_vkBuffer = VK_NULL_HANDLE;
		VmaAllocation m_vmaAllocation = VK_NULL_HANDLE;
		VmaAllocationInfo m_vmaAllocationInfo = {};
	public:
		explicit Buffer(const VkBufferCreateInfo& bufferCreateInfo);
		~Buffer() override;
		Buffer(const VkBufferCreateInfo& bufferCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo);

		void Copy(const Buffer& srcBuffer, VkDeviceSize size, VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0) const;

		[[nodiscard]] VkBuffer GetVkBuffer() const;

		[[nodiscard]] VmaAllocation GetVmaAllocation() const;

		[[nodiscard]] const VmaAllocationInfo& GetVmaAllocationInfo() const;
	};

	class Image final : public IGraphicsResource
	{
		VkImage m_vkImage = VK_NULL_HANDLE;
		VmaAllocation m_vmaAllocation = VK_NULL_HANDLE;
		VmaAllocationInfo m_vmaAllocationInfo = {};

		VkImageCreateFlags       m_flags;
		VkImageType              m_imageType;
		VkFormat                 m_format;
		VkExtent3D               m_extent;
		uint32_t                 m_mipLevels;
		uint32_t                 m_arrayLayers;
		VkSampleCountFlagBits    m_samples;
		VkImageTiling            m_tiling;
		VkImageUsageFlags        m_usage;
		VkSharingMode            m_sharingMode;
		std::vector<uint32_t>	 m_queueFamilyIndices;
		VkImageLayout            m_initialLayout;

		VkImageLayout			 m_layout;
	public:
		explicit Image(const VkImageCreateInfo& imageCreateInfo);
		Image(const VkImageCreateInfo& imageCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo);
		bool HasStencilComponent() const;
		~Image() override;
		void TransitionImageLayout(VkImageLayout newLayout);
		void Copy(const VkBuffer& srcBuffer, VkDeviceSize srcOffset = 0) const;

		[[nodiscard]] VkImage GetVkImage() const;
		[[nodiscard]] VkFormat GetFormat() const;
		[[nodiscard]] VmaAllocation GetVmaAllocation() const;
		[[nodiscard]] VkExtent3D GetExtent() const;
		[[nodiscard]] VkImageLayout GetLayout() const;
		[[nodiscard]] const VmaAllocationInfo& GetVmaAllocationInfo() const;
	};

	class Sampler final : public IGraphicsResource
	{
		VkSampler m_vkSampler;
	public:
		explicit Sampler(const VkSamplerCreateInfo& samplerCreateInfo);
		~Sampler() override;
		[[nodiscard]] VkSampler GetVkSampler() const;
	};

	class DescriptorSetLayout final : public IGraphicsResource
	{
		VkDescriptorSetLayout m_vkDescriptorSetLayout = VK_NULL_HANDLE;
	public:
		explicit DescriptorSetLayout(const VkDescriptorSetLayoutCreateInfo& descriptorSetLayoutCreateInfo);
		~DescriptorSetLayout() override;
		[[nodiscard]] VkDescriptorSetLayout GetVkDescriptorSetLayout() const;
	};

	class DescriptorPool final : public IGraphicsResource
	{
		VkDescriptorPool m_vkDescriptorPool = VK_NULL_HANDLE;
	public:
		explicit DescriptorPool(const VkDescriptorPoolCreateInfo& descriptorPoolCreateInfo);
		~DescriptorPool() override;
		[[nodiscard]] VkDescriptorPool GetVkDescriptorPool() const;
	};

	class ShaderEXT final : public IGraphicsResource
	{
		VkShaderEXT m_shaderExt = VK_NULL_HANDLE;

		VkShaderCreateFlagsEXT          m_flags;
		VkShaderStageFlagBits           m_stage;
		VkShaderStageFlags              m_nextStage;
		VkShaderCodeTypeEXT             m_codeType;
		std::string						m_name;
		std::vector<VkDescriptorSetLayout> m_setLayouts;
		std::vector<VkPushConstantRange>	m_pushConstantRanges;
		std::optional<VkSpecializationInfo> m_specializationInfo;
	public:
		explicit ShaderEXT(const VkShaderCreateInfoEXT& shaderCreateInfoExt);
		~ShaderEXT() override;
		[[nodiscard]] const VkShaderEXT &GetVkShaderEXT() const;
	};
}
