#include "GraphicsResources.hpp"

#include "Graphics.hpp"
#include "Utilities.hpp"

using namespace EvoEngine;

template<typename T>
void ApplyVector(std::vector<T>& target, uint32_t size, const T* data)
{
	if (size == 0 || data == nullptr) return;
	target.resize(size);
	memcpy(target.data(), data,
		sizeof(T) * size);
}

Fence::Fence(const VkFenceCreateInfo& vkFenceCreateInfo)
{
	if (vkCreateFence(Graphics::GetVkDevice(), &vkFenceCreateInfo, nullptr, &m_vkFence) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create vkFence!");
	}
	m_flags = vkFenceCreateInfo.flags;
}

Fence::~Fence()
{
	if (m_vkFence != VK_NULL_HANDLE)
	{
		vkDestroyFence(Graphics::GetVkDevice(), m_vkFence, nullptr);
		m_vkFence = nullptr;
	}
}

VkFence Fence::GetVkFence() const
{
	return m_vkFence;
}

Semaphore::Semaphore(const VkSemaphoreCreateInfo& semaphoreCreateInfo)
{
	if (vkCreateSemaphore(Graphics::GetVkDevice(), &semaphoreCreateInfo, nullptr, &m_vkSemaphore) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create vkSemaphore!");
	}
	m_flags = semaphoreCreateInfo.flags;
}

Semaphore::~Semaphore()
{
	if (m_vkSemaphore != VK_NULL_HANDLE)
	{
		vkDestroySemaphore(Graphics::GetVkDevice(), m_vkSemaphore, nullptr);
		m_vkSemaphore = VK_NULL_HANDLE;
	}
}



VkSemaphore Semaphore::GetVkSemaphore() const
{
	return m_vkSemaphore;
}

Swapchain::Swapchain(const VkSwapchainCreateInfoKHR& swapChainCreateInfo)
{
	const auto& device = Graphics::GetVkDevice();
	if (vkCreateSwapchainKHR(Graphics::GetVkDevice(), &swapChainCreateInfo, nullptr, &m_vkSwapchain) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create swap chain!");
	}
	uint32_t imageCount = 0;
	vkGetSwapchainImagesKHR(device, m_vkSwapchain, &imageCount, nullptr);
	m_vkImages.resize(imageCount);
	vkGetSwapchainImagesKHR(device, m_vkSwapchain, &imageCount, m_vkImages.data());
	m_flags = swapChainCreateInfo.flags;
	m_surface = swapChainCreateInfo.surface;
	m_minImageCount = swapChainCreateInfo.minImageCount;
	m_imageFormat = swapChainCreateInfo.imageFormat;
	m_imageExtent = swapChainCreateInfo.imageExtent;
	m_imageArrayLayers = swapChainCreateInfo.imageArrayLayers;
	m_imageUsage = swapChainCreateInfo.imageUsage;
	m_imageSharingMode = swapChainCreateInfo.imageSharingMode;
	ApplyVector(m_queueFamilyIndices, swapChainCreateInfo.queueFamilyIndexCount, swapChainCreateInfo.pQueueFamilyIndices);
	m_preTransform = swapChainCreateInfo.preTransform;
	m_compositeAlpha = swapChainCreateInfo.compositeAlpha;
	m_presentMode = swapChainCreateInfo.presentMode;
	m_clipped = swapChainCreateInfo.clipped;

	m_vkImageViews.resize(m_vkImages.size());
	for (size_t i = 0; i < m_vkImages.size(); i++) {
		VkImageViewCreateInfo imageViewCreateInfo{};
		imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCreateInfo.image = m_vkImages[i];
		imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCreateInfo.format = m_imageFormat;
		imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
		imageViewCreateInfo.subresourceRange.levelCount = 1;
		imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
		imageViewCreateInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(Graphics::GetVkDevice(), &imageViewCreateInfo, nullptr, &m_vkImageViews[i]) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create image views!");
		}
	}
}

Swapchain::~Swapchain()
{
	for (const auto& imageView : m_vkImageViews) {
		if (imageView != VK_NULL_HANDLE) {
			vkDestroyImageView(Graphics::GetVkDevice(), imageView, nullptr);
		}
	}
	m_vkImageViews.clear();
	if (m_vkSwapchain != VK_NULL_HANDLE) {
		vkDestroySwapchainKHR(Graphics::GetVkDevice(), m_vkSwapchain, nullptr);
		m_vkSwapchain = VK_NULL_HANDLE;
	}
}

VkSwapchainKHR Swapchain::GetVkSwapchain() const
{
	return m_vkSwapchain;
}

const std::vector<VkImage>& Swapchain::GetVkImages() const
{
	return m_vkImages;
}

const std::vector<VkImageView>& Swapchain::GetVkImageViews() const
{
	return m_vkImageViews;
}

VkFormat Swapchain::GetImageFormat() const
{
	return m_imageFormat;
}

VkExtent2D Swapchain::GetImageExtent() const
{
	return m_imageExtent;
}



ImageView::ImageView(const VkImageViewCreateInfo& imageViewCreateInfo)
{
	if (vkCreateImageView(Graphics::GetVkDevice(), &imageViewCreateInfo, nullptr, &m_vkImageView) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image views!");
	}
	m_image = imageViewCreateInfo.image;
	m_flags = imageViewCreateInfo.flags;
	m_viewType = imageViewCreateInfo.viewType;
	m_format = imageViewCreateInfo.format;
	m_components = imageViewCreateInfo.components;
	m_subresourceRange = imageViewCreateInfo.subresourceRange;
}

ImageView::~ImageView()
{
	if (m_vkImageView != VK_NULL_HANDLE) {
		vkDestroyImageView(Graphics::GetVkDevice(), m_vkImageView, nullptr);
		m_vkImageView = VK_NULL_HANDLE;
	}
}

VkImageView ImageView::GetVkImageView() const
{
	return m_vkImageView;
}

Framebuffer::Framebuffer(const VkFramebufferCreateInfo& framebufferCreateInfo)
{
	if (vkCreateFramebuffer(Graphics::GetVkDevice(), &framebufferCreateInfo, nullptr, &m_vkFramebuffer) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create framebuffer!");
	}
	m_flags = framebufferCreateInfo.flags;
	m_renderPass = framebufferCreateInfo.renderPass;
	ApplyVector(m_attachments, framebufferCreateInfo.attachmentCount, framebufferCreateInfo.pAttachments);
	m_width = framebufferCreateInfo.width;
	m_height = framebufferCreateInfo.height;
	m_layers = framebufferCreateInfo.layers;
}

Framebuffer::~Framebuffer()
{
	if (m_vkFramebuffer != VK_NULL_HANDLE) {
		vkDestroyFramebuffer(Graphics::GetVkDevice(), m_vkFramebuffer, nullptr);
		m_vkFramebuffer = VK_NULL_HANDLE;
	}
}

VkFramebuffer Framebuffer::GetVkFrameBuffer() const
{
	return m_vkFramebuffer;
}

ShaderModule::~ShaderModule()
{
	if (m_vkShaderModule != VK_NULL_HANDLE) {
		vkDestroyShaderModule(Graphics::GetVkDevice(), m_vkShaderModule, nullptr);
		m_vkShaderModule = VK_NULL_HANDLE;
	}
}

ShaderModule::ShaderModule(shaderc_shader_kind shaderKind, const std::string& code)
{
	m_shaderKind = shaderKind;
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	const auto binary = ShaderUtils::CompileFile("Shader", m_shaderKind, code);
	createInfo.pCode = binary.data();
	createInfo.codeSize = binary.size() * sizeof(uint32_t);
	if (vkCreateShaderModule(Graphics::GetVkDevice(), &createInfo, nullptr, &m_vkShaderModule) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create shader module!");
	}

	m_code = code;
}

VkShaderModule ShaderModule::GetVkShaderModule() const
{
	return m_vkShaderModule;
}


RenderPass::RenderPass(const VkRenderPassCreateInfo& renderPassCreateInfo)
{
	if (vkCreateRenderPass(Graphics::GetVkDevice(), &renderPassCreateInfo, nullptr, &m_vkRenderPass) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create render pass!");
	}
	m_flags = renderPassCreateInfo.flags;

	ApplyVector(m_attachments, renderPassCreateInfo.attachmentCount, renderPassCreateInfo.pAttachments);
	m_subpasses.resize(renderPassCreateInfo.subpassCount);
	for(int i = 0; i < renderPassCreateInfo.subpassCount; i++)
	{
		auto& subpass = m_subpasses[i];
		const auto& subpassInfo = renderPassCreateInfo.pSubpasses[i];
		subpass.m_flags = subpassInfo.flags;
		subpass.m_pipelineBindPoint = subpassInfo.pipelineBindPoint;
		ApplyVector(subpass.m_inputAttachments, subpassInfo.inputAttachmentCount, subpassInfo.pInputAttachments);
		ApplyVector(subpass.m_colorAttachments, subpassInfo.colorAttachmentCount, subpassInfo.pColorAttachments);
		if(subpassInfo.pResolveAttachments != VK_NULL_HANDLE) ApplyVector(subpass.m_resolveAttachments, subpassInfo.colorAttachmentCount, subpassInfo.pResolveAttachments);
		if(subpassInfo.pDepthStencilAttachment != VK_NULL_HANDLE)
			subpass.m_depthStencilAttachment = *subpassInfo.pDepthStencilAttachment;
		ApplyVector(subpass.m_preserveAttachment, subpassInfo.preserveAttachmentCount, subpassInfo.pPreserveAttachments);
	}

	m_dependencies.resize(renderPassCreateInfo.dependencyCount);
	memcpy(m_dependencies.data(), renderPassCreateInfo.pDependencies,
		sizeof(VkSubpassDependency) * renderPassCreateInfo.dependencyCount);
}

RenderPass::~RenderPass()
{
	if (m_vkRenderPass != VK_NULL_HANDLE) {
		vkDestroyRenderPass(Graphics::GetVkDevice(), m_vkRenderPass, nullptr);
		m_vkRenderPass = VK_NULL_HANDLE;
	}
}

VkRenderPass RenderPass::GetVkRenderPass() const
{
	return m_vkRenderPass;
}

PipelineLayout::PipelineLayout(const VkPipelineLayoutCreateInfo& pipelineLayoutCreateInfo)
{
	if (vkCreatePipelineLayout(Graphics::GetVkDevice(), &pipelineLayoutCreateInfo, nullptr, &m_vkPipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create pipeline layout!");
	}

	m_flags = pipelineLayoutCreateInfo.flags;
	ApplyVector(m_setLayouts, pipelineLayoutCreateInfo.setLayoutCount, pipelineLayoutCreateInfo.pSetLayouts);
	ApplyVector(m_pushConstantRanges, pipelineLayoutCreateInfo.pushConstantRangeCount, pipelineLayoutCreateInfo.pPushConstantRanges);
}

PipelineLayout::~PipelineLayout()
{
	if (m_vkPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(Graphics::GetVkDevice(), m_vkPipelineLayout, nullptr);
		m_vkPipelineLayout = VK_NULL_HANDLE;
	}
}

VkPipelineLayout PipelineLayout::GetVkPipelineLayout() const
{
	return m_vkPipelineLayout;
}

void PipelineShaderStage::Apply(const VkPipelineShaderStageCreateInfo& vkPipelineShaderStageCreateInfo)
{
	m_flags = vkPipelineShaderStageCreateInfo.flags;
	m_stage = vkPipelineShaderStageCreateInfo.stage;
	m_module = vkPipelineShaderStageCreateInfo.module;
	m_name = vkPipelineShaderStageCreateInfo.pName;
	if(vkPipelineShaderStageCreateInfo.pSpecializationInfo != VK_NULL_HANDLE)
	{
		m_specializationInfo = *vkPipelineShaderStageCreateInfo.pSpecializationInfo;
	}
}

void PipelineVertexInputState::Apply(const VkPipelineVertexInputStateCreateInfo& vkPipelineShaderStageCreateInfo)
{
	m_flags = vkPipelineShaderStageCreateInfo.flags;
	ApplyVector(m_vertexBindingDescriptions, vkPipelineShaderStageCreateInfo.vertexBindingDescriptionCount, vkPipelineShaderStageCreateInfo.pVertexBindingDescriptions);
	ApplyVector(m_vertexAttributeDescriptions, vkPipelineShaderStageCreateInfo.vertexAttributeDescriptionCount, vkPipelineShaderStageCreateInfo.pVertexAttributeDescriptions);
}

void PipelineInputAssemblyState::Apply(
	const VkPipelineInputAssemblyStateCreateInfo& vkPipelineInputAssemblyStateCreateInfo)
{
	m_flags = vkPipelineInputAssemblyStateCreateInfo.flags;
	m_topology = vkPipelineInputAssemblyStateCreateInfo.topology;
	m_primitiveRestartEnable = vkPipelineInputAssemblyStateCreateInfo.primitiveRestartEnable;
}

void PipelineTessellationState::Apply(const VkPipelineTessellationStateCreateInfo& vkPipelineTessellationStateCreateInfo)
{
	m_flags = vkPipelineTessellationStateCreateInfo.flags;
	m_patchControlPoints = vkPipelineTessellationStateCreateInfo.patchControlPoints;
}

void PipelineViewportState::Apply(const VkPipelineViewportStateCreateInfo& vkPipelineViewportStateCreateInfo)
{
	m_flags = vkPipelineViewportStateCreateInfo.flags;
	ApplyVector(m_viewports, vkPipelineViewportStateCreateInfo.viewportCount, vkPipelineViewportStateCreateInfo.pViewports);
	ApplyVector(m_scissors, vkPipelineViewportStateCreateInfo.scissorCount, vkPipelineViewportStateCreateInfo.pScissors);
}

void PipelineRasterizationState::Apply(
	const VkPipelineRasterizationStateCreateInfo& vkPipelineRasterizationStateCreateInfo)
{
	m_flags = vkPipelineRasterizationStateCreateInfo.flags;
	m_depthClampEnable = vkPipelineRasterizationStateCreateInfo.depthClampEnable;
	m_rasterizerDiscardEnable = vkPipelineRasterizationStateCreateInfo.rasterizerDiscardEnable;
	m_polygonMode = vkPipelineRasterizationStateCreateInfo.polygonMode;
	m_cullMode = vkPipelineRasterizationStateCreateInfo.cullMode;
	m_frontFace = vkPipelineRasterizationStateCreateInfo.frontFace;
	m_depthBiasEnable = vkPipelineRasterizationStateCreateInfo.depthBiasEnable;
	m_depthBiasConstantFactor = vkPipelineRasterizationStateCreateInfo.depthBiasConstantFactor;
	m_depthBiasClamp = vkPipelineRasterizationStateCreateInfo.depthBiasClamp;
	m_depthBiasSlopeFactor = vkPipelineRasterizationStateCreateInfo.depthBiasSlopeFactor;
	m_lineWidth = vkPipelineRasterizationStateCreateInfo.lineWidth;
}

void PipelineMultisampleState::Apply(const VkPipelineMultisampleStateCreateInfo& vkPipelineMultisampleStateCreateInfo)
{
	m_flags = vkPipelineMultisampleStateCreateInfo.flags;
	m_rasterizationSamples = vkPipelineMultisampleStateCreateInfo.rasterizationSamples;
	m_sampleShadingEnable = vkPipelineMultisampleStateCreateInfo.sampleShadingEnable;
	m_minSampleShading = vkPipelineMultisampleStateCreateInfo.minSampleShading;
	if (vkPipelineMultisampleStateCreateInfo.pSampleMask) m_sampleMask = *vkPipelineMultisampleStateCreateInfo.pSampleMask;
	m_alphaToCoverageEnable = vkPipelineMultisampleStateCreateInfo.alphaToCoverageEnable;
	m_alphaToOneEnable = vkPipelineMultisampleStateCreateInfo.alphaToOneEnable;
}

void PipelineDepthStencilState::Apply(
	const VkPipelineDepthStencilStateCreateInfo& vkPipelineDepthStencilStateCreateInfo)
{
	m_flags = vkPipelineDepthStencilStateCreateInfo.flags;
	m_depthTestEnable = vkPipelineDepthStencilStateCreateInfo.depthTestEnable;
	m_depthWriteEnable = vkPipelineDepthStencilStateCreateInfo.depthWriteEnable;
	m_depthCompareOp = vkPipelineDepthStencilStateCreateInfo.depthCompareOp;
	m_depthBoundsTestEnable = vkPipelineDepthStencilStateCreateInfo.depthBoundsTestEnable;
	m_stencilTestEnable = vkPipelineDepthStencilStateCreateInfo.stencilTestEnable;
	m_front = vkPipelineDepthStencilStateCreateInfo.front;
	m_back = vkPipelineDepthStencilStateCreateInfo.back;
	m_minDepthBounds = vkPipelineDepthStencilStateCreateInfo.minDepthBounds;
	m_maxDepthBounds = vkPipelineDepthStencilStateCreateInfo.maxDepthBounds;
}

void PipelineColorBlendState::Apply(const VkPipelineColorBlendStateCreateInfo& vkPipelineColorBlendStateCreateInfo)
{
	m_flags = vkPipelineColorBlendStateCreateInfo.flags;
	m_logicOpEnable = vkPipelineColorBlendStateCreateInfo.logicOpEnable;
	m_logicOp = vkPipelineColorBlendStateCreateInfo.logicOp;
	ApplyVector(m_attachments, vkPipelineColorBlendStateCreateInfo.attachmentCount, vkPipelineColorBlendStateCreateInfo.pAttachments);
	m_blendConstants[0] = vkPipelineColorBlendStateCreateInfo.blendConstants[0];
	m_blendConstants[1] = vkPipelineColorBlendStateCreateInfo.blendConstants[1];
	m_blendConstants[2] = vkPipelineColorBlendStateCreateInfo.blendConstants[2];
	m_blendConstants[3] = vkPipelineColorBlendStateCreateInfo.blendConstants[3];
}

void PipelineDynamicState::Apply(const VkPipelineDynamicStateCreateInfo& vkPipelineDynamicStateCreateInfo)
{
	m_flags = vkPipelineDynamicStateCreateInfo.flags;
	ApplyVector(m_dynamicStates, vkPipelineDynamicStateCreateInfo.dynamicStateCount, vkPipelineDynamicStateCreateInfo.pDynamicStates);
}


GraphicsPipeline::GraphicsPipeline(const VkGraphicsPipelineCreateInfo& graphicsPipelineCreateInfo)
{
	if (vkCreateGraphicsPipelines(Graphics::GetVkDevice(), VK_NULL_HANDLE, 1,
		&graphicsPipelineCreateInfo, nullptr, &m_vkGraphicsPipeline) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create graphics pipeline!");
	}

	m_flags = graphicsPipelineCreateInfo.flags;
	m_stages.resize(graphicsPipelineCreateInfo.stageCount);
	for(int i = 0; i < graphicsPipelineCreateInfo.stageCount; i++)
	{
		m_stages[i].Apply(graphicsPipelineCreateInfo.pStages[i]);
	}
	if (graphicsPipelineCreateInfo.pVertexInputState) {
		m_vertexInputState.emplace();
		m_vertexInputState.value().Apply(*graphicsPipelineCreateInfo.pVertexInputState);
	}
	if (graphicsPipelineCreateInfo.pInputAssemblyState) {
		m_inputAssemblyState.emplace();
		m_inputAssemblyState.value().Apply(*graphicsPipelineCreateInfo.pInputAssemblyState);
	}
	if (graphicsPipelineCreateInfo.pTessellationState) {
		m_tessellationState.emplace();
		m_tessellationState.value().Apply(*graphicsPipelineCreateInfo.pTessellationState);
	}
	if (graphicsPipelineCreateInfo.pViewportState) {
		m_viewportState.emplace();
		m_viewportState.value().Apply(*graphicsPipelineCreateInfo.pViewportState);
	}
	if (graphicsPipelineCreateInfo.pRasterizationState) {
		m_rasterizationState.emplace();
		m_rasterizationState.value().Apply(*graphicsPipelineCreateInfo.pRasterizationState);
	}
	if (graphicsPipelineCreateInfo.pMultisampleState) {
		m_multisampleState.emplace();
		m_multisampleState.value().Apply(*graphicsPipelineCreateInfo.pMultisampleState);
	}
	if (graphicsPipelineCreateInfo.pDepthStencilState) {
		m_depthStencilState.emplace();
		m_depthStencilState.value().Apply(*graphicsPipelineCreateInfo.pDepthStencilState);
	}
	if (graphicsPipelineCreateInfo.pColorBlendState) {
		m_colorBlendState.emplace();
		m_colorBlendState.value().Apply(*graphicsPipelineCreateInfo.pColorBlendState);
	}
	if (graphicsPipelineCreateInfo.pDynamicState) {
		m_dynamicState.emplace();
		m_dynamicState.value().Apply(*graphicsPipelineCreateInfo.pDynamicState);
	}
	m_layout = graphicsPipelineCreateInfo.layout;
	m_renderPass = graphicsPipelineCreateInfo.renderPass;
	m_subpass = graphicsPipelineCreateInfo.subpass;
	m_basePipelineHandle = graphicsPipelineCreateInfo.basePipelineHandle;
	m_basePipelineIndex = graphicsPipelineCreateInfo.basePipelineIndex;
}

GraphicsPipeline::~GraphicsPipeline()
{
	if (m_vkGraphicsPipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(Graphics::GetVkDevice(), m_vkGraphicsPipeline, nullptr);
		m_vkGraphicsPipeline = VK_NULL_HANDLE;
	}
}

VkPipeline GraphicsPipeline::GetVkPipeline() const
{
	return m_vkGraphicsPipeline;
}




CommandPool::CommandPool(const VkCommandPoolCreateInfo& commandPoolCreateInfo)
{
	if (vkCreateCommandPool(Graphics::GetVkDevice(), &commandPoolCreateInfo, nullptr, &m_vkCommandPool) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create command pool!");
	}
}

CommandPool::~CommandPool()
{
	if (m_vkCommandPool != VK_NULL_HANDLE)
	{
		vkDestroyCommandPool(Graphics::GetVkDevice(), m_vkCommandPool, nullptr);
		m_vkCommandPool = VK_NULL_HANDLE;
	}
}

VkCommandPool CommandPool::GetVkCommandPool() const
{
	return m_vkCommandPool;
}

Image::Image(const VkImageCreateInfo& imageCreateInfo)
{
	VmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
	if (vmaCreateImage(Graphics::GetVmaAllocator(), &imageCreateInfo, &allocInfo, &m_vkImage, &m_vmaAllocation, &m_vmaAllocationInfo) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image!");
	}
	m_vkImageLayout = imageCreateInfo.initialLayout;
	m_extent = imageCreateInfo.extent;

}



Image::Image(const VkImageCreateInfo& imageCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo)
{
	if (vmaCreateImage(Graphics::GetVmaAllocator(), &imageCreateInfo, &vmaAllocationCreateInfo, &m_vkImage, &m_vmaAllocation, &m_vmaAllocationInfo) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image!");
	}
	m_vkImageLayout = imageCreateInfo.initialLayout;
	m_extent = imageCreateInfo.extent;
}

void Image::Copy(const VkBuffer& srcBuffer, VkDeviceSize srcOffset) const
{
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer)
		{
			VkBufferImageCopy region{};
			region.bufferOffset = srcOffset;
			region.bufferRowLength = 0;
			region.bufferImageHeight = 0;
			region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			region.imageSubresource.mipLevel = 0;
			region.imageSubresource.baseArrayLayer = 0;
			region.imageSubresource.layerCount = 1;
			region.imageOffset = { 0, 0, 0 };
			region.imageExtent = m_extent;
			vkCmdCopyBufferToImage(commandBuffer, srcBuffer, m_vkImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
		});
}

VkImage Image::GetVkImage() const
{
	return m_vkImage;
}

VmaAllocation Image::GetVmaAllocation() const
{
	return m_vmaAllocation;
}

VkExtent3D Image::GetVkExtent3D() const
{
	return m_extent;
}


Image::~Image()
{
	if (m_vkImage != VK_NULL_HANDLE || m_vmaAllocation != VK_NULL_HANDLE) {
		vmaDestroyImage(Graphics::GetVmaAllocator(), m_vkImage, m_vmaAllocation);
		m_vkImage = VK_NULL_HANDLE;
		m_vmaAllocation = VK_NULL_HANDLE;
		m_vmaAllocationInfo = {};
	}
}

void Image::TransitionImageLayout(VkImageLayout newLayout)
{
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer)
		{
			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = m_vkImageLayout;
			barrier.newLayout = newLayout;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image = m_vkImage;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;

			VkPipelineStageFlags sourceStage;
			VkPipelineStageFlags destinationStage;

			if (m_vkImageLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

				sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			}
			else if (m_vkImageLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

				sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
				destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			}
			else {
				throw std::invalid_argument("unsupported layout transition!");
			}

			vkCmdPipelineBarrier(
				commandBuffer,
				sourceStage, destinationStage,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier
			);
			m_vkImageLayout = newLayout;
		});
}

const VmaAllocationInfo& Image::GetVmaAllocationInfo() const
{
	return m_vmaAllocationInfo;
}

Sampler::Sampler(const VkSamplerCreateInfo& samplerCreateInfo)
{
	if (vkCreateSampler(Graphics::GetVkDevice(), &samplerCreateInfo, nullptr, &m_vkSampler) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create sampler!");
	}
}

Sampler::~Sampler()
{
	if (m_vkSampler != VK_NULL_HANDLE) {
		vkDestroySampler(Graphics::GetVkDevice(), m_vkSampler, nullptr);
		m_vkSampler = VK_NULL_HANDLE;
	}
}

VkSampler Sampler::GetVkSampler() const
{
	return m_vkSampler;
}

Buffer::Buffer(const VkBufferCreateInfo& bufferCreateInfo)
{
	VmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
	if (vmaCreateBuffer(Graphics::GetVmaAllocator(), &bufferCreateInfo, &allocInfo, &m_vkBuffer, &m_vmaAllocation, &m_vmaAllocationInfo) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create buffer!");
	}
}

Buffer::Buffer(const VkBufferCreateInfo& bufferCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo)
{
	if (vmaCreateBuffer(Graphics::GetVmaAllocator(), &bufferCreateInfo, &vmaAllocationCreateInfo, &m_vkBuffer, &m_vmaAllocation, &m_vmaAllocationInfo) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create buffer!");
	}
}

Buffer::~Buffer()
{
	if (m_vkBuffer != VK_NULL_HANDLE || m_vmaAllocation != VK_NULL_HANDLE)
	{
		vmaDestroyBuffer(Graphics::GetVmaAllocator(), m_vkBuffer, m_vmaAllocation);
		m_vkBuffer = VK_NULL_HANDLE;
		m_vmaAllocation = VK_NULL_HANDLE;
		m_vmaAllocationInfo = {};
	}
}

void Buffer::Copy(const Buffer& srcBuffer, const VkDeviceSize size, const VkDeviceSize srcOffset, const VkDeviceSize dstOffset) const
{
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer)
		{
			VkBufferCopy copyRegion{};
			copyRegion.size = size;
			copyRegion.srcOffset = srcOffset;
			copyRegion.dstOffset = dstOffset;
			vkCmdCopyBuffer(commandBuffer, srcBuffer.GetVkBuffer(), m_vkBuffer, 1, &copyRegion);
		});
}

VkBuffer Buffer::GetVkBuffer() const
{
	return m_vkBuffer;
}

VmaAllocation Buffer::GetVmaAllocation() const
{
	return m_vmaAllocation;
}

const VmaAllocationInfo& Buffer::GetVmaAllocationInfo() const
{
	return m_vmaAllocationInfo;
}

DescriptorSetLayout::DescriptorSetLayout(const VkDescriptorSetLayoutCreateInfo& descriptorSetLayoutCreateInfo)
{
	if (vkCreateDescriptorSetLayout(Graphics::GetVkDevice(), &descriptorSetLayoutCreateInfo, nullptr, &m_vkDescriptorSetLayout) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create descriptor set layout!");
	}
}

DescriptorSetLayout::~DescriptorSetLayout()
{
	if (m_vkDescriptorSetLayout != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorSetLayout(Graphics::GetVkDevice(), m_vkDescriptorSetLayout, nullptr);
		m_vkDescriptorSetLayout = VK_NULL_HANDLE;
	}
}

VkDescriptorSetLayout DescriptorSetLayout::GetVkDescriptorSetLayout() const
{
	return m_vkDescriptorSetLayout;
}

DescriptorPool::DescriptorPool(const VkDescriptorPoolCreateInfo& descriptorPoolCreateInfo)
{
	if (vkCreateDescriptorPool(Graphics::GetVkDevice(), &descriptorPoolCreateInfo, nullptr, &m_vkDescriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create descriptor pool!");
	}
}

DescriptorPool::~DescriptorPool()
{
	if (m_vkDescriptorPool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPool(Graphics::GetVkDevice(), m_vkDescriptorPool, nullptr);
		m_vkDescriptorPool = VK_NULL_HANDLE;
	}
}

VkDescriptorPool DescriptorPool::GetVkDescriptorPool() const
{
	return m_vkDescriptorPool;
}

ShaderEXT::ShaderEXT(const VkShaderCreateInfoEXT& shaderCreateInfoExt)
{
	if (vkCreateShadersEXT(Graphics::GetVkDevice(), 1, &shaderCreateInfoExt, nullptr, &m_shaderExt) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create shaderEXT!");
	}
	m_flags = shaderCreateInfoExt.flags;
	m_stage = shaderCreateInfoExt.stage;
	m_nextStage = shaderCreateInfoExt.nextStage;
	m_codeType = shaderCreateInfoExt.codeType;
	m_name = shaderCreateInfoExt.pName;
	ApplyVector(m_setLayouts, shaderCreateInfoExt.setLayoutCount, shaderCreateInfoExt.pSetLayouts);
	ApplyVector(m_pushConstantRanges, shaderCreateInfoExt.pushConstantRangeCount, shaderCreateInfoExt.pPushConstantRanges);
	if (shaderCreateInfoExt.pSpecializationInfo) m_specializationInfo = *shaderCreateInfoExt.pSpecializationInfo;
}

ShaderEXT::~ShaderEXT()
{
	if (m_shaderExt != VK_NULL_HANDLE)
	{
		vkDestroyShaderEXT(Graphics::GetVkDevice(), m_shaderExt, nullptr);
		m_shaderExt = VK_NULL_HANDLE;
	}
}

const VkShaderEXT& ShaderEXT::GetVkShaderEXT() const
{
	return m_shaderExt;
}
