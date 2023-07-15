#include "GraphicsPipeline.hpp"

#include "GraphicsLayer.hpp"
#include "Utilities.hpp"

using namespace EvoEngine;

void Fence::Create(const VkFenceCreateInfo& vkFenceCreateInfo)
{
	Destroy();
	if (vkCreateFence(GraphicsLayer::GetVkDevice(), &vkFenceCreateInfo, nullptr, &m_vkFence) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create vkFence!");
	}
}

void Fence::Destroy()
{
	if(m_vkFence != VK_NULL_HANDLE)
	{
		vkDestroyFence(GraphicsLayer::GetVkDevice(), m_vkFence, nullptr);
		m_vkFence = nullptr;
	}
}

VkFence Fence::GetVkFence() const
{
	return m_vkFence;
}

void Semaphore::Create(const VkSemaphoreCreateInfo& semaphoreCreateInfo)
{
	Destroy();
	if(vkCreateSemaphore(GraphicsLayer::GetVkDevice(), &semaphoreCreateInfo, nullptr, &m_vkSemaphore) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create vkSemaphore!");
	}
}

void Semaphore::Destroy()
{
	if(m_vkSemaphore != VK_NULL_HANDLE)
	{
		vkDestroySemaphore(GraphicsLayer::GetVkDevice(), m_vkSemaphore, nullptr);
		m_vkSemaphore = VK_NULL_HANDLE;
	}
}



VkSemaphore Semaphore::GetVkSemaphore() const
{
	return m_vkSemaphore;
}

void Swapchain::Create(const VkSwapchainCreateInfoKHR& swapChainCreateInfo)
{
	Destroy();
	const auto& device = GraphicsLayer::GetVkDevice();
	if (vkCreateSwapchainKHR(GraphicsLayer::GetVkDevice(), &swapChainCreateInfo, nullptr, &m_vkSwapchain) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create swap chain!");
	}
	uint32_t imageCount = 0;
	vkGetSwapchainImagesKHR(device, m_vkSwapchain, &imageCount, nullptr);
	m_vkImages.resize(imageCount);
	vkGetSwapchainImagesKHR(device, m_vkSwapchain, &imageCount, m_vkImages.data());

	m_vkFormat = swapChainCreateInfo.imageFormat;
	m_vkExtent2D = swapChainCreateInfo.imageExtent;

	m_imageViews.resize(m_vkImages.size());
	for (size_t i = 0; i < m_vkImages.size(); i++) {
		VkImageViewCreateInfo imageViewCreateInfo{};
		imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCreateInfo.image = m_vkImages[i];
		imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCreateInfo.format = m_vkFormat;
		imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
		imageViewCreateInfo.subresourceRange.levelCount = 1;
		imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
		imageViewCreateInfo.subresourceRange.layerCount = 1;
		m_imageViews[i].Create(imageViewCreateInfo);
	}
}

void Swapchain::Destroy()
{
	for (auto& imageView : m_imageViews) {
		imageView.Destroy();
	}
	m_imageViews.clear();
	if (m_vkSwapchain != VK_NULL_HANDLE) {
		vkDestroySwapchainKHR(GraphicsLayer::GetVkDevice(), m_vkSwapchain, nullptr);
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

const std::vector<ImageView>& Swapchain::GetImageViews() const
{
	return m_imageViews;
}

VkFormat Swapchain::GetVkFormat() const
{
	return m_vkFormat;
}

VkExtent2D Swapchain::GetVkExtent2D() const
{
	return m_vkExtent2D;
}

void Image::Create(const VkImageCreateInfo& imageCreateInfo)
{
	Destroy();
	if (vkCreateImage(GraphicsLayer::GetVkDevice(), &imageCreateInfo, nullptr, &m_vkImage) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image!");
	}
}

void Image::Destroy()
{
	if (m_vkImage != VK_NULL_HANDLE) {
		vkDestroyImage(GraphicsLayer::GetVkDevice(), m_vkImage, nullptr);
		m_vkImage = VK_NULL_HANDLE;
	}
}

VkImage Image::GetVkImage() const
{
	return m_vkImage;
}

void ImageView::Create(const VkImageViewCreateInfo& imageViewCreateInfo)
{
	Destroy();
	if (vkCreateImageView(GraphicsLayer::GetVkDevice(), &imageViewCreateInfo, nullptr, &m_vkImageView) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image views!");
	}
}

void ImageView::Destroy()
{
	if (m_vkImageView != VK_NULL_HANDLE) {
		vkDestroyImageView(GraphicsLayer::GetVkDevice(), m_vkImageView, nullptr);
		m_vkImageView = VK_NULL_HANDLE;
	}
}

VkImageView ImageView::GetVkImageView() const
{
	return m_vkImageView;
}

void Framebuffer::Create(const VkFramebufferCreateInfo& framebufferCreateInfo)
{
	Destroy();
	if (vkCreateFramebuffer(GraphicsLayer::GetVkDevice(), &framebufferCreateInfo, nullptr, &m_vkFramebuffer) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create framebuffer!");
	}
}

void Framebuffer::Destroy()
{
	if (m_vkFramebuffer != VK_NULL_HANDLE) {
		vkDestroyFramebuffer(GraphicsLayer::GetVkDevice(), m_vkFramebuffer, nullptr);
		m_vkFramebuffer = VK_NULL_HANDLE;
	}
}

VkFramebuffer Framebuffer::GetVkFrameBuffer() const
{
	return m_vkFramebuffer;
}

void ShaderModule::Create(shaderc_shader_kind shaderKind, const std::vector<char>& code)
{
	Destroy();
	m_shaderKind = shaderKind;
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
	if (vkCreateShaderModule(GraphicsLayer::GetVkDevice(), &createInfo, nullptr, &m_vkShaderModule) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create shader module!");
	}
}

void ShaderModule::Destroy()
{
	if (m_vkShaderModule != VK_NULL_HANDLE) {
		vkDestroyShaderModule(GraphicsLayer::GetVkDevice(), m_vkShaderModule, nullptr);
		m_vkShaderModule = VK_NULL_HANDLE;
	}
}

void ShaderModule::Create(shaderc_shader_kind shaderKind, const std::string& code)
{
	Destroy();
	m_shaderKind = shaderKind;
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	const auto binary = ShaderUtils::CompileFile("Shader", m_shaderKind, code);
	createInfo.pCode = binary.data();
	createInfo.codeSize = binary.size() * sizeof(uint32_t);
	if (vkCreateShaderModule(GraphicsLayer::GetVkDevice(), &createInfo, nullptr, &m_vkShaderModule) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create shader module!");
	}
}

VkShaderModule ShaderModule::GetVkShaderModule() const
{
	return m_vkShaderModule;
}

void RenderPass::Create(const VkRenderPassCreateInfo& renderPassCreateInfo)
{
	Destroy();
	if (vkCreateRenderPass(GraphicsLayer::GetVkDevice(), &renderPassCreateInfo, nullptr, &m_vkRenderPass) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create render pass!");
	}
}

void RenderPass::Destroy()
{
	if (m_vkRenderPass != VK_NULL_HANDLE) {
		vkDestroyRenderPass(GraphicsLayer::GetVkDevice(), m_vkRenderPass, nullptr);
		m_vkRenderPass = VK_NULL_HANDLE;
	}
}

VkRenderPass RenderPass::GetVkRenderPass() const
{
	return m_vkRenderPass;
}

void PipelineLayout::Create(const VkPipelineLayoutCreateInfo& pipelineLayoutCreateInfo)
{
	Destroy();
	if (vkCreatePipelineLayout(GraphicsLayer::GetVkDevice(), &pipelineLayoutCreateInfo, nullptr, &m_vkPipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create pipeline layout!");
	}
}

void PipelineLayout::Destroy()
{
	if (m_vkPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(GraphicsLayer::GetVkDevice(), m_vkPipelineLayout, nullptr);
		m_vkPipelineLayout = VK_NULL_HANDLE;
	}
}

VkPipelineLayout PipelineLayout::GetVkPipelineLayout() const
{
	return m_vkPipelineLayout;
}


void GraphicsPipeline::Create(const VkGraphicsPipelineCreateInfo& graphicsPipelineCreateInfo)
{
	Destroy();
	if (vkCreateGraphicsPipelines(GraphicsLayer::GetVkDevice(), VK_NULL_HANDLE, 1,
		&graphicsPipelineCreateInfo, nullptr, &m_vkGraphicsPipeline) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create graphics pipeline!");
	}
}

void GraphicsPipeline::Destroy()
{
	if (m_vkGraphicsPipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(GraphicsLayer::GetVkDevice(), m_vkGraphicsPipeline, nullptr);
		m_vkGraphicsPipeline = VK_NULL_HANDLE;
	}
}

VkPipeline GraphicsPipeline::GetVkPipeline() const
{
	return m_vkGraphicsPipeline;
}




void CommandPool::Create(const VkCommandPoolCreateInfo& commandPoolCreateInfo)
{
	Destroy();
	if (vkCreateCommandPool(GraphicsLayer::GetVkDevice(), &commandPoolCreateInfo, nullptr, &m_vkCommandPool) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create command pool!");
	}
}

void CommandPool::Destroy()
{
	if (m_vkCommandPool != VK_NULL_HANDLE)
	{
		vkDestroyCommandPool(GraphicsLayer::GetVkDevice(), m_vkCommandPool, nullptr);
		m_vkCommandPool = VK_NULL_HANDLE;
	}
}

VkCommandPool CommandPool::GetVkCommandPool() const
{
	return m_vkCommandPool;
}
