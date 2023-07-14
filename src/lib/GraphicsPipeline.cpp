#include "GraphicsPipeline.hpp"

#include "Graphics.hpp"
#include "Utilities.hpp"

using namespace EvoEngine;

void Fence::Create(const VkFenceCreateInfo& vkFenceCreateInfo)
{
	//Destroy();
	if (vkCreateFence(Graphics::GetVkDevice(), &vkFenceCreateInfo, nullptr, &m_vkFence) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create vkFence!");
	}
}

void Fence::Destroy()
{
	if(m_vkFence != VK_NULL_HANDLE)
	{
		vkDestroyFence(Graphics::GetVkDevice(), m_vkFence, nullptr);
		m_vkFence = nullptr;
	}
}

VkFence Fence::GetVkFence() const
{
	return m_vkFence;
}

void Semaphore::Create(const VkSemaphoreCreateInfo& semaphoreCreateInfo)
{
	//Destroy();
	if(vkCreateSemaphore(Graphics::GetVkDevice(), &semaphoreCreateInfo, nullptr, &m_vkSemaphore) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create vkSemaphore!");
	}
}

void Semaphore::Destroy()
{
	if(m_vkSemaphore != VK_NULL_HANDLE)
	{
		vkDestroySemaphore(Graphics::GetVkDevice(), m_vkSemaphore, nullptr);
		m_vkSemaphore = VK_NULL_HANDLE;
	}
}



VkSemaphore Semaphore::GetVkSemaphore() const
{
	return m_vkSemaphore;
}

void Swapchain::Create(const VkSwapchainCreateInfoKHR& swapchainCreateInfo)
{
	Destroy();
	m_vkFormat = swapchainCreateInfo.imageFormat;
	m_vkExtent2D = swapchainCreateInfo.imageExtent;

	const auto& device = Graphics::GetVkDevice();
	if (vkCreateSwapchainKHR(Graphics::GetVkDevice(), &swapchainCreateInfo, nullptr, &m_vkSwapchain) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create swap chain!");
	}
	uint32_t imageCount = 0;
	vkGetSwapchainImagesKHR(device, m_vkSwapchain, &imageCount, nullptr);
	m_vkImages.resize(imageCount);
	vkGetSwapchainImagesKHR(device, m_vkSwapchain, &imageCount, m_vkImages.data());
}

void Swapchain::Destroy()
{
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
	if (vkCreateImage(Graphics::GetVkDevice(), &imageCreateInfo, nullptr, &m_vkImage) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image!");
	}
}

void Image::Destroy()
{
	if (m_vkImage != VK_NULL_HANDLE) {
		vkDestroyImage(Graphics::GetVkDevice(), m_vkImage, nullptr);
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
	if (vkCreateImageView(Graphics::GetVkDevice(), &imageViewCreateInfo, nullptr, &m_vkImageView) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image views!");
	}
}

void ImageView::Destroy()
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

void Framebuffer::Create(const VkFramebufferCreateInfo& framebufferCreateInfo)
{
	Destroy();
	if (vkCreateFramebuffer(Graphics::GetVkDevice(), &framebufferCreateInfo, nullptr, &m_vkFramebuffer) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create framebuffer!");
	}
}

void Framebuffer::Destroy()
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

void ShaderModule::Create(shaderc_shader_kind shaderKind, const std::vector<char>& code)
{
	Destroy();
	m_shaderKind = shaderKind;
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
	if (vkCreateShaderModule(Graphics::GetVkDevice(), &createInfo, nullptr, &m_vkShaderModule) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create shader module!");
	}
}

void ShaderModule::Destroy()
{
	if (m_vkShaderModule != VK_NULL_HANDLE) {
		vkDestroyShaderModule(Graphics::GetVkDevice(), m_vkShaderModule, nullptr);
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
	if (vkCreateShaderModule(Graphics::GetVkDevice(), &createInfo, nullptr, &m_vkShaderModule) != VK_SUCCESS) {
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
	if (vkCreateRenderPass(Graphics::GetVkDevice(), &renderPassCreateInfo, nullptr, &m_vkRenderPass) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create render pass!");
	}
}

void RenderPass::Destroy()
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

void PipelineLayout::Create(const VkPipelineLayoutCreateInfo& pipelineLayoutCreateInfo)
{
	Destroy();
	if (vkCreatePipelineLayout(Graphics::GetVkDevice(), &pipelineLayoutCreateInfo, nullptr, &m_vkPipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create pipeline layout!");
	}
}

void PipelineLayout::Destroy()
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


void GraphicsPipeline::Create(const VkGraphicsPipelineCreateInfo& graphicsPipelineCreateInfo)
{
	Destroy();
	if (vkCreateGraphicsPipelines(Graphics::GetVkDevice(), VK_NULL_HANDLE, 1,
		&graphicsPipelineCreateInfo, nullptr, &m_vkGraphicsPipeline) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create graphics pipeline!");
	}
}

void GraphicsPipeline::Destroy()
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




void CommandPool::Create(const VkCommandPoolCreateInfo& commandPoolCreateInfo)
{
	Destroy();
	if (vkCreateCommandPool(Graphics::GetVkDevice(), &commandPoolCreateInfo, nullptr, &m_vkCommandPool) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create command pool!");
	}
}

void CommandPool::Destroy()
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
