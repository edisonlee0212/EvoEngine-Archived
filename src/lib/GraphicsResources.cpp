#include "GraphicsResources.hpp"

#include "Graphics.hpp"
#include "Utilities.hpp"

using namespace EvoEngine;

void IGraphicsResource::Destroy()
{
}

IGraphicsResource::~IGraphicsResource()
{
	Destroy();
}

void Fence::Create(const VkFenceCreateInfo& vkFenceCreateInfo)
{
	Destroy();
	if (vkCreateFence(Graphics::GetVkDevice(), &vkFenceCreateInfo, nullptr, &m_vkFence) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create vkFence!");
	}
}

void Fence::Destroy()
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

void Semaphore::Create(const VkSemaphoreCreateInfo& semaphoreCreateInfo)
{
	Destroy();
	if (vkCreateSemaphore(Graphics::GetVkDevice(), &semaphoreCreateInfo, nullptr, &m_vkSemaphore) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create vkSemaphore!");
	}
}

void Semaphore::Destroy()
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

void Swapchain::Create(const VkSwapchainCreateInfoKHR& swapChainCreateInfo)
{
	Destroy();
	const auto& device = Graphics::GetVkDevice();
	if (vkCreateSwapchainKHR(Graphics::GetVkDevice(), &swapChainCreateInfo, nullptr, &m_vkSwapchain) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create swap chain!");
	}
	uint32_t imageCount = 0;
	vkGetSwapchainImagesKHR(device, m_vkSwapchain, &imageCount, nullptr);
	m_vkImages.resize(imageCount);
	vkGetSwapchainImagesKHR(device, m_vkSwapchain, &imageCount, m_vkImages.data());

	m_vkFormat = swapChainCreateInfo.imageFormat;
	m_vkExtent2D = swapChainCreateInfo.imageExtent;

	m_vkImageViews.resize(m_vkImages.size());
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

		if (vkCreateImageView(Graphics::GetVkDevice(), &imageViewCreateInfo, nullptr, &m_vkImageViews[i]) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create image views!");
		}
	}
}

void Swapchain::Destroy()
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

VkFormat Swapchain::GetVkFormat() const
{
	return m_vkFormat;
}

VkExtent2D Swapchain::GetVkExtent2D() const
{
	return m_vkExtent2D;
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

void Image::Create(const VkImageCreateInfo& imageCreateInfo)
{
	Destroy();
	VmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
	if (vmaCreateImage(Graphics::GetVmaAllocator(), &imageCreateInfo, &allocInfo, &m_vkImage, &m_vmaAllocation, &m_vmaAllocationInfo) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image!");
	}
	m_vkImageLayout = imageCreateInfo.initialLayout;
	m_extent = imageCreateInfo.extent;

}



void Image::Create(const VkImageCreateInfo& imageCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo)
{
	Destroy();
	if (vmaCreateImage(Graphics::GetVmaAllocator(), &imageCreateInfo, &vmaAllocationCreateInfo, &m_vkImage, &m_vmaAllocation, &m_vmaAllocationInfo) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create image!");
	}
	m_vkImageLayout = imageCreateInfo.initialLayout;
	m_extent = imageCreateInfo.extent;
}

void Image::Copy(const Buffer& srcBuffer, VkDeviceSize srcOffset) const
{
	Graphics::SingleTimeCommands([&](VkCommandBuffer commandBuffer)
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
			vkCmdCopyBufferToImage(commandBuffer, srcBuffer.GetVkBuffer(), m_vkImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
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


void Image::Destroy()
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
	Graphics::SingleTimeCommands([&](VkCommandBuffer commandBuffer)
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

void Buffer::Create(const VkBufferCreateInfo& bufferCreateInfo)
{
	Destroy();
	VmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
	if (vmaCreateBuffer(Graphics::GetVmaAllocator(), &bufferCreateInfo, &allocInfo, &m_vkBuffer, &m_vmaAllocation, &m_vmaAllocationInfo) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create buffer!");
	}
}

void Buffer::Create(const VkBufferCreateInfo& bufferCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo)
{
	Destroy();
	if (vmaCreateBuffer(Graphics::GetVmaAllocator(), &bufferCreateInfo, &vmaAllocationCreateInfo, &m_vkBuffer, &m_vmaAllocation, &m_vmaAllocationInfo) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create buffer!");
	}
}

void Buffer::Destroy()
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
	Graphics::SingleTimeCommands([&](VkCommandBuffer commandBuffer)
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

void DescriptorSetLayout::Create(const VkDescriptorSetLayoutCreateInfo& descriptorSetLayoutCreateInfo)
{
	Destroy();
	if (vkCreateDescriptorSetLayout(Graphics::GetVkDevice(), &descriptorSetLayoutCreateInfo, nullptr, &m_vkDescriptorSetLayout) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create descriptor set layout!");
	}
}

void DescriptorSetLayout::Destroy()
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

void DescriptorPool::Create(const VkDescriptorPoolCreateInfo& descriptorPoolCreateInfo)
{
	Destroy();
	if (vkCreateDescriptorPool(Graphics::GetVkDevice(), &descriptorPoolCreateInfo, nullptr, &m_vkDescriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create descriptor pool!");
	}
}

void DescriptorPool::Destroy()
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
