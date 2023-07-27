#include "GraphicsResources.hpp"

#include "Console.hpp"
#include "Graphics.hpp"
#include "Utilities.hpp"

using namespace EvoEngine;

Fence::Fence(const VkFenceCreateInfo& vkFenceCreateInfo)
{
	Graphics::CheckVk(vkCreateFence(Graphics::GetVkDevice(), &vkFenceCreateInfo, nullptr, &m_vkFence));
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

const VkFence& Fence::GetVkFence() const
{
	return m_vkFence;
}

Semaphore::Semaphore(const VkSemaphoreCreateInfo& semaphoreCreateInfo)
{
	Graphics::CheckVk(vkCreateSemaphore(Graphics::GetVkDevice(), &semaphoreCreateInfo, nullptr, &m_vkSemaphore));
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



const VkSemaphore& Semaphore::GetVkSemaphore() const
{
	return m_vkSemaphore;
}

Swapchain::Swapchain(const VkSwapchainCreateInfoKHR& swapChainCreateInfo)
{
	const auto& device = Graphics::GetVkDevice();
	Graphics::CheckVk(vkCreateSwapchainKHR(Graphics::GetVkDevice(), &swapChainCreateInfo, nullptr, &m_vkSwapchain));
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

	m_vkImageViews.clear();
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
		auto imageView = std::make_shared<ImageView>(imageViewCreateInfo);
		m_vkImageViews.emplace_back(imageView);
	}
}

Swapchain::~Swapchain()
{
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

const std::vector<VkImage>& Swapchain::GetAllVkImages() const
{
	return m_vkImages;
}

const VkImage& Swapchain::GetVkImage() const
{
	return m_vkImages[Graphics::GetNextImageIndex()];
}

const VkImageView& Swapchain::GetVkImageView() const
{
	return m_vkImageViews[Graphics::GetNextImageIndex()]->GetVkImageView();
}

const std::vector<std::shared_ptr<ImageView>>& Swapchain::GetAllImageViews() const
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
	Graphics::CheckVk(vkCreateImageView(Graphics::GetVkDevice(), &imageViewCreateInfo, nullptr, &m_vkImageView));
	m_image = nullptr;
	m_flags = imageViewCreateInfo.flags;
	m_viewType = imageViewCreateInfo.viewType;
	m_format = imageViewCreateInfo.format;
	m_components = imageViewCreateInfo.components;
	m_subresourceRange = imageViewCreateInfo.subresourceRange;
}

ImageView::ImageView(const VkImageViewCreateInfo& imageViewCreateInfo, const std::shared_ptr<Image>& image)
{
	Graphics::CheckVk(vkCreateImageView(Graphics::GetVkDevice(), &imageViewCreateInfo, nullptr, &m_vkImageView));
	m_image = image;
	m_flags = imageViewCreateInfo.flags;
	m_viewType = imageViewCreateInfo.viewType;
	m_format = image->GetFormat();
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

const std::shared_ptr<Image>& ImageView::GetImage() const
{
	return m_image;
}


ShaderModule::~ShaderModule()
{
	if (m_vkShaderModule != VK_NULL_HANDLE) {
		vkDestroyShaderModule(Graphics::GetVkDevice(), m_vkShaderModule, nullptr);
		m_vkShaderModule = VK_NULL_HANDLE;
	}
}

ShaderModule::ShaderModule(const VkShaderModuleCreateInfo& createInfo)
{
	Graphics::CheckVk(vkCreateShaderModule(Graphics::GetVkDevice(), &createInfo, nullptr, &m_vkShaderModule));
}

VkShaderModule ShaderModule::GetVkShaderModule() const
{
	return m_vkShaderModule;
}



PipelineLayout::PipelineLayout(const VkPipelineLayoutCreateInfo& pipelineLayoutCreateInfo)
{
	Graphics::CheckVk(vkCreatePipelineLayout(Graphics::GetVkDevice(), &pipelineLayoutCreateInfo, nullptr, &m_vkPipelineLayout));

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



CommandPool::CommandPool(const VkCommandPoolCreateInfo& commandPoolCreateInfo)
{
	Graphics::CheckVk(vkCreateCommandPool(Graphics::GetVkDevice(), &commandPoolCreateInfo, nullptr, &m_vkCommandPool));
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
	if (vmaCreateImage(Graphics::GetVmaAllocator(), &imageCreateInfo, &allocInfo, &m_vkImage, &m_vmaAllocation, &m_vmaAllocationInfo)) {
		throw std::runtime_error("Failed to create image!");
	}
	m_flags = imageCreateInfo.flags;
	m_imageType = imageCreateInfo.imageType;
	m_format = imageCreateInfo.format;
	m_extent = imageCreateInfo.extent;
	m_mipLevels = imageCreateInfo.mipLevels;
	m_arrayLayers = imageCreateInfo.arrayLayers;
	m_samples = imageCreateInfo.samples;
	m_tiling = imageCreateInfo.tiling;
	m_usage = imageCreateInfo.usage;
	m_sharingMode = imageCreateInfo.sharingMode;
	ApplyVector(m_queueFamilyIndices, imageCreateInfo.queueFamilyIndexCount, imageCreateInfo.pQueueFamilyIndices);

	m_layout = m_initialLayout = imageCreateInfo.initialLayout;
}



Image::Image(const VkImageCreateInfo& imageCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo)
{
	if (vmaCreateImage(Graphics::GetVmaAllocator(), &imageCreateInfo, &vmaAllocationCreateInfo, &m_vkImage, &m_vmaAllocation, &m_vmaAllocationInfo)) {
		throw std::runtime_error("Failed to create image!");
	}
	m_flags = imageCreateInfo.flags;
	m_imageType = imageCreateInfo.imageType;
	m_format = imageCreateInfo.format;
	m_extent = imageCreateInfo.extent;
	m_mipLevels = imageCreateInfo.mipLevels;
	m_arrayLayers = imageCreateInfo.arrayLayers;
	m_samples = imageCreateInfo.samples;
	m_tiling = imageCreateInfo.tiling;
	m_usage = imageCreateInfo.usage;
	m_sharingMode = imageCreateInfo.sharingMode;
	ApplyVector(m_queueFamilyIndices, imageCreateInfo.queueFamilyIndexCount, imageCreateInfo.pQueueFamilyIndices);

	m_layout = m_initialLayout = imageCreateInfo.initialLayout;
}

bool Image::HasStencilComponent() const
{
	return m_format == VK_FORMAT_D32_SFLOAT_S8_UINT
		|| m_format == VK_FORMAT_D24_UNORM_S8_UINT;
}

void Image::Copy(VkCommandBuffer commandBuffer, const VkBuffer& srcBuffer, VkDeviceSize srcOffset) const
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
}

VkImage Image::GetVkImage() const
{
	return m_vkImage;
}

VkFormat Image::GetFormat() const
{
	return m_format;
}

VmaAllocation Image::GetVmaAllocation() const
{
	return m_vmaAllocation;
}

VkExtent3D Image::GetExtent() const
{
	return m_extent;
}

VkImageLayout Image::GetLayout() const
{
	return m_layout;
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


void Image::TransitionImageLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout)
{
	Graphics::TransitImageLayout(commandBuffer, m_vkImage, m_format, m_layout, newLayout);
	m_layout = newLayout;
}

const VmaAllocationInfo& Image::GetVmaAllocationInfo() const
{
	return m_vmaAllocationInfo;
}

Sampler::Sampler(const VkSamplerCreateInfo& samplerCreateInfo)
{
	Graphics::CheckVk(vkCreateSampler(Graphics::GetVkDevice(), &samplerCreateInfo, nullptr, &m_vkSampler));
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
	if (vmaCreateBuffer(Graphics::GetVmaAllocator(), &bufferCreateInfo, &allocInfo, &m_vkBuffer, &m_vmaAllocation, &m_vmaAllocationInfo)) {
		throw std::runtime_error("Failed to create buffer!");
	}
}

Buffer::Buffer(const VkBufferCreateInfo& bufferCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo)
{
	if (vmaCreateBuffer(Graphics::GetVmaAllocator(), &bufferCreateInfo, &vmaAllocationCreateInfo, &m_vkBuffer, &m_vmaAllocation, &m_vmaAllocationInfo)) {
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

const VkBuffer& Buffer::GetVkBuffer() const
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
	Graphics::CheckVk(vkCreateDescriptorSetLayout(Graphics::GetVkDevice(), &descriptorSetLayoutCreateInfo, nullptr, &m_vkDescriptorSetLayout));
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
	Graphics::CheckVk(vkCreateDescriptorPool(Graphics::GetVkDevice(), &descriptorPoolCreateInfo, nullptr, &m_vkDescriptorPool));
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
	Graphics::CheckVk(vkCreateShadersEXT(Graphics::GetVkDevice(), 1, &shaderCreateInfoExt, nullptr, &m_shaderExt));
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

CommandBufferStatus CommandBuffer::GetStatus() const
{
	return m_status;
}

void CommandBuffer::Allocate(const VkQueueFlagBits& queueType, const VkCommandBufferLevel& bufferLevel)
{
	VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
	commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	commandBufferAllocateInfo.commandPool = Graphics::GetVkCommandPool();
	commandBufferAllocateInfo.level = bufferLevel;
	commandBufferAllocateInfo.commandBufferCount = 1;
	Graphics::CheckVk(vkAllocateCommandBuffers(Graphics::GetVkDevice(), &commandBufferAllocateInfo, &m_vkCommandBuffer));
	m_status = CommandBufferStatus::Ready;
}

void CommandBuffer::Free()
{
	if (m_vkCommandBuffer != VK_NULL_HANDLE)
	{
		vkFreeCommandBuffers(Graphics::GetVkDevice(), Graphics::GetVkCommandPool(), 1, &m_vkCommandBuffer);
		m_vkCommandBuffer = VK_NULL_HANDLE;
	}
	m_status = CommandBufferStatus::Invalid;
}

const VkCommandBuffer& CommandBuffer::GetVkCommandBuffer() const
{
	return m_vkCommandBuffer;
}

void CommandBuffer::Begin(const VkCommandBufferUsageFlags& usage)
{
	if (m_status == CommandBufferStatus::Invalid)
	{
		EVOENGINE_ERROR("Command buffer invalid!");
		return;
	}
	if (m_status != CommandBufferStatus::Ready)
	{
		EVOENGINE_ERROR("Command buffer not ready!");
		return;
	}
	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = usage;
	Graphics::CheckVk(vkBeginCommandBuffer(m_vkCommandBuffer, &beginInfo));
	m_status = CommandBufferStatus::Recording;
}

void CommandBuffer::End()
{
	if (m_status == CommandBufferStatus::Invalid)
	{
		EVOENGINE_ERROR("Command buffer invalid!");
		return;
	}
	if (m_status != CommandBufferStatus::Recording)
	{
		EVOENGINE_ERROR("Command buffer not recording!");
		return;
	}
	Graphics::CheckVk(vkEndCommandBuffer(m_vkCommandBuffer));
	m_status = CommandBufferStatus::Recorded;
}

void CommandBuffer::SubmitIdle()
{
	if (m_status == CommandBufferStatus::Invalid)
	{
		EVOENGINE_ERROR("Command buffer invalid!");
		return;
	}
	if (m_status == CommandBufferStatus::Recording)
	{
		EVOENGINE_ERROR("Command buffer recording!");
		return;
	}
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &m_vkCommandBuffer;

	VkFenceCreateInfo fenceCreateInfo = {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

	VkFence fence;
	auto device = Graphics::GetVkDevice();
	if (vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));

	Graphics::CheckVk(vkResetFences(device, 1, &fence));

	Graphics::CheckVk(vkQueueSubmit(Graphics::GetGraphicsVkQueue(), 1, &submitInfo, fence));

	Graphics::CheckVk(vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max()));

	vkDestroyFence(device, fence, nullptr);
}

void CommandBuffer::Submit(const VkSemaphore& waitSemaphore, const VkSemaphore& signalSemaphore, VkFence fence)
{
	if (m_status == CommandBufferStatus::Invalid)
	{
		EVOENGINE_ERROR("Command buffer invalid!");
		return;
	}
	if (m_status == CommandBufferStatus::Recording)
	{
		EVOENGINE_ERROR("Command buffer recording!");
		return;
	}
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &m_vkCommandBuffer;

	if (waitSemaphore != VK_NULL_HANDLE)
	{
		// Pipeline stages used to wait at for graphics queue submissions.
		static VkPipelineStageFlags submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		submitInfo.pWaitDstStageMask = &submitPipelineStages;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &waitSemaphore;
	}

	if (signalSemaphore != VK_NULL_HANDLE)
	{
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &signalSemaphore;
	}

	if (fence != VK_NULL_HANDLE)
	{
		//Graphics::CheckVk(vkResetFences(Graphics::GetVkDevice(), 1, &fence));
		//Renderer::CheckVk(vkWaitForFences(*logicalDevice, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max()));
	}

	Graphics::CheckVk(vkQueueSubmit(Graphics::GetGraphicsVkQueue(), 1, &submitInfo, fence));
}

void CommandBuffer::Reset()
{
	if (m_status == CommandBufferStatus::Invalid)
	{
		EVOENGINE_ERROR("Command buffer invalid!");
		return;
	}
	Graphics::CheckVk(vkResetCommandBuffer(m_vkCommandBuffer, 0));
	m_status = CommandBufferStatus::Ready;
}
