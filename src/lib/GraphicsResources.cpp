#include "GraphicsResources.hpp"

#include "Application.hpp"
#include "Console.hpp"
#include "Graphics.hpp"
#include "Utilities.hpp"

using namespace evo_engine;

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
#ifdef _WIN64
void* Semaphore::GetVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType) const
{
	void* handle;

	VkSemaphoreGetWin32HandleInfoKHR vulkanSemaphoreGetWin32HandleInfoKHR = {};
	vulkanSemaphoreGetWin32HandleInfoKHR.sType =
		VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
	vulkanSemaphoreGetWin32HandleInfoKHR.pNext = NULL;
	vulkanSemaphoreGetWin32HandleInfoKHR.semaphore = m_vkSemaphore;
	vulkanSemaphoreGetWin32HandleInfoKHR.handleType =
		externalSemaphoreHandleType;
	auto func = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(
		Graphics::GetVkDevice(), "vkGetSemaphoreWin32HandleKHR");
	func(Graphics::GetVkDevice(), &vulkanSemaphoreGetWin32HandleInfoKHR,
		&handle);

	return handle;
}
#else
int Semaphore::GetVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType) const
{
	if (externalSemaphoreHandleType ==
		VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
		int fd;

		VkSemaphoreGetFdInfoKHR vulkanSemaphoreGetFdInfoKHR = {};
		vulkanSemaphoreGetFdInfoKHR.sType =
			VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
		vulkanSemaphoreGetFdInfoKHR.pNext = NULL;
		vulkanSemaphoreGetFdInfoKHR.semaphore = m_vkSemaphore;
		vulkanSemaphoreGetFdInfoKHR.handleType =
			VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

		vkGetSemaphoreFdKHR(Graphics::GetVkDevice(), &vulkanSemaphoreGetFdInfoKHR, &fd);

		return fd;
	}
	return -1;
}
#endif
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
	return m_vkImageViews[Graphics::GetNextImageIndex()]->m_vkImageView;
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

uint32_t Image::GetMipLevels() const
{
	return m_mipLevels;
}

Image::Image(VkImageCreateInfo imageCreateInfo)
{
	VkExternalMemoryImageCreateInfo vkExternalMemImageCreateInfo = {};
	vkExternalMemImageCreateInfo.sType =
		VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
	vkExternalMemImageCreateInfo.pNext = NULL;
#ifdef _WIN64
	vkExternalMemImageCreateInfo.handleTypes =
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
	vkExternalMemImageCreateInfo.handleTypes =
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

	imageCreateInfo.pNext = &vkExternalMemImageCreateInfo;

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



Image::Image(VkImageCreateInfo imageCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo)
{
	VkExternalMemoryImageCreateInfo vkExternalMemImageCreateInfo = {};
	vkExternalMemImageCreateInfo.sType =
		VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
	vkExternalMemImageCreateInfo.pNext = NULL;
#ifdef _WIN64
	vkExternalMemImageCreateInfo.handleTypes =
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
	vkExternalMemImageCreateInfo.handleTypes =
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

	imageCreateInfo.pNext = &vkExternalMemImageCreateInfo;

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

void Image::CopyFromBuffer(VkCommandBuffer commandBuffer, const VkBuffer& srcBuffer, VkDeviceSize srcOffset) const
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

void Image::GenerateMipmaps(VkCommandBuffer commandBuffer)
{

	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.image = m_vkImage;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = m_arrayLayers;
	barrier.subresourceRange.levelCount = 1;

	int32_t mipWidth = m_extent.width;
	int32_t mipHeight = m_extent.height;

	for (uint32_t i = 1; i < m_mipLevels; i++) {
		barrier.subresourceRange.baseMipLevel = i - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		VkImageBlit blit{};
		blit.srcOffsets[0] = { 0, 0, 0 };
		blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
		blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.srcSubresource.mipLevel = i - 1;
		blit.srcSubresource.baseArrayLayer = 0;
		blit.srcSubresource.layerCount = m_arrayLayers;
		blit.dstOffsets[0] = { 0, 0, 0 };
		blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
		blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.dstSubresource.mipLevel = i;
		blit.dstSubresource.baseArrayLayer = 0;
		blit.dstSubresource.layerCount = m_arrayLayers;

		vkCmdBlitImage(commandBuffer,
			m_vkImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			m_vkImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &blit,
			VK_FILTER_LINEAR);

		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		if (mipWidth > 1) mipWidth /= 2;
		if (mipHeight > 1) mipHeight /= 2;
	}
	barrier.subresourceRange.baseMipLevel = m_mipLevels - 1;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

	vkCmdPipelineBarrier(commandBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
		0, nullptr,
		0, nullptr,
		1, &barrier);
	m_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
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


void Image::TransitImageLayout(const VkCommandBuffer commandBuffer, const VkImageLayout newLayout)
{
	//if (newLayout == m_layout) return;
	Graphics::TransitImageLayout(commandBuffer, m_vkImage, m_format, m_arrayLayers, m_layout, newLayout, m_mipLevels);
	m_layout = newLayout;
}

const VmaAllocationInfo& Image::GetVmaAllocationInfo() const
{
	return m_vmaAllocationInfo;
}
#ifdef _WIN64
void* Image::GetVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType) const
{
	void* handle;

	VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
	vkMemoryGetWin32HandleInfoKHR.sType =
		VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
	vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
	vkMemoryGetWin32HandleInfoKHR.memory = m_vmaAllocationInfo.deviceMemory;
	vkMemoryGetWin32HandleInfoKHR.handleType =
		(VkExternalMemoryHandleTypeFlagBitsKHR)externalMemoryHandleType;
	vkGetMemoryWin32HandleKHR(Graphics::GetVkDevice(), &vkMemoryGetWin32HandleInfoKHR, &handle);
	return handle;
}
#else
int Image::GetVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType) const
{
	if (externalMemoryHandleType ==
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR) {
		int fd;

		VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
		vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
		vkMemoryGetFdInfoKHR.pNext = NULL;
		vkMemoryGetFdInfoKHR.memory = m_vmaAllocationInfo.deviceMemory;
		vkMemoryGetFdInfoKHR.handleType =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

		vkGetMemoryFdKHR(Graphics::GetVkDevice(), &vkMemoryGetFdInfoKHR, &fd);

		return fd;
	}
	return -1;
}
#endif

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

void Buffer::UploadData(const size_t size, const void* src)
{
	if (size > m_size) Resize(size);
	if (m_vmaAllocationCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		|| m_vmaAllocationCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)
	{
		void* mapping;
		vmaMapMemory(Graphics::GetVmaAllocator(), m_vmaAllocation, &mapping);
		memcpy(mapping, src, size);
		vmaUnmapMemory(Graphics::GetVmaAllocator(), m_vmaAllocation);
	}
	else
	{
		Buffer stagingBuffer(size);
		stagingBuffer.UploadData(size, src);
		CopyFromBuffer(stagingBuffer, size, 0, 0);
	}
}

void Buffer::DownloadData(const size_t size, void* dst)
{
	if (size > m_size) Resize(size);
	if (m_vmaAllocationCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		|| m_vmaAllocationCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)
	{
		void* mapping;
		vmaMapMemory(Graphics::GetVmaAllocator(), m_vmaAllocation, &mapping);
		memcpy(dst, mapping, size);
		vmaUnmapMemory(Graphics::GetVmaAllocator(), m_vmaAllocation);
	}
	else
	{
		Buffer stagingBuffer(size);
		stagingBuffer.CopyFromBuffer(*this, size, 0, 0);
		stagingBuffer.DownloadData(size, dst);
	}
}

void Buffer::Allocate(VkBufferCreateInfo bufferCreateInfo,
                      const VmaAllocationCreateInfo& vmaAllocationCreateInfo)
{
	VkExternalMemoryBufferCreateInfo vkExternalMemBufferCreateInfo = {};
	vkExternalMemBufferCreateInfo.sType =
		VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	vkExternalMemBufferCreateInfo.pNext = NULL;
#ifdef _WIN64
	vkExternalMemBufferCreateInfo.handleTypes =
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
	vkExternalMemBufferCreateInfo.handleTypes =
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

	bufferCreateInfo.pNext = &vkExternalMemBufferCreateInfo;

	if (vmaCreateBuffer(Graphics::GetVmaAllocator(), &bufferCreateInfo, &vmaAllocationCreateInfo, &m_vkBuffer, &m_vmaAllocation, &m_vmaAllocationInfo)) {
		throw std::runtime_error("Failed to create buffer!");
	}
	assert(bufferCreateInfo.usage != 0);
	m_flags = bufferCreateInfo.flags;
	m_size = bufferCreateInfo.size;
	m_usage = bufferCreateInfo.usage;
	m_sharingMode = bufferCreateInfo.sharingMode;
	ApplyVector(m_queueFamilyIndices, bufferCreateInfo.queueFamilyIndexCount, bufferCreateInfo.pQueueFamilyIndices);
	m_vmaAllocationCreateInfo = vmaAllocationCreateInfo;
}

Buffer::Buffer(const size_t stagingBufferSize, bool randomAccess)
{
	VkBufferCreateInfo stagingBufferCreateInfo{};
	stagingBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingBufferCreateInfo.size = stagingBufferSize;
	stagingBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	stagingBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo stagingBufferVmaAllocationCreateInfo{};
	stagingBufferVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
	stagingBufferVmaAllocationCreateInfo.flags = randomAccess ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	Allocate(stagingBufferCreateInfo, stagingBufferVmaAllocationCreateInfo);
}

Buffer::Buffer(const VkBufferCreateInfo& bufferCreateInfo)
{
	VmaAllocationCreateInfo allocInfo = {};
	allocInfo.usage = VMA_MEMORY_USAGE_AUTO;	
	Allocate(bufferCreateInfo, allocInfo);
}

Buffer::Buffer(const VkBufferCreateInfo& bufferCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo)
{
	Allocate(bufferCreateInfo, vmaAllocationCreateInfo);
}

void Buffer::Resize(VkDeviceSize newSize)
{
	if (newSize == m_size) return;
	if (m_vkBuffer != VK_NULL_HANDLE || m_vmaAllocation != VK_NULL_HANDLE)
	{
		vmaDestroyBuffer(Graphics::GetVmaAllocator(), m_vkBuffer, m_vmaAllocation);
		m_vkBuffer = VK_NULL_HANDLE;
		m_vmaAllocation = VK_NULL_HANDLE;
		m_vmaAllocationInfo = {};
	}	
	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.flags = m_flags;
	bufferCreateInfo.size = newSize;
	bufferCreateInfo.usage = m_usage;
	bufferCreateInfo.sharingMode = m_sharingMode;
	bufferCreateInfo.queueFamilyIndexCount = m_queueFamilyIndices.size();
	bufferCreateInfo.pQueueFamilyIndices = m_queueFamilyIndices.data();

	VkExternalMemoryBufferCreateInfo vkExternalMemBufferCreateInfo = {};
	vkExternalMemBufferCreateInfo.sType =
		VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
	vkExternalMemBufferCreateInfo.pNext = NULL;
#ifdef _WIN64
	vkExternalMemBufferCreateInfo.handleTypes =
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
	vkExternalMemBufferCreateInfo.handleTypes =
		VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

	bufferCreateInfo.pNext = &vkExternalMemBufferCreateInfo;
	if (vmaCreateBuffer(Graphics::GetVmaAllocator(), &bufferCreateInfo, &m_vmaAllocationCreateInfo, &m_vkBuffer, &m_vmaAllocation, &m_vmaAllocationInfo)) {
		throw std::runtime_error("Failed to create buffer!");
	}
	m_size = newSize;
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

void Buffer::CopyFromBuffer(const Buffer& srcBuffer, const VkDeviceSize size, const VkDeviceSize srcOffset, const VkDeviceSize dstOffset)
{
	Resize(size);
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer)
		{
			VkBufferCopy copyRegion{};
			copyRegion.size = size;
			copyRegion.srcOffset = srcOffset;
			copyRegion.dstOffset = dstOffset;
			vkCmdCopyBuffer(commandBuffer, srcBuffer.GetVkBuffer(), m_vkBuffer, 1, &copyRegion);
		});
}

void Buffer::CopyFromImage(Image& srcImage, const VkBufferImageCopy& imageCopyInfo) const
{
	Graphics::ImmediateSubmit([&](const VkCommandBuffer commandBuffer)
		{
			const auto prevLayout = srcImage.GetLayout();
			srcImage.TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
			vkCmdCopyImageToBuffer(commandBuffer, srcImage.GetVkImage(), srcImage.GetLayout(), m_vkBuffer, 1, &imageCopyInfo);
			srcImage.TransitImageLayout(commandBuffer, prevLayout);
		}
	);
}

void Buffer::CopyFromImage(Image& srcImage)
{
	Resize(srcImage.GetExtent().width * srcImage.GetExtent().height * sizeof(glm::vec4));
	VkBufferImageCopy imageCopyInfo{};
	imageCopyInfo.bufferOffset = 0;
	imageCopyInfo.bufferRowLength = 0;
	imageCopyInfo.bufferImageHeight = 0;
	imageCopyInfo.imageSubresource.layerCount = 1;
	imageCopyInfo.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	imageCopyInfo.imageSubresource.baseArrayLayer = 0;
	imageCopyInfo.imageSubresource.mipLevel = 0;

	imageCopyInfo.imageExtent = srcImage.GetExtent();
	imageCopyInfo.imageOffset.x = 0;
	imageCopyInfo.imageOffset.y = 0;
	imageCopyInfo.imageOffset.z = 0;
	CopyFromImage(srcImage, imageCopyInfo);
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

DescriptorSetLayout::~DescriptorSetLayout()
{
	if (m_vkDescriptorSetLayout != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorSetLayout(Graphics::GetVkDevice(), m_vkDescriptorSetLayout, nullptr);
		m_vkDescriptorSetLayout = VK_NULL_HANDLE;
	}
}

void DescriptorSetLayout::PushDescriptorBinding(uint32_t bindingIndex, VkDescriptorType type, VkShaderStageFlags stageFlags, VkDescriptorBindingFlags bindingFlags, const uint32_t descriptorCount)
{
	DescriptorBinding binding;
	VkDescriptorSetLayoutBinding bindingInfo{};
	bindingInfo.binding = bindingIndex;
	bindingInfo.descriptorCount = descriptorCount;
	bindingInfo.descriptorType = type;
	bindingInfo.pImmutableSamplers = nullptr;
	bindingInfo.stageFlags = stageFlags;
	binding.m_binding = bindingInfo;
	binding.m_bindingFlags = bindingFlags;
	m_descriptorSetLayoutBindings[bindingIndex] = binding;
}

void DescriptorSetLayout::Initialize()
{
	if(m_vkDescriptorSetLayout != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorSetLayout(Graphics::GetVkDevice(), m_vkDescriptorSetLayout, nullptr);
		m_vkDescriptorSetLayout = VK_NULL_HANDLE;
	}
	
	std::vector<VkDescriptorSetLayoutBinding> listOfBindings;
	std::vector<VkDescriptorBindingFlags> listOfBindingFlags;
	for(const auto& binding : m_descriptorSetLayoutBindings)
	{
		listOfBindings.emplace_back(binding.second.m_binding);
		listOfBindingFlags.emplace_back(binding.second.m_bindingFlags);
	}
	
	VkDescriptorSetLayoutBindingFlagsCreateInfoEXT extendedInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT, nullptr };
	extendedInfo.bindingCount = static_cast<uint32_t>(listOfBindingFlags.size());
	extendedInfo.pBindingFlags = listOfBindingFlags.data();

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
	descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(listOfBindings.size());
	descriptorSetLayoutCreateInfo.pBindings = listOfBindings.data();
	descriptorSetLayoutCreateInfo.pNext = &extendedInfo;
	Graphics::CheckVk(vkCreateDescriptorSetLayout(Graphics::GetVkDevice(), &descriptorSetLayoutCreateInfo, nullptr, &m_vkDescriptorSetLayout));
}

const VkDescriptorSet& DescriptorSet::GetVkDescriptorSet() const
{
	return m_descriptorSet;
}

DescriptorSet::~DescriptorSet()
{
	if(m_descriptorSet != VK_NULL_HANDLE)
	{
		Graphics::CheckVk(vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), 1, &m_descriptorSet));
		m_descriptorSet = VK_NULL_HANDLE;
	}
}

DescriptorSet::DescriptorSet(const std::shared_ptr<DescriptorSetLayout>& targetLayout)
{
	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = Graphics::GetDescriptorPool()->GetVkDescriptorPool();
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = &targetLayout->GetVkDescriptorSetLayout();
	
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, &m_descriptorSet) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}
	m_descriptorSetLayout = targetLayout;
}

void DescriptorSet::UpdateImageDescriptorBinding(const uint32_t bindingIndex, const VkDescriptorImageInfo& imageInfo, uint32_t arrayElement) const
{
	const auto& descriptorBinding = m_descriptorSetLayout->m_descriptorSetLayoutBindings[bindingIndex];
	VkWriteDescriptorSet writeInfo{};
	writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeInfo.dstSet = m_descriptorSet;
	writeInfo.dstBinding = bindingIndex;
	writeInfo.dstArrayElement = arrayElement;
	writeInfo.descriptorType = descriptorBinding.m_binding.descriptorType;
	writeInfo.descriptorCount = 1;
	writeInfo.pImageInfo = &imageInfo;
	vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
}

void DescriptorSet::UpdateBufferDescriptorBinding(const uint32_t bindingIndex, const VkDescriptorBufferInfo& bufferInfo, uint32_t arrayElement) const
{
	const auto& descriptorBinding = m_descriptorSetLayout->m_descriptorSetLayoutBindings[bindingIndex];
	VkWriteDescriptorSet writeInfo{};
	writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeInfo.dstSet = m_descriptorSet;
	writeInfo.dstBinding = bindingIndex;
	writeInfo.dstArrayElement = arrayElement;
	writeInfo.descriptorType = descriptorBinding.m_binding.descriptorType;
	writeInfo.descriptorCount = 1;
	writeInfo.pBufferInfo = &bufferInfo;
	vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
}

const VkDescriptorSetLayout& DescriptorSetLayout::GetVkDescriptorSetLayout() const
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
