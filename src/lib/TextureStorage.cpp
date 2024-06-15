#include "TextureStorage.hpp"

#include "Application.hpp"
#include "RenderLayer.hpp"
#include "EditorLayer.hpp"
using namespace evo_engine;



void CubemapStorage::Initialize(uint32_t resolution, uint32_t mipLevels)
{
	Clear();
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = resolution;
	imageInfo.extent.height = resolution;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = mipLevels;
	imageInfo.arrayLayers = 6;
	imageInfo.format = Graphics::Constants::TEXTURE_2D;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	m_image = std::make_shared<Image>(imageInfo);

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = m_image->GetVkImage();
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
	viewInfo.format = Graphics::Constants::TEXTURE_2D;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = mipLevels;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 6;

	m_imageView = std::make_shared<ImageView>(viewInfo);


	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_TRUE;
	samplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	if (mipLevels > 1) {
		samplerInfo.minLod = 0;
		samplerInfo.maxLod = static_cast<float>(mipLevels);
	}
	m_sampler = std::make_shared<Sampler>(samplerInfo);

	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer) {
		m_image->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
	);

	for (int i = 0; i < 6; i++)
	{
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_image->GetVkImage();
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = Graphics::Constants::TEXTURE_2D;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = i;
		viewInfo.subresourceRange.layerCount = 1;

		m_faceViews.emplace_back(std::make_shared<ImageView>(viewInfo));
	}

	m_imTextureIds.resize(6);
	for (int i = 0; i < 6; i++)
	{
		EditorLayer::UpdateTextureId(m_imTextureIds[i], m_sampler->GetVkSampler(), m_faceViews[i]->GetVkImageView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}
}

VkImageLayout CubemapStorage::GetLayout() const
{
	return m_image->GetLayout();
}

VkImage CubemapStorage::GetVkImage() const
{
	if (m_image)
	{
		return m_image->GetVkImage();
	}
	return VK_NULL_HANDLE;
}

VkImageView CubemapStorage::GetVkImageView() const
{
	if (m_imageView)
	{
		return m_imageView->GetVkImageView();
	}
	return VK_NULL_HANDLE;
}

VkImageLayout Texture2DStorage::GetLayout() const
{
	return m_image->GetLayout();
}

VkImage Texture2DStorage::GetVkImage() const
{
	if (m_image)
	{
		return m_image->GetVkImage();
	}
	return VK_NULL_HANDLE;
}

VkImageView Texture2DStorage::GetVkImageView() const
{
	if (m_imageView)
	{
		return m_imageView->GetVkImageView();
	}
	return VK_NULL_HANDLE;
}

VkSampler Texture2DStorage::GetVkSampler() const
{
	if (m_sampler)
	{
		return m_sampler->GetVkSampler();
	}
	return VK_NULL_HANDLE;
}
VkSampler CubemapStorage::GetVkSampler() const
{
	if (m_sampler)
	{
		return m_sampler->GetVkSampler();
	}
	return VK_NULL_HANDLE;
}
std::shared_ptr<Image> Texture2DStorage::GetImage() const
{
	return m_image;
}
std::shared_ptr<Image> CubemapStorage::GetImage() const
{
	return m_image;
}

void Texture2DStorage::Initialize(const glm::uvec2& resolution)
{
	Clear();
	uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(resolution.x, resolution.y)))) + 1;
	mipLevels = 1;
	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = resolution.x;
	imageInfo.extent.height = resolution.y;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = mipLevels;
	imageInfo.arrayLayers = 1;
	imageInfo.format = Graphics::Constants::TEXTURE_2D;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	m_image = std::make_shared<Image>(imageInfo);
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = m_image->GetVkImage();
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = Graphics::Constants::TEXTURE_2D;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = imageInfo.mipLevels;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	m_imageView = std::make_shared<ImageView>(viewInfo);


	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_TRUE;
	samplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.minLod = 0;
	samplerInfo.maxLod = static_cast<float>(imageInfo.mipLevels);
	samplerInfo.mipLodBias = 0.0f;

	m_sampler = std::make_shared<Sampler>(samplerInfo);
}

void Texture2DStorage::SetDataImmediately(const std::vector<glm::vec4>& data, const glm::uvec2& resolution)
{
	UploadData(data, resolution);
}

void Texture2DStorage::UploadData(const std::vector<glm::vec4>& data, const glm::uvec2& resolution)
{
	Initialize(resolution);
	const auto imageSize = resolution.x * resolution.y * sizeof(glm::vec4);
	VkBufferCreateInfo stagingBufferCreateInfo{};
	stagingBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingBufferCreateInfo.size = imageSize;
	stagingBufferCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	stagingBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo stagingBufferVmaAllocationCreateInfo{};
	stagingBufferVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
	stagingBufferVmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

	Buffer stagingBuffer{ stagingBufferCreateInfo, stagingBufferVmaAllocationCreateInfo };
	void* deviceData = nullptr;
	vmaMapMemory(Graphics::GetVmaAllocator(), stagingBuffer.GetVmaAllocation(), &deviceData);
	memcpy(deviceData, data.data(), imageSize);
	vmaUnmapMemory(Graphics::GetVmaAllocator(), stagingBuffer.GetVmaAllocation());

	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer)
		{
			m_image->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			m_image->CopyFromBuffer(commandBuffer, stagingBuffer.GetVkBuffer());
			m_image->GenerateMipmaps(commandBuffer);
			m_image->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		});

	
	EditorLayer::UpdateTextureId(m_imTextureId, m_sampler->GetVkSampler(), m_imageView->GetVkImageView(), m_image->GetLayout());

	
}

void Texture2DStorage::Clear()
{
	if (m_imTextureId != nullptr) {
		ImGui_ImplVulkan_RemoveTexture(static_cast<VkDescriptorSet>(m_imTextureId));
		m_imTextureId = nullptr;
	}
	m_sampler.reset();
	m_imageView.reset();
	m_image.reset();
}

void CubemapStorage::Clear()
{
	for(auto& m_imTextureId : m_imTextureIds){
		if (m_imTextureId != nullptr) {
			ImGui_ImplVulkan_RemoveTexture(static_cast<VkDescriptorSet>(m_imTextureId));
			m_imTextureId = nullptr;
		}
	}
	m_sampler.reset();
	m_imageView.reset();
	m_image.reset();
	m_faceViews.clear();

}
void Texture2DStorage::SetData(const std::vector<glm::vec4>& data, const glm::uvec2& resolution)
{
	m_newData = data;
	m_newResolution = resolution;
}

void TextureStorage::DeviceSync()
{
	auto& storage = GetInstance();
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();

	const auto renderLayer = Application::GetLayer<RenderLayer>();

	for (int textureIndex = 0; textureIndex < storage.m_texture2Ds.size(); textureIndex++)
	{
		auto& textureStorage = storage.m_texture2Ds[textureIndex];
		if (textureStorage.m_pendingDelete)
		{
			storage.m_texture2Ds[textureIndex] = storage.m_texture2Ds.back();
			storage.m_texture2Ds[textureIndex].m_handle->m_value = textureIndex;
			storage.m_texture2Ds.pop_back();
			textureIndex--;
		}
	}

	for (int textureIndex = 0; textureIndex < storage.m_texture2Ds.size(); textureIndex++)
	{
		auto& textureStorage = storage.m_texture2Ds[textureIndex];
		if (!textureStorage.m_newData.empty()){
			textureStorage.UploadData(textureStorage.m_newData, textureStorage.m_newResolution);
			textureStorage.m_newData.clear();
			textureStorage.m_newResolution = {};
		}
		VkDescriptorImageInfo imageInfo;
		imageInfo.imageLayout = textureStorage.GetLayout();
		imageInfo.imageView = textureStorage.GetVkImageView();
		imageInfo.sampler = textureStorage.GetVkSampler();
		if(!renderLayer->per_frame_descriptor_sets_.empty()) renderLayer->per_frame_descriptor_sets_[currentFrameIndex]->UpdateImageDescriptorBinding(13, imageInfo, textureIndex);
	}

	for (int textureIndex = 0; textureIndex < storage.m_cubemaps.size(); textureIndex++)
	{
		auto& textureStorage = storage.m_cubemaps[textureIndex];
		if (textureStorage.m_pendingDelete)
		{
			storage.m_cubemaps[textureIndex] = storage.m_cubemaps.back();
			storage.m_cubemaps[textureIndex].m_handle->m_value = textureIndex;
			storage.m_cubemaps.pop_back();
			textureIndex--;
		}
	}

	for (int textureIndex = 0; textureIndex < storage.m_cubemaps.size(); textureIndex++)
	{
		auto& textureStorage = storage.m_cubemaps[textureIndex];
		//if (!textureStorage.m_newData.empty()) textureStorage.UploadData();
		VkDescriptorImageInfo imageInfo;
		imageInfo.imageLayout = textureStorage.GetLayout();
		imageInfo.imageView = textureStorage.GetVkImageView();
		imageInfo.sampler = textureStorage.GetVkSampler();
		if(!renderLayer->per_frame_descriptor_sets_.empty()) renderLayer->per_frame_descriptor_sets_[currentFrameIndex]->UpdateImageDescriptorBinding(14, imageInfo, textureIndex);
	}
}

const Texture2DStorage& TextureStorage::PeekTexture2DStorage(const std::shared_ptr<TextureStorageHandle>& handle)
{
	auto& storage = GetInstance();
	return storage.m_texture2Ds.at(handle->m_value);
}

Texture2DStorage& TextureStorage::RefTexture2DStorage(const std::shared_ptr<TextureStorageHandle>& handle)
{
	auto& storage = GetInstance();
	return storage.m_texture2Ds.at(handle->m_value);
}

const CubemapStorage& TextureStorage::PeekCubemapStorage(const std::shared_ptr<TextureStorageHandle>& handle)
{
	auto& storage = GetInstance();
	return storage.m_cubemaps.at(handle->m_value);
}

CubemapStorage& TextureStorage::RefCubemapStorage(const std::shared_ptr<TextureStorageHandle>& handle)
{
	auto& storage = GetInstance();
	return storage.m_cubemaps.at(handle->m_value);
}

void TextureStorage::UnRegisterTexture2D(const std::shared_ptr<TextureStorageHandle>& handle)
{
	auto& storage = GetInstance();
	storage.m_texture2Ds[handle->m_value].m_pendingDelete = true;
}

void TextureStorage::UnRegisterCubemap(const std::shared_ptr<TextureStorageHandle>& handle)
{
	auto& storage = GetInstance();
	storage.m_cubemaps[handle->m_value].m_pendingDelete = true;
}

std::shared_ptr<TextureStorageHandle> TextureStorage::RegisterTexture2D()
{
	auto& storage = GetInstance();
	const auto retVal = std::make_shared<TextureStorageHandle>();
	retVal->m_value = storage.m_texture2Ds.size();
	storage.m_texture2Ds.emplace_back();
	auto& newTexture2DStorage = storage.m_texture2Ds.back();
	newTexture2DStorage.m_handle = retVal;
	storage.m_texture2Ds.back().Initialize({1, 1});
	return retVal;
}

std::shared_ptr<TextureStorageHandle> TextureStorage::RegisterCubemap()
{
	auto& storage = GetInstance();
	const auto retVal = std::make_shared<TextureStorageHandle>();
	retVal->m_value = storage.m_cubemaps.size();
	storage.m_cubemaps.emplace_back();
	auto& newCubemapStorage = storage.m_cubemaps.back();
	newCubemapStorage.m_handle = retVal;
	storage.m_cubemaps.back().Initialize(1, 1);
	return retVal;
}

void TextureStorage::Initialize()
{
}



