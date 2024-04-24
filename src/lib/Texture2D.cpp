#include "Texture2D.hpp"

#include "Application.hpp"
#include "ClassRegistry.hpp"
#include "Console.hpp"
#include "EditorLayer.hpp"
#include "Graphics.hpp"
#include "Jobs.hpp"
#include "TextureStorage.hpp"

using namespace EvoEngine;

void Texture2D::SetData(const void* data, const glm::uvec2& resolution)
{
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
	memcpy(deviceData, data, imageSize);
	vmaUnmapMemory(Graphics::GetVmaAllocator(), stagingBuffer.GetVmaAllocation());

	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer)
		{
			m_image->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			m_image->CopyFromBuffer(commandBuffer, stagingBuffer.GetVkBuffer());
			m_image->GenerateMipmaps(commandBuffer);
			m_image->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		});

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
	EditorLayer::UpdateTextureId(m_imTextureId, m_sampler->GetVkSampler(), m_imageView->GetVkImageView(), m_image->GetLayout());
	TextureStorage::RegisterTexture2D(std::dynamic_pointer_cast<Texture2D>(GetSelf()));
}

bool Texture2D::SaveInternal(const std::filesystem::path& path)
{
	if (path.extension() == ".png") {
		StoreToPng(path.string());
	}
	else if (path.extension() == ".jpg") {
		StoreToJpg(path.string());
	}
	else if (path.extension() == ".hdr") {
		StoreToHdr(path.string());
	}
	else {
		EVOENGINE_ERROR("Not implemented!");
		return false;
	}
	return true;
}

bool Texture2D::LoadInternal(const std::filesystem::path& path)
{
	if (m_imTextureId != nullptr) {
		ImGui_ImplVulkan_RemoveTexture(static_cast<VkDescriptorSet>(m_imTextureId));
		m_imTextureId = nullptr;
	}
	m_sampler.reset();
	m_imageView.reset();
	m_image.reset();

	stbi_set_flip_vertically_on_load(true);
	int width, height, nrComponents;


	float actualGamma = 2.2f;

	stbi_hdr_to_ldr_gamma(actualGamma);
	stbi_ldr_to_hdr_gamma(actualGamma);

	void* data;
	data = stbi_loadf(path.string().c_str(), &width, &height, &nrComponents, STBI_rgb_alpha);
	if (data)
	{
		SetData(data, { width, height });
	}
	else
	{
		EVOENGINE_ERROR("Texture failed to load at path: " + path.filename().string());
		return false;
	}
	stbi_image_free(data);
	return true;
}

uint32_t Texture2D::GetTextureStorageIndex() const
{
	return m_textureStorageIndex;
}

Texture2D::~Texture2D()
{
	if(const auto self = GetSelf()) TextureStorage::UnRegisterTexture2D(std::dynamic_pointer_cast<Texture2D>(self));
}

void Texture2D::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	if (m_imTextureId) {
		static float debugSacle = 0.25f;
		ImGui::DragFloat("Scale", &debugSacle, 0.01f, 0.1f, 1.0f);
		debugSacle = glm::clamp(debugSacle, 0.1f, 1.0f);
		ImGui::Image(m_imTextureId,
			ImVec2(m_image->GetExtent().width * debugSacle, m_image->GetExtent().height * debugSacle),
			ImVec2(0, 1),
			ImVec2(1, 0));
	}
}

glm::vec2 Texture2D::GetResolution() const
{
	return { m_image->GetExtent().width, m_image->GetExtent().height };
}

void Texture2D::StoreToPng(const std::string& path, int resizeX, int resizeY,
	unsigned compressionLevel) const
{
	stbi_write_png_compression_level = static_cast<int>(compressionLevel);
	const auto resolutionX = m_image->GetExtent().width;
	const auto resolutionY = m_image->GetExtent().height;
	const size_t storeChannels = 4;
	const size_t channels = 4;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * channels);
	//Retrieve image data here.
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*m_image);
	imageBuffer.DownloadVector(dst, resolutionX * resolutionY * channels);
	std::vector<uint8_t> pixels;
	if (resizeX > 0 && resizeY > 0 && (resizeX != resolutionX || resizeY != resolutionY))
	{
		std::vector<float> res;
		res.resize(resizeX * resizeY * storeChannels);
		stbir_resize_float(dst.data(), resolutionX, resolutionY, 0, res.data(), resizeX, resizeY, 0, storeChannels);
		pixels.resize(resizeX * resizeY * storeChannels);
		for (int i = 0; i < resizeX * resizeY; i++)
		{
			pixels[i * storeChannels] = glm::clamp<int>(int(255.9f * res[i * channels]), 0, 255);
			pixels[i * storeChannels + 1] = glm::clamp<int>(int(255.9f * res[i * channels + 1]), 0, 255);
			pixels[i * storeChannels + 2] = glm::clamp<int>(int(255.9f * res[i * channels + 2]), 0, 255);
			if (storeChannels == 4)
				pixels[i * storeChannels + 3] = glm::clamp<int>(int(255.9f * res[i * channels + 3]), 0, 255);
		}
		stbi_flip_vertically_on_write(true);
		stbi_write_png(path.c_str(), resizeX, resizeY, storeChannels, pixels.data(), resizeX * storeChannels);
	}
	else
	{
		pixels.resize(resolutionX * resolutionY * channels);
		for (int i = 0; i < resolutionX * resolutionY; i++)
		{
			pixels[i * storeChannels] = glm::clamp<int>(int(255.9f * dst[i * channels]), 0, 255);
			pixels[i * storeChannels + 1] = glm::clamp<int>(int(255.9f * dst[i * channels + 1]), 0, 255);
			pixels[i * storeChannels + 2] = glm::clamp<int>(int(255.9f * dst[i * channels + 2]), 0, 255);
			if (storeChannels == 4)
				pixels[i * storeChannels + 3] = glm::clamp<int>(int(255.9f * dst[i * channels + 3]), 0, 255);
		}
		stbi_flip_vertically_on_write(true);
		stbi_write_png(path.c_str(), resolutionX, resolutionY, storeChannels, pixels.data(), resolutionX * storeChannels);
	}
}

void Texture2D::StoreToJpg(const std::string& path, int resizeX, int resizeY, unsigned quality) const
{
	const auto resolutionX = m_image->GetExtent().width;
	const auto resolutionY = m_image->GetExtent().height;
	std::vector<float> dst;
	const size_t storeChannels = 3;
	const size_t channels = 4;
	dst.resize(resolutionX * resolutionY * channels);
	//Retrieve image data here.
	//Retrieve image data here.
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*m_image);
	imageBuffer.DownloadVector(dst, resolutionX * resolutionY * channels);
	std::vector<uint8_t> pixels;
	if (resizeX > 0 && resizeY > 0 && (resizeX != resolutionX || resizeY != resolutionY))
	{
		std::vector<float> res;
		res.resize(resizeX * resizeY * storeChannels);
		stbir_resize_float(dst.data(), resolutionX, resolutionY, 0, res.data(), resizeX, resizeY, 0, storeChannels);
		pixels.resize(resizeX * resizeY * storeChannels);
		for (int i = 0; i < resizeX * resizeY; i++)
		{
			pixels[i * storeChannels] = glm::clamp<int>(int(255.9f * res[i * channels]), 0, 255);
			pixels[i * storeChannels + 1] = glm::clamp<int>(int(255.9f * res[i * channels + 1]), 0, 255);
			pixels[i * storeChannels + 2] = glm::clamp<int>(int(255.9f * res[i * channels + 2]), 0, 255);
			if (storeChannels == 4)
				pixels[i * storeChannels + 3] = glm::clamp<int>(int(255.9f * res[i * channels + 3]), 0, 255);
		}
		stbi_flip_vertically_on_write(true);
		stbi_write_jpg(path.c_str(), resizeX, resizeY, storeChannels, pixels.data(), quality);
	}
	else
	{
		pixels.resize(resolutionX * resolutionY * 3);
		for (int i = 0; i < resolutionX * resolutionY; i++)
		{
			pixels[i * storeChannels] = glm::clamp<int>(int(255.9f * dst[i * channels]), 0, 255);
			pixels[i * storeChannels + 1] = glm::clamp<int>(int(255.9f * dst[i * channels + 1]), 0, 255);
			pixels[i * storeChannels + 2] = glm::clamp<int>(int(255.9f * dst[i * channels + 2]), 0, 255);
			if (storeChannels == 4)
				pixels[i * storeChannels + 3] = glm::clamp<int>(int(255.9f * dst[i * channels + 3]), 0, 255);
		}
		stbi_flip_vertically_on_write(true);
		stbi_write_jpg(path.c_str(), resolutionX, resolutionY, storeChannels, pixels.data(), quality);
	}
}

void Texture2D::StoreToHdr(const std::string& path, int resizeX, int resizeY, bool alphaChannel, unsigned quality) const
{
	const auto resolutionX = m_image->GetExtent().width;
	const auto resolutionY = m_image->GetExtent().height;
	const size_t channels = 4;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * channels);
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*m_image);
	imageBuffer.DownloadVector(dst, resolutionX * resolutionY * channels);
	//Retrieve image data here.

	stbi_flip_vertically_on_write(true);
	if (resizeX > 0 && resizeY > 0 && (resizeX != resolutionX || resizeY != resolutionY))
	{
		std::vector<float> pixels;
		pixels.resize(resizeX * resizeY * channels);
		stbir_resize_float(dst.data(), resolutionX, resolutionY, 0, pixels.data(), resizeX, resizeY, 0, channels);
		stbi_write_hdr(path.c_str(), resolutionX, resolutionY, channels, pixels.data());
	}
	else
	{
		stbi_write_hdr(path.c_str(), resolutionX, resolutionY, channels, dst.data());
	}
}

ImTextureID Texture2D::GetImTextureId() const
{
	return m_imTextureId;
}

VkImageLayout Texture2D::GetLayout() const
{
	return m_image->GetLayout();
}

VkImage Texture2D::GetVkImage() const
{
	if (m_image)
	{
		return m_image->GetVkImage();
	}
	return VK_NULL_HANDLE;
}

VkImageView Texture2D::GetVkImageView() const
{
	if (m_imageView)
	{
		return m_imageView->GetVkImageView();
	}
	return VK_NULL_HANDLE;
}

VkSampler Texture2D::GetVkSampler() const
{
	if (m_sampler)
	{
		return m_sampler->GetVkSampler();
	}
	return VK_NULL_HANDLE;
}

std::shared_ptr<Image> Texture2D::GetImage() const
{
	return m_image;
}

void Texture2D::GetRgbaChannelData(std::vector<glm::vec4>& dst, int resizeX, int resizeY) const
{
	const auto resolutionX = m_image->GetExtent().width;
	const auto resolutionY = m_image->GetExtent().height;
	dst.resize(resolutionX * resolutionY);
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*m_image);
	imageBuffer.DownloadVector(dst, resolutionX * resolutionY);
}

void Texture2D::GetRgbChannelData(std::vector<glm::vec3>& dst, int resizeX, int resizeY) const
{
	const auto resolutionX = m_image->GetExtent().width;
	const auto resolutionY = m_image->GetExtent().height;
	std::vector<glm::vec4> pixels;
	pixels.resize(resolutionX * resolutionY);
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*m_image);
	imageBuffer.DownloadVector(pixels, resolutionX * resolutionY);
	dst.resize(pixels.size());
	Jobs::RunParallelFor(pixels.size(), [&](unsigned i)
		{
			dst[i] = pixels[i];
		}
	);
}

void Texture2D::GetRgChannelData(std::vector<glm::vec2>& dst, int resizeX, int resizeY) const
{
	const auto resolutionX = m_image->GetExtent().width;
	const auto resolutionY = m_image->GetExtent().height;
	std::vector<glm::vec4> pixels;
	pixels.resize(resolutionX * resolutionY);
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*m_image);
	imageBuffer.DownloadVector(pixels, resolutionX * resolutionY);
	dst.resize(pixels.size());
	Jobs::RunParallelFor(pixels.size(), [&](unsigned i)
		{
			dst[i] = glm::vec2(pixels[i].r, pixels[i].g);
		}
	);
}

void Texture2D::GetRedChannelData(std::vector<float>& dst, int resizeX, int resizeY) const
{
	const auto resolutionX = m_image->GetExtent().width;
	const auto resolutionY = m_image->GetExtent().height;
	std::vector<glm::vec4> pixels;
	pixels.resize(resolutionX * resolutionY);
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*m_image);
	imageBuffer.DownloadVector(pixels, resolutionX * resolutionY);
	dst.resize(pixels.size());
	Jobs::RunParallelFor(pixels.size(), [&](unsigned i)
		{
			dst[i] = pixels[i].r;
		}
	);
}

void Texture2D::SetRgbaChannelData(const std::vector<glm::vec4>& src, const glm::uvec2& resolution)
{
	SetData(src.data(), resolution);
}

void Texture2D::SetRgbChannelData(const std::vector<glm::vec3>& src, const glm::uvec2& resolution)
{
	std::vector<glm::vec4> imageData;
	imageData.resize(resolution.x * resolution.y);
	Jobs::RunParallelFor(imageData.size(), [&](unsigned i)
		{
			imageData[i] = glm::vec4(src[i], 1.0f);
		}
	);
	SetData(imageData.data(), resolution);
}

void Texture2D::SetRgChannelData(const std::vector<glm::vec2>& src, const glm::uvec2& resolution)
{
	std::vector<glm::vec4> imageData;
	imageData.resize(resolution.x * resolution.y);
	Jobs::RunParallelFor(imageData.size(), [&](unsigned i)
		{
			imageData[i] = glm::vec4(src[i], 0.0f, 1.0f);
		}
	);
	SetData(imageData.data(), resolution);
}

void Texture2D::SetRedChannelData(const std::vector<float>& src, const glm::uvec2& resolution)
{
	std::vector<glm::vec4> imageData;
	imageData.resize(resolution.x * resolution.y);
	Jobs::RunParallelFor(imageData.size(), [&](unsigned i)
		{
			imageData[i] = glm::vec4(src[i], 0.0f, 0.0f, 1.0f);
		}
	);
	SetData(imageData.data(), resolution);
}
