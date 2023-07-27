#include "Texture2D.hpp"

#include "Application.hpp"
#include "ClassRegistry.hpp"
#include "Console.hpp"
#include "EditorLayer.hpp"
#include "Graphics.hpp"

using namespace EvoEngine;

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
	stbi_set_flip_vertically_on_load(true);
	int width, height, nrComponents;
	float actualGamma = 1.0f;
	if (path.extension() == ".hdr")
	{
		actualGamma = 2.2f;
	}

	stbi_hdr_to_ldr_gamma(actualGamma);
	stbi_ldr_to_hdr_gamma(actualGamma);

	float* data = stbi_loadf(path.string().c_str(), &width, &height, &nrComponents, 0);
	if (data)
	{
		if (nrComponents == 1)
		{
			m_imageFormat = VK_FORMAT_R32_SFLOAT;
		}
		else if (nrComponents == 2)
		{
			m_imageFormat = VK_FORMAT_R32G32_SFLOAT;
		}
		else if (nrComponents == 3)
		{
			m_imageFormat = VK_FORMAT_R32G32B32_SFLOAT;
		}
		else if (nrComponents == 4)
		{
			m_imageFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
		}
		else
		{
			EVOENGINE_ERROR("Format not supported!");
		}

		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = m_imageFormat;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		m_image = std::make_unique<Image>(imageInfo);
		const auto imageSize = nrComponents * width * height * sizeof(float);
		VkBufferCreateInfo stagingBufferCreateInfo{};
		stagingBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		stagingBufferCreateInfo.size = imageSize;
		stagingBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
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
				m_image->TransitionImageLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
				m_image->Copy(commandBuffer, stagingBuffer.GetVkBuffer());
				m_image->TransitionImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			});

		
		stbi_image_free(data);


		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_image->GetVkImage();
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = m_imageFormat;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		m_imageView = std::make_unique<ImageView>(viewInfo);


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

		m_sampler = std::make_unique<Sampler>(samplerInfo);
		if (const auto editorLayer = Application::GetLayer<EditorLayer>()) {
			editorLayer->UpdateTextureId(m_imTextureId, m_imageView->GetVkImageView(), m_image->GetLayout());
		}
	}
	else
	{
		EVOENGINE_ERROR("Texture failed to load at path: " + path.filename().string());
		stbi_image_free(data);
		return false;
	}
	m_gamma = actualGamma;
	return true;
}

glm::vec2 Texture2D::GetResolution() const
{
	return { m_image->GetExtent().width, m_image->GetExtent().height };
}

void Texture2D::StoreToPng(const std::string& path, int resizeX, int resizeY, bool alphaChannel,
	unsigned compressionLevel) const
{
	stbi_write_png_compression_level = static_cast<int>(compressionLevel);
	const auto resolutionX = m_image->GetExtent().width;
	const auto resolutionY = m_image->GetExtent().height;
	float channels = 3;
	if (alphaChannel)
		channels = 4;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * channels);
	//Retrieve image data here.
	std::vector<uint8_t> pixels;
	if (resizeX > 0 && resizeY > 0 && (resizeX != resolutionX || resizeY != resolutionY))
	{
		std::vector<float> res;
		res.resize(resizeX * resizeY * channels);
		stbir_resize_float(dst.data(), resolutionX, resolutionY, 0, res.data(), resizeX, resizeY, 0, channels);
		pixels.resize(resizeX * resizeY * channels);
		for (int i = 0; i < resizeX * resizeY; i++)
		{
			pixels[i * channels] = glm::clamp<int>(int(255.99f * res[i * channels]), 0, 255);
			pixels[i * channels + 1] = glm::clamp<int>(int(255.99f * res[i * channels + 1]), 0, 255);
			pixels[i * channels + 2] = glm::clamp<int>(int(255.99f * res[i * channels + 2]), 0, 255);
			if (channels == 4)
				pixels[i * channels + 3] = glm::clamp<int>(int(255.99f * res[i * channels + 3]), 0, 255);
		}
		stbi_flip_vertically_on_write(true);
		stbi_write_png(path.c_str(), resizeX, resizeY, channels, pixels.data(), resizeX * channels);
	}
	else
	{
		pixels.resize(resolutionX * resolutionY * channels);
		for (int i = 0; i < resolutionX * resolutionY; i++)
		{
			pixels[i * channels] = glm::clamp<int>(int(255.99f * dst[i * channels]), 0, 255);
			pixels[i * channels + 1] = glm::clamp<int>(int(255.99f * dst[i * channels + 1]), 0, 255);
			pixels[i * channels + 2] = glm::clamp<int>(int(255.99f * dst[i * channels + 2]), 0, 255);
			if (channels == 4)
				pixels[i * channels + 3] = glm::clamp<int>(int(255.99f * dst[i * channels + 3]), 0, 255);
		}
		stbi_flip_vertically_on_write(true);
		stbi_write_png(path.c_str(), resolutionX, resolutionY, channels, pixels.data(), resolutionX * channels);
	}
}

void Texture2D::StoreToJpg(const std::string& path, int resizeX, int resizeY, unsigned quality) const
{
	const auto resolutionX = m_image->GetExtent().width;
	const auto resolutionY = m_image->GetExtent().height;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * 3);
	//Retrieve image data here.

	std::vector<uint8_t> pixels;
	if (resizeX > 0 && resizeY > 0 && (resizeX != resolutionX || resizeY != resolutionY))
	{
		std::vector<float> res;
		res.resize(resizeX * resizeY * 3);
		stbir_resize_float(dst.data(), resolutionX, resolutionY, 0, res.data(), resizeX, resizeY, 0, 3);
		pixels.resize(resizeX * resizeY * 3);
		for (int i = 0; i < resizeX * resizeY; i++)
		{
			pixels[i * 3] = glm::clamp<int>(int(255.99f * res[i * 3]), 0, 255);
			pixels[i * 3 + 1] = glm::clamp<int>(int(255.99f * res[i * 3 + 1]), 0, 255);
			pixels[i * 3 + 2] = glm::clamp<int>(int(255.99f * res[i * 3 + 2]), 0, 255);
		}
		stbi_flip_vertically_on_write(true);
		stbi_write_jpg(path.c_str(), resizeX, resizeY, 3, pixels.data(), quality);
	}
	else
	{
		pixels.resize(resolutionX * resolutionY * 3);
		for (int i = 0; i < resolutionX * resolutionY; i++)
		{
			pixels[i * 3] = glm::clamp<int>(int(255.99f * dst[i * 3]), 0, 255);
			pixels[i * 3 + 1] = glm::clamp<int>(int(255.99f * dst[i * 3 + 1]), 0, 255);
			pixels[i * 3 + 2] = glm::clamp<int>(int(255.99f * dst[i * 3 + 2]), 0, 255);
		}
		stbi_flip_vertically_on_write(true);
		stbi_write_jpg(path.c_str(), resolutionX, resolutionY, 3, pixels.data(), quality);
	}
}

void Texture2D::StoreToHdr(const std::string& path, int resizeX, int resizeY, bool alphaChannel, unsigned quality) const
{
	const auto resolutionX = m_image->GetExtent().width;
	const auto resolutionY = m_image->GetExtent().height;

	float channels = 3;
	if (alphaChannel)
		channels = 4;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * channels);
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
