#include "Texture2D.hpp"

#include "Console.hpp"
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
		const Buffer stagingBuffer{ stagingBufferCreateInfo, stagingBufferVmaAllocationCreateInfo };
		void* data = nullptr;
		vmaMapMemory(Graphics::GetVmaAllocator(), stagingBuffer.GetVmaAllocation(), &data);
		memcpy(data, data, imageSize);
		vmaUnmapMemory(Graphics::GetVmaAllocator(), stagingBuffer.GetVmaAllocation());

		m_image->TransitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		m_image->Copy(stagingBuffer);
		m_image->TransitionImageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		/*
		GLsizei mipmap = static_cast<GLsizei>(log2((glm::max)(width, height))) + 1;
		m_texture = std::make_shared<OpenGLUtils::GLTexture2D>(mipmap, GL_RGBA32F, width, height, true);
		m_texture->SetData(0, format, GL_FLOAT, data);
		m_texture->SetInt(GL_TEXTURE_WRAP_S, GL_REPEAT);
		m_texture->SetInt(GL_TEXTURE_WRAP_T, GL_REPEAT);
		m_texture->SetInt(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		m_texture->SetInt(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		m_texture->GenerateMipMap();
		*/
		stbi_image_free(data);
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
	return { m_image->GetVkExtent3D().width, m_image->GetVkExtent3D().height };
}

void Texture2D::StoreToPng(const std::string& path, int resizeX, int resizeY, bool alphaChannel,
	unsigned compressionLevel) const
{
	stbi_write_png_compression_level = static_cast<int>(compressionLevel);
	const auto resolutionX = m_image->GetVkExtent3D().width;
	const auto resolutionY = m_image->GetVkExtent3D().height;
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
	const auto resolutionX = m_image->GetVkExtent3D().width;
	const auto resolutionY = m_image->GetVkExtent3D().height;
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
	const auto resolutionX = m_image->GetVkExtent3D().width;
	const auto resolutionY = m_image->GetVkExtent3D().height;

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
