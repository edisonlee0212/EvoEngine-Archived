#include "Texture2D.hpp"

#include "Application.hpp"
#include "ClassRegistry.hpp"
#include "Console.hpp"
#include "EditorLayer.hpp"
#include "Graphics.hpp"
#include "Jobs.hpp"
#include "TextureStorage.hpp"

using namespace EvoEngine;

void Texture2D::SetData(const std::vector<glm::vec4>& data, const glm::uvec2& resolution) const
{
	auto& textureStorage = TextureStorage::RefTexture2DStorage(m_textureStorageHandle);
	textureStorage.SetData(data, resolution);
}

bool Texture2D::SaveInternal(const std::filesystem::path& path) const
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

	float actualGamma = 2.2f;

	stbi_hdr_to_ldr_gamma(actualGamma);
	stbi_ldr_to_hdr_gamma(actualGamma);

	void* data = stbi_loadf(path.string().c_str(), &width, &height, &nrComponents, STBI_rgb_alpha);

	if (data)
	{
		std::vector<glm::vec4> imageData;
		imageData.resize(width * height);
		memcpy(imageData.data(), data, sizeof(glm::vec4) * width * height);
		SetData(imageData, { width, height });
	}
	else
	{
		EVOENGINE_ERROR("Texture failed to load at path: " + path.filename().string());
		return false;
	}
	stbi_image_free(data);
	return true;
}

Texture2D::Texture2D()
{
	m_textureStorageHandle = TextureStorage::RegisterTexture2D();
}

const Texture2DStorage& Texture2D::PeekTexture2DStorage() const
{
	return TextureStorage::PeekTexture2DStorage(m_textureStorageHandle);
}

Texture2DStorage& Texture2D::RefTexture2DStorage() const
{
	return TextureStorage::RefTexture2DStorage(m_textureStorageHandle);
}

uint32_t Texture2D::GetTextureStorageIndex() const
{
	return m_textureStorageHandle->m_value;
}

Texture2D::~Texture2D()
{
	TextureStorage::UnRegisterTexture2D(m_textureStorageHandle);
}

bool Texture2D::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	bool changed = false;

	const auto textureStorage = PeekTexture2DStorage();

	if (textureStorage.m_imTextureId) {
		static float debugSacle = 0.25f;
		ImGui::DragFloat("Scale", &debugSacle, 0.01f, 0.1f, 1.0f);
		debugSacle = glm::clamp(debugSacle, 0.1f, 1.0f);
		ImGui::Image(textureStorage.m_imTextureId,
			ImVec2(textureStorage.m_image->GetExtent().width * debugSacle, textureStorage.m_image->GetExtent().height * debugSacle),
			ImVec2(0, 1),
			ImVec2(1, 0));
	}

	return changed;
}

glm::vec2 Texture2D::GetResolution() const
{
	const auto textureStorage = PeekTexture2DStorage();
	return { textureStorage.m_image->GetExtent().width, textureStorage.m_image->GetExtent().height };
}

void Texture2D::StoreToPng(const std::string& path, int resizeX, int resizeY,
	unsigned compressionLevel) const
{
	const auto textureStorage = PeekTexture2DStorage();
	stbi_write_png_compression_level = static_cast<int>(compressionLevel);
	const auto resolutionX = textureStorage.m_image->GetExtent().width;
	const auto resolutionY = textureStorage.m_image->GetExtent().height;
	const size_t storeChannels = 4;
	const size_t channels = 4;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * channels);
	//Retrieve image data here.
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*textureStorage.m_image);
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
	const auto textureStorage = PeekTexture2DStorage();
	const auto resolutionX = textureStorage.m_image->GetExtent().width;
	const auto resolutionY = textureStorage.m_image->GetExtent().height;
	std::vector<float> dst;
	const size_t storeChannels = 3;
	const size_t channels = 4;
	dst.resize(resolutionX * resolutionY * channels);
	//Retrieve image data here.
	//Retrieve image data here.
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*textureStorage.m_image);
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
	const auto textureStorage = PeekTexture2DStorage();
	const auto resolutionX = textureStorage.m_image->GetExtent().width;
	const auto resolutionY = textureStorage.m_image->GetExtent().height;
	const size_t channels = 4;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * channels);
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*textureStorage.m_image);
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
	const auto textureStorage = PeekTexture2DStorage();
	return textureStorage.m_imTextureId;
}

VkImageLayout Texture2D::GetLayout() const
{
	const auto textureStorage = PeekTexture2DStorage();
	return textureStorage.m_image->GetLayout();
}

VkImage Texture2D::GetVkImage() const
{
	const auto textureStorage = PeekTexture2DStorage();
	if (textureStorage.m_image)
	{
		return textureStorage.m_image->GetVkImage();
	}
	return VK_NULL_HANDLE;
}

VkImageView Texture2D::GetVkImageView() const
{
	const auto textureStorage = PeekTexture2DStorage();
	if (textureStorage.m_imageView)
	{
		return textureStorage.m_imageView->GetVkImageView();
	}
	return VK_NULL_HANDLE;
}

VkSampler Texture2D::GetVkSampler() const
{
	const auto textureStorage = PeekTexture2DStorage();
	if (textureStorage.m_sampler)
	{
		return textureStorage.m_sampler->GetVkSampler();
	}
	return VK_NULL_HANDLE;
}

std::shared_ptr<Image> Texture2D::GetImage() const
{
	const auto textureStorage = PeekTexture2DStorage();
	return textureStorage.m_image;
}

void Texture2D::GetRgbaChannelData(std::vector<glm::vec4>& dst, int resizeX, int resizeY) const
{
	const auto textureStorage = PeekTexture2DStorage();
	const auto resolutionX = textureStorage.m_image->GetExtent().width;
	const auto resolutionY = textureStorage.m_image->GetExtent().height;
	dst.resize(resolutionX * resolutionY);
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*textureStorage.m_image);
	imageBuffer.DownloadVector(dst, resolutionX * resolutionY);
}

void Texture2D::GetRgbChannelData(std::vector<glm::vec3>& dst, int resizeX, int resizeY) const
{
	const auto textureStorage = PeekTexture2DStorage();
	const auto resolutionX = textureStorage.m_image->GetExtent().width;
	const auto resolutionY = textureStorage.m_image->GetExtent().height;
	std::vector<glm::vec4> pixels;
	pixels.resize(resolutionX * resolutionY);
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*textureStorage.m_image);
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
	const auto textureStorage = PeekTexture2DStorage();
	const auto resolutionX = textureStorage.m_image->GetExtent().width;
	const auto resolutionY = textureStorage.m_image->GetExtent().height;
	std::vector<glm::vec4> pixels;
	pixels.resize(resolutionX * resolutionY);
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*textureStorage.m_image);
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
	const auto textureStorage = PeekTexture2DStorage();
	const auto resolutionX = textureStorage.m_image->GetExtent().width;
	const auto resolutionY = textureStorage.m_image->GetExtent().height;
	std::vector<glm::vec4> pixels;
	pixels.resize(resolutionX * resolutionY);
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*textureStorage.m_image);
	imageBuffer.DownloadVector(pixels, resolutionX * resolutionY);
	dst.resize(pixels.size());
	Jobs::RunParallelFor(pixels.size(), [&](unsigned i)
		{
			dst[i] = pixels[i].r;
		}
	);
}

void Texture2D::SetRgbaChannelData(const std::vector<glm::vec4>& src, const glm::uvec2& resolution) const
{
	SetData(src, resolution);
}

void Texture2D::SetRgbChannelData(const std::vector<glm::vec3>& src, const glm::uvec2& resolution) const
{
	std::vector<glm::vec4> imageData;
	imageData.resize(resolution.x * resolution.y);
	Jobs::RunParallelFor(imageData.size(), [&](unsigned i)
		{
			imageData[i] = glm::vec4(src[i], 1.0f);
		}
	);
	SetData(imageData, resolution);
}

void Texture2D::SetRgChannelData(const std::vector<glm::vec2>& src, const glm::uvec2& resolution) const
{
	std::vector<glm::vec4> imageData;
	imageData.resize(resolution.x * resolution.y);
	Jobs::RunParallelFor(imageData.size(), [&](unsigned i)
		{
			imageData[i] = glm::vec4(src[i], 0.0f, 1.0f);
		}
	);
	SetData(imageData, resolution);
}

void Texture2D::SetRedChannelData(const std::vector<float>& src, const glm::uvec2& resolution) const
{
	std::vector<glm::vec4> imageData;
	imageData.resize(resolution.x * resolution.y);
	Jobs::RunParallelFor(imageData.size(), [&](unsigned i)
		{
			imageData[i] = glm::vec4(src[i], 0.0f, 0.0f, 1.0f);
		}
	);
	SetData(imageData, resolution);
}
