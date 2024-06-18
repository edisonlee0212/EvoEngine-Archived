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
		StoreToPng(path);
	}
	else if (path.extension() == ".jpg") {
		StoreToJpg(path);
	}
	else if (path.extension() == ".tga") {
		StoreToTga(path);
	}
	else if (path.extension() == ".hdr") {
		StoreToHdr(path);
	}
	else if (path.extension() == ".evetexture2d")
	{
		auto directory = path;
		directory.remove_filename();
		std::filesystem::create_directories(directory);
		YAML::Emitter out;
		out << YAML::BeginMap;
		Serialize(out);
		std::ofstream outStream(path.string());
		outStream << out.c_str();
		outStream.flush();
		return true;
	}
	else {
		EVOENGINE_ERROR("Not implemented!");
		return false;
	}
	return true;
}

bool Texture2D::LoadInternal(const std::filesystem::path& path)
{
	if (path.extension() == ".evetexture2d")
	{
		std::ifstream stream(path.string());
		std::stringstream stringStream;
		stringStream << stream.rdbuf();
		YAML::Node in = YAML::Load(stringStream.str());
		Deserialize(in);
		return true;
	}
	m_hdr = false;
	if (path.extension() == ".hdr") m_hdr = true;
	stbi_set_flip_vertically_on_load(true);
	int width, height, nrComponents;

	float actualGamma = m_hdr ? 2.2f : 1.f;

	stbi_hdr_to_ldr_gamma(actualGamma);
	stbi_ldr_to_hdr_gamma(actualGamma);

	void* data = stbi_loadf(path.string().c_str(), &width, &height, &nrComponents, STBI_rgb_alpha);

	if (nrComponents == 1)
	{
		m_redChannel = true;
		m_greenChannel = false;
		m_blueChannel = false;
		m_alphaChannel = false;
	}
	else if (nrComponents == 2)
	{
		m_redChannel = true;
		m_greenChannel = true;
		m_blueChannel = false;
		m_alphaChannel = false;
	}
	else if (nrComponents == 3)
	{
		m_redChannel = true;
		m_greenChannel = true;
		m_blueChannel = true;
		m_alphaChannel = false;
	}
	else if (nrComponents == 4)
	{
		m_redChannel = true;
		m_greenChannel = true;
		m_blueChannel = true;
		m_alphaChannel = true;
	}

	if (data)
	{
		std::vector<glm::vec4> imageData;
		imageData.resize(width * height);
		memcpy(imageData.data(), data, sizeof(glm::vec4) * width * height);
		auto& textureStorage = TextureStorage::RefTexture2DStorage(m_textureStorageHandle);
		textureStorage.SetDataImmediately(imageData, { width, height });
	}
	else
	{
		EVOENGINE_ERROR("Texture failed to load at path: " + path.filename().string());
		return false;
	}
	stbi_image_free(data);
	return true;
}



void Texture2D::ApplyOpacityMap(const std::shared_ptr<Texture2D>& target)
{
	std::vector<glm::vec4> colorData;
	if (!target) return;
	GetRgbaChannelData(colorData);
	std::vector<glm::vec4> alphaData;
	const auto resolution = GetResolution();
	target->GetRgbaChannelData(alphaData, resolution.x, resolution.y);
	Jobs::RunParallelFor(colorData.size(), [&](unsigned i)
		{
			colorData[i].a = alphaData[i].r;
		}
	);
	SetRgbaChannelData(colorData, target->GetResolution());
	m_alphaChannel = true;
	SetUnsaved();
}

void Texture2D::Serialize(YAML::Emitter& out) const
{
	std::vector<glm::vec4> pixels;
	GetRgbaChannelData(pixels);
	out << YAML::Key << "m_hdr" << YAML::Value << m_hdr;

	out << YAML::Key << "m_redChannel" << YAML::Value << m_redChannel;
	out << YAML::Key << "m_greenChannel" << YAML::Value << m_greenChannel;
	out << YAML::Key << "m_blueChannel" << YAML::Value << m_blueChannel;
	out << YAML::Key << "m_alphaChannel" << YAML::Value << m_alphaChannel;
	auto resolution = GetResolution();
	out << YAML::Key << "m_resolution" << YAML::Value << resolution;
	if (resolution.x != 0 && resolution.y != 0) {
		if (m_hdr) {
			Serialization::SerializeVector("m_pixels", pixels, out);
		}
		else
		{
			std::vector<unsigned char> transferredPixels;
			size_t targetChannelSize = 0;
			if (m_redChannel) targetChannelSize++;
			if (m_greenChannel) targetChannelSize++;
			if (m_blueChannel) targetChannelSize++;
			if (m_alphaChannel) targetChannelSize++;

			transferredPixels.resize(resolution.x * resolution.y * targetChannelSize);
			Jobs::RunParallelFor(resolution.x * resolution.y, [&](unsigned i)
				{
					for (int channel = 0; channel < targetChannelSize; channel++)
					{
						transferredPixels[i * targetChannelSize + channel] = static_cast<unsigned char>(glm::clamp(pixels[i][channel] * 255.9f, 0.f, 255.f));

					}
				});
			Serialization::SerializeVector("m_pixels", transferredPixels, out);
		}
	}
}

void Texture2D::Deserialize(const YAML::Node& in)
{
	std::vector<glm::vec4> pixels;
	glm::ivec2 resolution = glm::ivec2(0);

	if (in["m_redChannel"]) m_redChannel = in["m_redChannel"].as<bool>();
	if (in["m_greenChannel"]) m_greenChannel = in["m_greenChannel"].as<bool>();
	if (in["m_blueChannel"]) m_blueChannel = in["m_blueChannel"].as<bool>();
	if (in["m_alphaChannel"]) m_alphaChannel = in["m_alphaChannel"].as<bool>();

	if (in["m_hdr"]) m_hdr = in["m_hdr"].as<bool>();


	if (in["m_resolution"]) resolution = in["m_resolution"].as<glm::ivec2>();

	if (resolution.x != 0 && resolution.y != 0) {
		if (m_hdr) {
			Serialization::DeserializeVector("m_pixels", pixels, in);
			SetRgbaChannelData(pixels, resolution);
		}
		else
		{
			size_t targetChannelSize = 0;
			if (m_redChannel) targetChannelSize++;
			if (m_greenChannel) targetChannelSize++;
			if (m_blueChannel) targetChannelSize++;
			if (m_alphaChannel) targetChannelSize++;

			std::vector<unsigned char> transferredPixels;

			Serialization::DeserializeVector("m_pixels", transferredPixels, in);
			transferredPixels.resize(resolution.x * resolution.y * targetChannelSize);
			pixels.resize(resolution.x * resolution.y);

			Jobs::RunParallelFor(pixels.size(), [&](unsigned i)
				{
					for (int channel = 0; channel < targetChannelSize; channel++)
					{
						pixels[i][channel] = glm::clamp(transferredPixels[i * targetChannelSize + channel] / 256.f, 0.f, 1.f);
					}
					if (targetChannelSize < 4)
					{
						pixels[i][3] = 1.f;
					}
					if (targetChannelSize < 3)
					{
						pixels[i][2] = 0.f;
					}
					if (targetChannelSize < 2)
					{
						pixels[i][1] = 0.f;
					}
				});

			SetRgbaChannelData(pixels, resolution);
		}
	}
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
	ImGui::Text((std::string("Red Channel: ") + (m_redChannel ? "True" : "False")).c_str());
	ImGui::Text((std::string("Green Channel: ") + (m_greenChannel ? "True" : "False")).c_str());
	ImGui::Text((std::string("Blue Channel: ") + (m_blueChannel ? "True" : "False")).c_str());
	ImGui::Text((std::string("Alpha Channel: ") + (m_alphaChannel ? "True" : "False")).c_str());

	const auto textureStorage = PeekTexture2DStorage();
	static AssetRef temp;
	if (EditorLayer::DragAndDropButton<Texture2D>(temp, "Apply Opacity..."))
	{
		if (const auto tex = temp.Get<Texture2D>())
		{
			ApplyOpacityMap(tex);
			temp.Clear();
		}
	}
	if (textureStorage.m_imTextureId) {
		static float debugScale = 0.25f;
		ImGui::DragFloat("Scale", &debugScale, 0.01f, 0.1f, 10.0f);
		debugScale = glm::clamp(debugScale, 0.1f, 1.0f);
		ImGui::Image(textureStorage.m_imTextureId,
			ImVec2(textureStorage.m_image->GetExtent().width * debugScale, textureStorage.m_image->GetExtent().height * debugScale),
			ImVec2(0, 1),
			ImVec2(1, 0));
	}

	return changed;
}

glm::ivec2 Texture2D::GetResolution() const
{
	const auto textureStorage = PeekTexture2DStorage();
	return { textureStorage.m_image->GetExtent().width, textureStorage.m_image->GetExtent().height };
}

void Texture2D::StoreToPng(const std::filesystem::path& path, const int resizeX, const int resizeY,
	const unsigned compressionLevel) const
{
	const auto& textureStorage = PeekTexture2DStorage();

	const auto resolutionX = textureStorage.m_image->GetExtent().width;
	const auto resolutionY = textureStorage.m_image->GetExtent().height;

	size_t targetChannelSize = 0;
	if (m_redChannel) targetChannelSize++;
	if (m_greenChannel) targetChannelSize++;
	if (m_blueChannel) targetChannelSize++;
	if (m_alphaChannel) targetChannelSize++;

	constexpr size_t deviceChannels = 4;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * deviceChannels);
	//Retrieve image data here.
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*textureStorage.m_image);
	imageBuffer.DownloadVector(dst, resolutionX * resolutionY * deviceChannels);

	StoreToPng(path, dst, resolutionX, resolutionY, 4, targetChannelSize, compressionLevel, resizeX, resizeY);
}
void Texture2D::StoreToPng(const std::filesystem::path& path, const std::vector<float>& srcData, const int srcX, const int srcY,
	const int srcChannelSize, const int targetChannelSize, const unsigned compressionLevel, const int resizeX, const int resizeY)
{
	stbi_write_png_compression_level = static_cast<int>(compressionLevel);
	stbi_flip_vertically_on_write(true);
	std::vector<uint8_t> pixels;
	if (resizeX > 0 && resizeY > 0 && (resizeX != srcX || resizeY != srcY))
	{
		std::vector<float> res;
		res.resize(resizeX * resizeY * srcChannelSize);
		stbir_resize_float(srcData.data(), srcX, srcY, 0, res.data(), resizeX, resizeY, 0, srcChannelSize);

		pixels.resize(resizeX * resizeY * targetChannelSize);
		Jobs::RunParallelFor(resizeX * resizeY, [&](unsigned i)
			{
				for (int targetChannelIndex = 0; targetChannelIndex < targetChannelSize; targetChannelIndex++)
				{
					pixels[i * targetChannelSize + targetChannelIndex] = glm::clamp(static_cast<int>(255.9f * res[i * srcChannelSize + targetChannelIndex]), 0, 255);
				}
			});
		stbi_write_png(path.string().c_str(), resizeX, resizeY, targetChannelSize, pixels.data(), resizeX * targetChannelSize);
	}
	else
	{
		pixels.resize(srcX * srcY * targetChannelSize);
		Jobs::RunParallelFor(srcX * srcY, [&](unsigned i)
			{
				for (int targetChannelIndex = 0; targetChannelIndex < targetChannelSize; targetChannelIndex++)
				{
					pixels[i * targetChannelSize + targetChannelIndex] = glm::clamp(static_cast<int>(255.9f * srcData[i * srcChannelSize + targetChannelIndex]), 0, 255);
				}
			});
		stbi_write_png(path.string().c_str(), srcX, srcY, targetChannelSize, pixels.data(), srcX * targetChannelSize);
	}
}


void Texture2D::StoreToTga(const std::filesystem::path& path, const int resizeX, const int resizeY) const
{
	const auto& textureStorage = PeekTexture2DStorage();

	const auto resolutionX = textureStorage.m_image->GetExtent().width;
	const auto resolutionY = textureStorage.m_image->GetExtent().height;

	size_t targetChannelSize = 0;
	if (m_redChannel) targetChannelSize++;
	if (m_greenChannel) targetChannelSize++;
	if (m_blueChannel) targetChannelSize++;
	if (m_alphaChannel) targetChannelSize++;

	constexpr size_t deviceChannels = 4;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * deviceChannels);
	//Retrieve image data here.
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*textureStorage.m_image);
	imageBuffer.DownloadVector(dst, resolutionX * resolutionY * deviceChannels);

	StoreToTga(path, dst, resolutionX, resolutionY, 4, targetChannelSize, resizeX, resizeY);
}

void Texture2D::StoreToJpg(const std::filesystem::path& path, const int resizeX, const int resizeY, const unsigned quality) const
{
	const auto& textureStorage = PeekTexture2DStorage();

	const auto resolutionX = textureStorage.m_image->GetExtent().width;
	const auto resolutionY = textureStorage.m_image->GetExtent().height;

	size_t targetChannelSize = 0;
	if (m_redChannel) targetChannelSize++;
	if (m_greenChannel) targetChannelSize++;
	if (m_blueChannel) targetChannelSize++;
	if (m_alphaChannel) targetChannelSize++;

	constexpr size_t deviceChannels = 4;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * deviceChannels);
	//Retrieve image data here.
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*textureStorage.m_image);
	imageBuffer.DownloadVector(dst, resolutionX * resolutionY * deviceChannels);

	StoreToJpg(path, dst, resolutionX, resolutionY, 4, targetChannelSize, quality, resizeX, resizeY);
}

void Texture2D::StoreToJpg(const std::filesystem::path& path, const std::vector<float>& srcData, const int srcX, const int srcY,
	const int srcChannelSize, int targetChannelSize, const unsigned quality, const int resizeX, const int resizeY)
{
	stbi_flip_vertically_on_write(true);

	targetChannelSize = glm::max(targetChannelSize, 3);
	std::vector<uint8_t> pixels;
	if (resizeX > 0 && resizeY > 0 && (resizeX != srcX || resizeY != srcY))
	{
		std::vector<float> res;
		res.resize(resizeX * resizeY * srcChannelSize);
		stbir_resize_float(srcData.data(), srcX, srcY, 0, res.data(), resizeX, resizeY, 0, srcChannelSize);

		pixels.resize(resizeX * resizeY * targetChannelSize);

		Jobs::RunParallelFor(resizeX * resizeY, [&](unsigned i)
			{
				for (int targetChannelIndex = 0; targetChannelIndex < targetChannelSize; targetChannelIndex++)
				{
					pixels[i * targetChannelSize + targetChannelIndex] = glm::clamp(static_cast<int>(255.9f * res[i * srcChannelSize + targetChannelIndex]), 0, 255);
				}
			});


		stbi_write_jpg(path.string().c_str(), resizeX, resizeY, targetChannelSize, pixels.data(), quality);
	}
	else
	{
		pixels.resize(srcX * srcY * targetChannelSize);
		Jobs::RunParallelFor(srcX * srcY, [&](unsigned i)
			{
				for (int targetChannelIndex = 0; targetChannelIndex < targetChannelSize; targetChannelIndex++)
				{
					pixels[i * targetChannelSize + targetChannelIndex] = glm::clamp(static_cast<int>(255.9f * srcData[i * srcChannelSize + targetChannelIndex]), 0, 255);
				}
			});
		stbi_write_jpg(path.string().c_str(), srcX, srcY, targetChannelSize, pixels.data(), quality);
	}
}

void Texture2D::StoreToTga(const std::filesystem::path& path, const std::vector<float>& srcData, const int srcX, const int srcY,
	const int srcChannelSize, const int targetChannelSize, const int resizeX, const int resizeY)
{
	stbi_flip_vertically_on_write(true);

	std::vector<uint8_t> pixels;
	if (resizeX > 0 && resizeY > 0 && (resizeX != srcX || resizeY != srcY))
	{
		std::vector<float> res;
		res.resize(resizeX * resizeY * srcChannelSize);
		stbir_resize_float(srcData.data(), srcX, srcY, 0, res.data(), resizeX, resizeY, 0, srcChannelSize);

		pixels.resize(resizeX * resizeY * targetChannelSize);
		Jobs::RunParallelFor(resizeX * resizeY, [&](unsigned i)
			{
				for (int targetChannelIndex = 0; targetChannelIndex < targetChannelSize; targetChannelIndex++)
				{
					pixels[i * targetChannelSize + targetChannelIndex] = glm::clamp(static_cast<int>(255.9f * res[i * srcChannelSize + targetChannelIndex]), 0, 255);
				}
			});

		stbi_write_tga(path.string().c_str(), resizeX, resizeY, targetChannelSize, pixels.data());
	}
	else
	{
		pixels.resize(srcX * srcY * targetChannelSize);
		Jobs::RunParallelFor(srcX * srcY, [&](unsigned i)
			{
				for (int targetChannelIndex = 0; targetChannelIndex < targetChannelSize; targetChannelIndex++)
				{
					pixels[i * targetChannelSize + targetChannelIndex] = glm::clamp(static_cast<int>(255.9f * srcData[i * srcChannelSize + targetChannelIndex]), 0, 255);
				}
			});
		stbi_write_tga(path.string().c_str(), srcX, srcY, targetChannelSize, pixels.data());
	}
}

void Texture2D::StoreToHdr(const std::filesystem::path& path, const std::vector<float>& srcData, const int srcX, const int srcY,
	const int srcChannelSize, const int targetChannelSize, const int resizeX, const int resizeY)
{
	std::vector<float> pixels;
	stbi_flip_vertically_on_write(true);
	if (resizeX > 0 && resizeY > 0 && (resizeX != srcX || resizeY != srcY))
	{
		std::vector<float> res;
		res.resize(resizeX * resizeY * srcChannelSize);
		stbir_resize_float(srcData.data(), srcX, srcY, 0, res.data(), resizeX, resizeY, 0, srcChannelSize);

		pixels.resize(resizeX * resizeY * targetChannelSize);
		Jobs::RunParallelFor(resizeX * resizeY, [&](unsigned i)
			{
				for (int targetChannelIndex = 0; targetChannelIndex < targetChannelSize; targetChannelIndex++)
				{
					pixels[i * targetChannelSize + targetChannelIndex] = res[i * srcChannelSize + targetChannelIndex];
				}
			});

		stbi_write_hdr(path.string().c_str(), resizeX, resizeY, targetChannelSize, pixels.data());
	}
	else
	{
		pixels.resize(srcX * srcY * targetChannelSize);
		Jobs::RunParallelFor(srcX * srcY, [&](unsigned i)
			{
				for (int targetChannelIndex = 0; targetChannelIndex < targetChannelSize; targetChannelIndex++)
				{
					pixels[i * targetChannelSize + targetChannelIndex] = srcData[i * srcChannelSize + targetChannelIndex];
				}
			});
		stbi_write_hdr(path.string().c_str(), srcX, srcY, targetChannelSize, pixels.data());
	}
}


void Texture2D::StoreToHdr(const std::filesystem::path& path, const int resizeX, const int resizeY) const
{
	const auto& textureStorage = PeekTexture2DStorage();

	const auto resolutionX = textureStorage.m_image->GetExtent().width;
	const auto resolutionY = textureStorage.m_image->GetExtent().height;

	size_t targetChannelSize = 0;
	if (m_redChannel) targetChannelSize++;
	if (m_greenChannel) targetChannelSize++;
	if (m_blueChannel) targetChannelSize++;
	if (m_alphaChannel) targetChannelSize++;

	constexpr size_t deviceChannels = 4;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * deviceChannels);
	//Retrieve image data here.
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*textureStorage.m_image);
	imageBuffer.DownloadVector(dst, resolutionX * resolutionY * deviceChannels);

	StoreToHdr(path, dst, resolutionX, resolutionY, 4, targetChannelSize, resizeX, resizeY);
}

ImTextureID Texture2D::GetImTextureId() const
{
	const auto& textureStorage = PeekTexture2DStorage();
	return textureStorage.m_imTextureId;
}

VkImageLayout Texture2D::GetLayout() const
{
	const auto& textureStorage = PeekTexture2DStorage();
	return textureStorage.m_image->GetLayout();
}

VkImage Texture2D::GetVkImage() const
{
	const auto& textureStorage = PeekTexture2DStorage();
	if (textureStorage.m_image)
	{
		return textureStorage.m_image->GetVkImage();
	}
	return VK_NULL_HANDLE;
}

VkImageView Texture2D::GetVkImageView() const
{
	const auto& textureStorage = PeekTexture2DStorage();
	if (textureStorage.m_imageView)
	{
		return textureStorage.m_imageView->GetVkImageView();
	}
	return VK_NULL_HANDLE;
}

VkSampler Texture2D::GetVkSampler() const
{
	const auto& textureStorage = PeekTexture2DStorage();
	if (textureStorage.m_sampler)
	{
		return textureStorage.m_sampler->GetVkSampler();
	}
	return VK_NULL_HANDLE;
}

std::shared_ptr<Image> Texture2D::GetImage() const
{
	const auto& textureStorage = PeekTexture2DStorage();
	return textureStorage.m_image;
}

void Texture2D::GetRgbaChannelData(std::vector<glm::vec4>& dst, const int resizeX, const int resizeY) const
{
	const auto& textureStorage = PeekTexture2DStorage();
	const auto resolutionX = textureStorage.m_image->GetExtent().width;
	const auto resolutionY = textureStorage.m_image->GetExtent().height;
	if ((resizeX == -1 && resizeY == -1) || (resolutionX == resizeX && resolutionY == resizeY))
	{
		Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
		imageBuffer.CopyFromImage(*textureStorage.m_image);
		imageBuffer.DownloadVector(dst, resolutionX * resolutionY);
		return;
	}
	std::vector<glm::vec4> src;
	src.resize(resolutionX * resolutionY);
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*textureStorage.m_image);
	imageBuffer.DownloadVector(src, resolutionX * resolutionY);

	dst.resize(resizeX * resizeY);
	stbir_resize_float(reinterpret_cast<float*>(src.data()), resolutionX, resolutionY, 0, reinterpret_cast<float*>(dst.data()), resizeX, resizeY, 0, 4);
}

void Texture2D::GetRgbChannelData(std::vector<glm::vec3>& dst, int resizeX, int resizeY) const
{
	const auto& textureStorage = PeekTexture2DStorage();
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
	const auto& textureStorage = PeekTexture2DStorage();
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
	const auto& textureStorage = PeekTexture2DStorage();
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

void Texture2D::SetRgbaChannelData(const std::vector<glm::vec4>& src, const glm::uvec2& resolution)
{
	SetData(src, resolution);
	m_redChannel = true;
	m_greenChannel = true;
	m_blueChannel = true;
	m_alphaChannel = true;
	SetUnsaved();
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
	SetData(imageData, resolution);
	m_redChannel = true;
	m_greenChannel = true;
	m_blueChannel = true;
	m_alphaChannel = false;

	SetUnsaved();
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
	SetData(imageData, resolution);
	m_redChannel = true;
	m_greenChannel = true;
	m_blueChannel = false;
	m_alphaChannel = false;


	SetUnsaved();
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
	SetData(imageData, resolution);
	m_redChannel = true;
	m_greenChannel = false;
	m_blueChannel = false;
	m_alphaChannel = false;

	SetUnsaved();
}
