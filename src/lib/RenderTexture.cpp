#include "RenderTexture.hpp"

#include "Console.hpp"
#include "EditorLayer.hpp"
#include "Graphics.hpp"

using namespace EvoEngine;

void RenderTexture::Initialize(VkExtent3D extent, VkImageViewType imageViewType)
{
	m_colorImageView.reset();
	m_colorImage.reset();

	m_depthImageView.reset();
	m_stencilImageView.reset();
	m_depthStencilImage.reset();

	int layerCount = imageViewType == VK_IMAGE_VIEW_TYPE_CUBE ? 6 : 1;

	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	switch (imageViewType)
	{
	case VK_IMAGE_VIEW_TYPE_1D:
		imageInfo.imageType = VK_IMAGE_TYPE_1D;
		break;
	case VK_IMAGE_VIEW_TYPE_2D:
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		break;
	case VK_IMAGE_VIEW_TYPE_3D:
		imageInfo.imageType = VK_IMAGE_TYPE_3D;
		break;
	case VK_IMAGE_VIEW_TYPE_CUBE:
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		break;
	case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
		imageInfo.imageType = VK_IMAGE_TYPE_1D;
		break;
	case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		break;
	case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		break;
	case VK_IMAGE_VIEW_TYPE_MAX_ENUM:
		EVOENGINE_ERROR("Wrong imageViewType!");
		break;
	}

	imageInfo.extent = extent;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = layerCount;
	imageInfo.format = Graphics::Constants::RENDER_TEXTURE_COLOR;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	m_colorImage = std::make_shared<Image>(imageInfo);
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer)
		{
			m_colorImage->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
		});



	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = m_colorImage->GetVkImage();
	viewInfo.viewType = imageViewType;
	viewInfo.format = Graphics::Constants::RENDER_TEXTURE_COLOR;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = layerCount;

	m_colorImageView = std::make_shared<ImageView>(viewInfo);

	VkImageCreateInfo depthStencilInfo{};
	depthStencilInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	depthStencilInfo.imageType = imageInfo.imageType;
	depthStencilInfo.extent = extent;
	depthStencilInfo.mipLevels = 1;
	depthStencilInfo.arrayLayers = layerCount;
	depthStencilInfo.format = Graphics::Constants::RENDER_TEXTURE_DEPTH_STENCIL;
	depthStencilInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	depthStencilInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthStencilInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	depthStencilInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	depthStencilInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	m_depthStencilImage = std::make_shared<Image>(depthStencilInfo);
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer)
		{
			m_depthStencilImage->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
		});


	VkImageViewCreateInfo depthViewInfo{};
	depthViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	depthViewInfo.image = m_depthStencilImage->GetVkImage();
	depthViewInfo.viewType = imageViewType;
	depthViewInfo.format = Graphics::Constants::RENDER_TEXTURE_DEPTH_STENCIL;
	depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	depthViewInfo.subresourceRange.baseMipLevel = 0;
	depthViewInfo.subresourceRange.levelCount = 1;
	depthViewInfo.subresourceRange.baseArrayLayer = 0;
	depthViewInfo.subresourceRange.layerCount = layerCount;

	m_depthImageView = std::make_shared<ImageView>(depthViewInfo);

	VkImageViewCreateInfo stencilViewInfo{};
	stencilViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	stencilViewInfo.image = m_depthStencilImage->GetVkImage();
	stencilViewInfo.viewType = imageViewType;
	stencilViewInfo.format = Graphics::Constants::RENDER_TEXTURE_DEPTH_STENCIL;
	stencilViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
	stencilViewInfo.subresourceRange.baseMipLevel = 0;
	stencilViewInfo.subresourceRange.levelCount = 1;
	stencilViewInfo.subresourceRange.baseArrayLayer = 0;
	stencilViewInfo.subresourceRange.layerCount = layerCount;

	m_stencilImageView = std::make_shared<ImageView>(stencilViewInfo);



	m_extent = extent;
	m_imageViewType = imageViewType;
	m_colorFormat = Graphics::Constants::RENDER_TEXTURE_COLOR;
	m_depthStencilFormat = Graphics::Constants::RENDER_TEXTURE_DEPTH_STENCIL;

	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	samplerInfo.anisotropyEnable = VK_TRUE;
	samplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
	samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	m_colorSampler = std::make_shared<Sampler>(samplerInfo);

	EditorLayer::UpdateTextureId(m_colorImTextureId, m_colorSampler->GetVkSampler(), m_colorImageView->GetVkImageView(), m_colorImage->GetLayout());

	m_descriptorSet = std::make_shared<DescriptorSet>(Graphics::GetDescriptorSetLayout("RENDER_TEXTURE_PRESENT"));
	VkDescriptorImageInfo descriptorImageInfo;
	descriptorImageInfo.imageLayout = m_colorImage->GetLayout();
	descriptorImageInfo.imageView = m_colorImageView->GetVkImageView();
	descriptorImageInfo.sampler = m_colorSampler->GetVkSampler();
	m_descriptorSet->UpdateImageDescriptorBinding(0, descriptorImageInfo);
}

RenderTexture::RenderTexture(const VkExtent3D extent, const VkImageViewType imageViewType)
{
	Initialize(extent, imageViewType);
}
void RenderTexture::Resize(const VkExtent3D extent)
{
	Initialize(extent, m_imageViewType);
}

void RenderTexture::AppendColorAttachmentInfos(std::vector<VkRenderingAttachmentInfo>& attachmentInfos, const VkAttachmentLoadOp loadOp, const VkAttachmentStoreOp storeOp) const
{
	VkRenderingAttachmentInfo attachment{};
	attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

	attachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
	attachment.loadOp = loadOp;
	attachment.storeOp = storeOp;

	attachment.clearValue.color = { 0, 0, 0, 0 };
	attachment.imageView = m_colorImageView->GetVkImageView();
	attachmentInfos.push_back(attachment);
}

VkRenderingAttachmentInfo RenderTexture::GetDepthAttachmentInfo(const VkAttachmentLoadOp loadOp, const VkAttachmentStoreOp storeOp) const
{
	VkRenderingAttachmentInfo attachment{};
	attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

	attachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
	attachment.loadOp = loadOp;
	attachment.storeOp = storeOp;

	attachment.clearValue.depthStencil = { 1, 0 };
	attachment.imageView = m_depthImageView->GetVkImageView();
	return attachment;
}

VkRenderingAttachmentInfo RenderTexture::GetStencilAttachmentInfo(const VkAttachmentLoadOp loadOp, const VkAttachmentStoreOp storeOp) const
{
	VkRenderingAttachmentInfo attachment{};
	attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

	attachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
	attachment.loadOp = loadOp;
	attachment.storeOp = storeOp;

	attachment.clearValue.depthStencil = { 1, 0 };
	attachment.imageView = m_stencilImageView->GetVkImageView();
	return attachment;
}

VkExtent3D RenderTexture::GetExtent() const
{
	return m_extent;
}

VkImageViewType RenderTexture::GetImageViewType() const
{
	return m_imageViewType;
}

VkFormat RenderTexture::GetColorFormat() const
{
	return m_colorFormat;
}

VkFormat RenderTexture::GetDepthStencilFormat() const
{
	return m_depthStencilFormat;
}

const std::shared_ptr<Sampler>& RenderTexture::GetSampler() const
{
	return m_colorSampler;
}

const std::shared_ptr<Image>& RenderTexture::GetColorImage()
{
	return m_colorImage;
}

const std::shared_ptr<Image>& RenderTexture::GetDepthStencilImage()
{
	return m_depthStencilImage;
}

const std::shared_ptr<ImageView>& RenderTexture::GetColorImageView()
{
	return m_colorImageView;
}

const std::shared_ptr<ImageView>& RenderTexture::GetDepthImageView()
{
	return m_depthImageView;
}

ImTextureID RenderTexture::GetColorImTextureId() const
{
	return m_colorImTextureId;
}

const std::vector<VkAttachmentDescription>& RenderTexture::GetAttachmentDescriptions()
{
	static std::vector<VkAttachmentDescription> attachments{};
	if (attachments.empty()) {
		static VkAttachmentDescription colorAttachment{};
		colorAttachment.format = Graphics::Constants::RENDER_TEXTURE_COLOR;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;

		static VkAttachmentDescription depthAttachment{};
		depthAttachment.format = Graphics::Constants::RENDER_TEXTURE_DEPTH_STENCIL;
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
		attachments = { depthAttachment, colorAttachment };
	}
	return attachments;
}

bool RenderTexture::Save(const std::filesystem::path& path) const
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

void RenderTexture::StoreToPng(const std::string& path, int resizeX, int resizeY, unsigned compressionLevel) const
{
	stbi_write_png_compression_level = static_cast<int>(compressionLevel);
	const auto resolutionX = m_colorImage->GetExtent().width;
	const auto resolutionY = m_colorImage->GetExtent().height;
	size_t channels = 4;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * channels);
	//Retrieve image data here.
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*m_colorImage);
	imageBuffer.DownloadVector(dst, resolutionX * resolutionY * channels);
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

void RenderTexture::StoreToJpg(const std::string& path, int resizeX, int resizeY, unsigned quality) const
{
	const auto resolutionX = m_colorImage->GetExtent().width;
	const auto resolutionY = m_colorImage->GetExtent().height;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * 4);
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

void RenderTexture::StoreToHdr(const std::string& path, int resizeX, int resizeY,
	unsigned quality) const
{
	const auto resolutionX = m_colorImage->GetExtent().width;
	const auto resolutionY = m_colorImage->GetExtent().height;

	size_t channels = 4;
	std::vector<float> dst;
	dst.resize(resolutionX * resolutionY * channels);
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*m_colorImage);
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
