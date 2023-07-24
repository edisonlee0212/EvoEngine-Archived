#include "RenderTexture.hpp"

#include "Console.hpp"
#include "EditorLayer.hpp"
#include "Graphics.hpp"

using namespace EvoEngine;

void RenderTexture::Initialize(VkExtent3D extent, VkImageViewType imageViewType, VkFormat colorFormat,
	VkFormat depthStencilFormat)
{
	m_colorImage.reset();
	m_colorImageView.reset();
	m_depthStencilImage.reset();
	m_depthStencilImageView.reset();

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
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		break;
	case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
		imageInfo.imageType = VK_IMAGE_TYPE_3D;
		break;
	case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
		imageInfo.imageType = VK_IMAGE_TYPE_3D;
		break;
	case VK_IMAGE_VIEW_TYPE_MAX_ENUM:
		EVOENGINE_ERROR("Wrong imageViewType!");
		break;
	}

	imageInfo.extent = extent;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.format = colorFormat;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	m_colorImage = std::make_unique<Image>(imageInfo);

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = m_colorImage->GetVkImage();
	viewInfo.viewType = imageViewType;
	viewInfo.format = colorFormat;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	m_colorImageView = std::make_unique<ImageView>(viewInfo);

	VkImageCreateInfo depthStencilInfo{};
	depthStencilInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	depthStencilInfo.imageType = imageInfo.imageType;
	depthStencilInfo.extent = extent;
	depthStencilInfo.mipLevels = 1;
	depthStencilInfo.arrayLayers = 1;
	depthStencilInfo.format = depthStencilFormat;
	depthStencilInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	depthStencilInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthStencilInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	depthStencilInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	depthStencilInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	m_depthStencilImage = std::make_unique<Image>(depthStencilInfo);

	VkImageViewCreateInfo depthStencilViewInfo{};
	depthStencilViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	depthStencilViewInfo.image = m_depthStencilImage->GetVkImage();
	depthStencilViewInfo.viewType = imageViewType;
	depthStencilViewInfo.format = depthStencilFormat;
	depthStencilViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
	depthStencilViewInfo.subresourceRange.baseMipLevel = 0;
	depthStencilViewInfo.subresourceRange.levelCount = 1;
	depthStencilViewInfo.subresourceRange.baseArrayLayer = 0;
	depthStencilViewInfo.subresourceRange.layerCount = 1;

	m_depthStencilImageView = std::make_unique<ImageView>(viewInfo);

	m_extent = extent;
	m_imageViewType = imageViewType;
	m_colorFormat = colorFormat;
	m_depthStencilFormat = depthStencilFormat;

	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_TRUE;
	samplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
	samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	m_colorSampler = std::make_unique<Sampler>(samplerInfo);

	if (const auto editorLayer = Application::GetLayer<EditorLayer>()) {
		m_colorImTextureId = editorLayer->GetTextureId(m_colorImageView->GetVkImageView());
		m_depthStencilImTextureId = editorLayer->GetTextureId(m_depthStencilImageView->GetVkImageView());
	}

}

RenderTexture::RenderTexture(const VkExtent3D extent, const VkImageViewType imageViewType, const VkFormat colorFormat,
	const VkFormat depthStencilFormat)
{
	Initialize(extent, imageViewType, colorFormat, depthStencilFormat);
}
void RenderTexture::Resize(const VkExtent3D extent)
{
	Initialize(extent, m_imageViewType, m_colorFormat, m_depthStencilFormat);
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

const std::unique_ptr<Sampler>& RenderTexture::GetSampler() const
{
	return m_colorSampler;
}

const std::unique_ptr<Image>& RenderTexture::GetColorImage()
{
	return m_colorImage;
}

const std::unique_ptr<Image>& RenderTexture::GetDepthStencilImage()
{
	return m_depthStencilImage;
}

const std::unique_ptr<ImageView>& RenderTexture::GetColorImageView()
{
	return m_colorImageView;
}

const std::unique_ptr<ImageView>& RenderTexture::GetDepthStencilImageView()
{
	return m_depthStencilImageView;
}

ImTextureID RenderTexture::GetColorImTextureId() const
{
	return m_colorImTextureId;
}

ImTextureID RenderTexture::GetDepthStencilImTextureId() const
{
	return m_depthStencilImTextureId;
}

const std::vector<VkAttachmentDescription>& RenderTexture::GetAttachmentDescriptions()
{
	static std::vector<VkAttachmentDescription> attachments{};
	if (attachments.empty()) {
		static VkAttachmentDescription colorAttachment{};
		colorAttachment.format = Graphics::GetSwapchain()->GetImageFormat();
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		static VkAttachmentDescription depthAttachment{};
		depthAttachment.format = VK_FORMAT_D24_UNORM_S8_UINT;
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments = { colorAttachment, depthAttachment };
	}
	return attachments;
}
