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
	m_depthImage.reset();

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

	VkImageCreateInfo depthInfo{};
	depthInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	depthInfo.imageType = imageInfo.imageType;
	depthInfo.extent = extent;
	depthInfo.mipLevels = 1;
	depthInfo.arrayLayers = layerCount;
	depthInfo.format = Graphics::Constants::RENDER_TEXTURE_DEPTH;
	depthInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	depthInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	depthInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	depthInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	m_depthImage = std::make_shared<Image>(depthInfo);
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer)
		{
			m_depthImage->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
		});


	VkImageViewCreateInfo depthViewInfo{};
	depthViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	depthViewInfo.image = m_depthImage->GetVkImage();
	depthViewInfo.viewType = imageViewType;
	depthViewInfo.format = Graphics::Constants::RENDER_TEXTURE_DEPTH;
	depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	depthViewInfo.subresourceRange.baseMipLevel = 0;
	depthViewInfo.subresourceRange.levelCount = 1;
	depthViewInfo.subresourceRange.baseArrayLayer = 0;
	depthViewInfo.subresourceRange.layerCount = layerCount;

	m_depthImageView = std::make_shared<ImageView>(depthViewInfo);

	VkSamplerCreateInfo depthSamplerInfo{};
	depthSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	depthSamplerInfo.magFilter = VK_FILTER_LINEAR;
	depthSamplerInfo.minFilter = VK_FILTER_LINEAR;
	depthSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	depthSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	depthSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	depthSamplerInfo.anisotropyEnable = VK_TRUE;
	depthSamplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
	depthSamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	depthSamplerInfo.unnormalizedCoordinates = VK_FALSE;
	depthSamplerInfo.compareEnable = VK_FALSE;
	depthSamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	depthSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	m_depthSampler = std::make_shared<Sampler>(depthSamplerInfo);

	m_extent = extent;
	m_imageViewType = imageViewType;
	m_colorFormat = Graphics::Constants::RENDER_TEXTURE_COLOR;
	m_depthFormat = Graphics::Constants::RENDER_TEXTURE_DEPTH;

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

void RenderTexture::Clear(const VkCommandBuffer commandBuffer)
{
	VkImageSubresourceRange depthSubresourceRange{};
	depthSubresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	depthSubresourceRange.baseMipLevel = 0;
	depthSubresourceRange.levelCount = 1;
	depthSubresourceRange.baseArrayLayer = 0;
	VkClearDepthStencilValue depthStencilValue{};
	depthStencilValue = { 1, 0 };
	vkCmdClearDepthStencilImage(commandBuffer, m_depthImage->GetVkImage(), m_depthImage->GetLayout(), &
		depthStencilValue, 1, &depthSubresourceRange);
	VkImageSubresourceRange colorSubresourceRange{};
	colorSubresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	colorSubresourceRange.baseMipLevel = 0;
	colorSubresourceRange.levelCount = 1;
	colorSubresourceRange.baseArrayLayer = 0;
	VkClearColorValue colorValue{};
	colorValue = { 0, 0, 0, 1 };
	vkCmdClearColorImage(commandBuffer, m_colorImage->GetVkImage(), m_colorImage->GetLayout(), &colorValue, 1, &colorSubresourceRange);
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

VkFormat RenderTexture::GetDepthFormat() const
{
	return m_depthFormat;
}

const std::shared_ptr<Sampler>& RenderTexture::GetColorSampler() const
{
	return m_colorSampler;
}

const std::shared_ptr<Sampler>& RenderTexture::GetDepthSampler() const
{
	return m_depthSampler;
}

const std::shared_ptr<Image>& RenderTexture::GetColorImage()
{
	return m_colorImage;
}

const std::shared_ptr<Image>& RenderTexture::GetDepthImage()
{
	return m_depthImage;
}

const std::shared_ptr<ImageView>& RenderTexture::GetColorImageView()
{
	return m_colorImageView;
}

const std::shared_ptr<ImageView>& RenderTexture::GetDepthImageView()
{
	return m_depthImageView;
}

void RenderTexture::BeginRendering(const VkCommandBuffer commandBuffer, const VkAttachmentLoadOp loadOp, const VkAttachmentStoreOp storeOp) const
{
	m_depthImage->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
	m_colorImage->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
	VkRect2D renderArea;
	renderArea.offset = { 0, 0 };
	renderArea.extent.width = m_extent.width;
	renderArea.extent.height = m_extent.height;
	VkRenderingInfo renderInfo{};
	const auto depthAttachment = GetDepthAttachmentInfo(loadOp, storeOp);
	std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos;
	AppendColorAttachmentInfos(colorAttachmentInfos, loadOp, storeOp);
	renderInfo.pDepthAttachment = &depthAttachment;
	renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
	renderInfo.pColorAttachments = colorAttachmentInfos.data();
	renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
	renderInfo.renderArea = renderArea;
	renderInfo.layerCount = 1;
	vkCmdBeginRendering(commandBuffer, &renderInfo);
}

void RenderTexture::EndRendering(const VkCommandBuffer commandBuffer) const
{
	vkCmdEndRendering(commandBuffer);
	//m_depthImage->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	//m_colorImage->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	//m_stencilImage->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

ImTextureID RenderTexture::GetColorImTextureId() const
{
	return m_colorImTextureId;
}

void RenderTexture::ApplyGraphicsPipelineStates(GraphicsPipelineStates& globalPipelineState) const
{
	VkViewport viewport;
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = m_extent.width;
	viewport.height = m_extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor;
	scissor.offset = { 0, 0 };
	scissor.extent.width = m_extent.width;
	scissor.extent.height = m_extent.height;
	globalPipelineState.m_viewPort = viewport;
	globalPipelineState.m_scissor = scissor;
	globalPipelineState.m_colorBlendAttachmentStates.clear();
	globalPipelineState.m_colorBlendAttachmentStates.resize(1);
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
		depthAttachment.format = Graphics::Constants::RENDER_TEXTURE_DEPTH;
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
	const size_t storeChannels = 4;
	const size_t channels = 4;
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

void RenderTexture::StoreToJpg(const std::string& path, int resizeX, int resizeY, unsigned quality) const
{
	const auto resolutionX = m_colorImage->GetExtent().width;
	const auto resolutionY = m_colorImage->GetExtent().height;
	std::vector<float> dst;
	const size_t storeChannels = 3;
	const size_t channels = 4;
	dst.resize(resolutionX * resolutionY * channels);
	//Retrieve image data here.
	//Retrieve image data here.
	Buffer imageBuffer(sizeof(glm::vec4) * resolutionX * resolutionY);
	imageBuffer.CopyFromImage(*m_colorImage);
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

void RenderTexture::StoreToHdr(const std::string& path, int resizeX, int resizeY,
	unsigned quality) const
{
	const auto resolutionX = m_colorImage->GetExtent().width;
	const auto resolutionY = m_colorImage->GetExtent().height;
	const size_t channels = 4;
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
