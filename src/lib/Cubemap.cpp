#include "Cubemap.hpp"

#include "Application.hpp"
#include "Console.hpp"
#include "Graphics.hpp"
#include "RenderLayer.hpp"
using namespace EvoEngine;

void Cubemap::ConvertFromEquirectangularTexture(const std::shared_ptr<Texture2D>& targetTexture)
{
	if (!targetTexture->m_image) {
		EVOENGINE_ERROR("Target texture doesn't contain any content!");
		return;
	}
	m_sampler.reset();
	m_imageView.reset();
	m_image.reset();

	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_3D;
	imageInfo.extent.width = Graphics::StorageSizes::m_cubemapResolution;
	imageInfo.extent.height = Graphics::StorageSizes::m_cubemapResolution;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 6;
	imageInfo.format = targetTexture->m_hdr ? Graphics::ImageFormats::m_texture2DHDR : Graphics::ImageFormats::m_texture2D;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	m_image = std::make_unique<Image>(imageInfo);

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = m_image->GetVkImage();
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
	viewInfo.format = targetTexture->m_hdr ? Graphics::ImageFormats::m_texture2DHDR : Graphics::ImageFormats::m_texture2D;
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

	std::unique_ptr<DescriptorSet> tempSet = std::make_unique<DescriptorSet>(Graphics::GetDescriptorSetLayout("EQUIRECTANGULAR_TO_CUBE_LAYOUT"));
	VkDescriptorImageInfo descriptorImageInfo{};
	descriptorImageInfo.imageView = targetTexture->GetVkImageView();
	descriptorImageInfo.imageLayout = targetTexture->GetLayout();
	descriptorImageInfo.sampler = targetTexture->GetVkSampler();
	
	tempSet->UpdateImageDescriptorBinding(0, descriptorImageInfo);



}
