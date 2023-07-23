#pragma once
#include "GraphicsResources.hpp"

namespace EvoEngine{
	class RenderTexture
	{
		std::unique_ptr<Image> m_colorImage = {};
		std::unique_ptr<ImageView> m_colorImageView = {};

		std::unique_ptr<Image> m_depthStencilImage = {};
		std::unique_ptr<ImageView> m_depthStencilImageView = {};

		VkExtent3D m_extent;
		VkImageViewType m_imageViewType;
		VkFormat m_colorFormat;
		VkFormat m_depthStencilFormat;

		std::unique_ptr<Sampler> m_sampler = {};
		VkDescriptorSet m_colorImTextureId = VK_NULL_HANDLE;
		VkDescriptorSet m_depthStencilImTextureId = VK_NULL_HANDLE;
		void Initialize(VkExtent3D extent, VkImageViewType imageViewType, VkFormat colorFormat, VkFormat depthStencilFormat);

	public:
		explicit RenderTexture(VkExtent3D extent, VkImageViewType imageViewType = VK_IMAGE_VIEW_TYPE_2D, VkFormat colorFormat = VK_FORMAT_R32G32B32A32_SFLOAT, VkFormat depthStencilFormat = VK_FORMAT_D24_UNORM_S8_UINT);

		void Resize(VkExtent3D extent);

		[[nodiscard]] VkExtent3D GetExtent() const;
		[[nodiscard]] VkImageViewType GetImageViewType() const;
		[[nodiscard]] VkFormat GetColorFormat() const;
		[[nodiscard]] VkFormat GetDepthStencilFormat() const;

		[[nodiscard]] const std::unique_ptr<Image>& GetColorImage();
		[[nodiscard]] const std::unique_ptr<Image>& GetDepthStencilImage();
		[[nodiscard]] const std::unique_ptr<ImageView>& GetColorImageView();
		[[nodiscard]] const std::unique_ptr<ImageView>& GetDepthStencilImageView();

		[[nodiscard]] VkDescriptorSet GetColorImTextureId() const;
		[[nodiscard]] VkDescriptorSet GetDepthStencilImTextureId() const;

		[[nodiscard]] static const std::vector<VkAttachmentDescription>& GetAttachmentDescriptions();
	};
}
