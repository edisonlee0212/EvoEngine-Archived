#pragma once
#include "GraphicsPipelineStates.hpp"
#include "GraphicsResources.hpp"
namespace EvoEngine{
	struct RenderTextureCreateInfo
	{
		VkExtent3D m_extent = {1, 1, 1};
		VkImageViewType m_imageViewType = VK_IMAGE_VIEW_TYPE_2D;
		bool m_color = true;
		bool m_depth = true;
	};
	
	class RenderTexture
	{
		friend class Graphics;
		friend class RenderLayer;
		friend class Camera;
		std::shared_ptr<Image> m_colorImage = {};
		std::shared_ptr<ImageView> m_colorImageView = {};

		std::shared_ptr<Image> m_depthImage = {};
		std::shared_ptr<ImageView> m_depthImageView = {};

		VkExtent3D m_extent;
		VkImageViewType m_imageViewType;
		std::shared_ptr<Sampler> m_colorSampler = {};
		std::shared_ptr<Sampler> m_depthSampler = {};
		ImTextureID m_colorImTextureId = nullptr;

		bool m_color = true;
		bool m_depth = true;
		void Initialize(const RenderTextureCreateInfo& renderTextureCreateInfo);
		std::shared_ptr<DescriptorSet> m_descriptorSet;
	public:
		void Clear(VkCommandBuffer commandBuffer) const;
		explicit RenderTexture(const RenderTextureCreateInfo& renderTextureCreateInfo);
		void Resize(VkExtent3D extent);
		void AppendColorAttachmentInfos(std::vector<VkRenderingAttachmentInfo>& attachmentInfos, VkAttachmentLoadOp loadOp, VkAttachmentStoreOp storeOp) const;
		[[nodiscard]] VkRenderingAttachmentInfo GetDepthAttachmentInfo(VkAttachmentLoadOp loadOp, VkAttachmentStoreOp storeOp) const;
		[[nodiscard]] VkExtent3D GetExtent() const;
		[[nodiscard]] VkImageViewType GetImageViewType() const;
		
		[[nodiscard]] const std::shared_ptr<Sampler>& GetColorSampler() const;
		[[nodiscard]] const std::shared_ptr<Sampler>& GetDepthSampler() const;
		[[nodiscard]] const std::shared_ptr<Image>& GetColorImage();
		[[nodiscard]] const std::shared_ptr<Image>& GetDepthImage();
		[[nodiscard]] const std::shared_ptr<ImageView>& GetColorImageView();
		[[nodiscard]] const std::shared_ptr<ImageView>& GetDepthImageView();
		void BeginRendering(VkCommandBuffer commandBuffer, VkAttachmentLoadOp loadOp, VkAttachmentStoreOp storeOp) const;
		void EndRendering(VkCommandBuffer commandBuffer) const;
		[[nodiscard]] ImTextureID GetColorImTextureId() const;
		void ApplyGraphicsPipelineStates(GraphicsPipelineStates& globalPipelineState) const;
		[[maybe_unused]] bool Save(const std::filesystem::path& path) const;
		void StoreToPng(
			const std::string& path,
			int resizeX = -1,
			int resizeY = -1,
			unsigned compressionLevel = 8) const;
		void StoreToJpg(const std::string& path, int resizeX = -1, int resizeY = -1, unsigned quality = 100) const;
		void StoreToHdr(const std::string& path, int resizeX = -1, int resizeY = -1, unsigned quality = 100) const;
	};
}
