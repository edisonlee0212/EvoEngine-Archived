#pragma once
#include "GraphicsResources.hpp"

#include "Cubemap.hpp"
namespace EvoEngine
{
	struct TextureStorageHandle
	{
		int m_value = 0;
	};

	class Texture2DStorage
	{

	public:
		bool m_pendingDelete = false;

		std::shared_ptr<TextureStorageHandle> m_handle;

		std::shared_ptr<Image> m_image = {};
		std::shared_ptr<ImageView> m_imageView = {};
		std::shared_ptr<Sampler> m_sampler = {};
		ImTextureID m_imTextureId = VK_NULL_HANDLE;

		[[nodiscard]] VkImageLayout GetLayout() const;
		[[nodiscard]] VkImage GetVkImage() const;
		[[nodiscard]] VkImageView GetVkImageView() const;
		[[nodiscard]] VkSampler GetVkSampler() const;
		[[nodiscard]] std::shared_ptr<Image> GetImage() const;

		void SetData(const void* data, const glm::uvec2& resolution);
		void Clear();
	};
	class CubemapStorage
	{
	public:
		bool m_pendingDelete = false;

		std::shared_ptr<TextureStorageHandle> m_handle;

		std::shared_ptr<Image> m_image = {};
		std::shared_ptr<ImageView> m_imageView = {};
		std::shared_ptr<Sampler> m_sampler = {};

		std::vector<std::shared_ptr<ImageView>> m_faceViews;
		std::vector<ImTextureID> m_imTextureIds;
	};
	class TextureStorage : public ISingleton<TextureStorage>
	{
		std::vector<Texture2DStorage> m_texture2Ds;
		std::vector<std::weak_ptr<Cubemap>> m_cubemaps;

		bool m_requireTexture2DUpdate = false;
		std::vector<std::vector<bool>> m_cubemapPendingUpdates;

		friend class RenderLayer;
		friend class Graphics;
		static void DeviceSync();
	public:
		static const Texture2DStorage& PeekTexture2DStorage(const std::shared_ptr<TextureStorageHandle>& handle);
		static Texture2DStorage& RefTexture2DStorage(const std::shared_ptr<TextureStorageHandle>& handle);

		static void UnRegisterTexture2D(const std::shared_ptr<TextureStorageHandle>& handle);
		static void UnRegisterCubemap(const uint32_t index);
		static std::shared_ptr<TextureStorageHandle> RegisterTexture2D();
		static void RegisterCubemap(const std::shared_ptr<Cubemap>& cubemap);
		static void Initialize();
	};
}
