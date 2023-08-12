#pragma once
#include "GraphicsResources.hpp"
#include "Texture2D.hpp"
#include "Cubemap.hpp"
namespace EvoEngine
{
	class TextureStorage : public ISingleton<TextureStorage>
	{
		std::vector<std::shared_ptr<Texture2D>> m_texture2Ds;
		std::vector<std::shared_ptr<Cubemap>> m_cubemaps;

		std::vector<std::vector<bool>> m_texture2DPendingUpdates;
		std::vector<std::vector<bool>> m_cubemapPendingUpdates;

		friend class RenderLayer;
		static void DeviceSync();
	public:
		
		static void UnRegisterTexture2D(const std::shared_ptr<Texture2D>& texture2D);
		static void UnRegisterCubemap(const std::shared_ptr<Cubemap>& cubemap);
		static void RegisterTexture2D(const std::shared_ptr<Texture2D>& texture2D);
		static void RegisterCubemap(const std::shared_ptr<Cubemap>& cubemap);
		static void Initialize();
	};
}
