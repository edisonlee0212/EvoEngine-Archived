#pragma once
#include "GraphicsResources.hpp"
#include "IAsset.hpp"
#include "Texture2D.hpp"

namespace EvoEngine
{
	class Cubemap : public IAsset
	{
		std::unique_ptr<Image> m_image = {};
		std::unique_ptr<ImageView> m_imageView = {};
		std::unique_ptr<Sampler> m_sampler = {};

		//ImTextureID m_imTextureId = VK_NULL_HANDLE;
	public:
		void ConvertFromEquirectangularTexture(const std::shared_ptr<Texture2D>& targetTexture);
	};
}
