#pragma once
#include "Cubemap.hpp"

namespace evo_engine
{
	class ReflectionProbe : public IAsset
	{
		std::shared_ptr<Cubemap> m_cubemap;
		friend class RenderLayer;
		friend class Camera;
		std::vector<std::vector<std::shared_ptr<ImageView>>> m_mipMapViews;
	public:
		void Initialize(uint32_t resolution = 512);
		[[nodiscard]] std::shared_ptr<Cubemap> GetCubemap() const;
		void ConstructFromCubemap(const std::shared_ptr<Cubemap>& targetCubemap);
		bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
	};
}
