#pragma once
#include "Cubemap.hpp"

namespace evo_engine
{
	class LightProbe : public IAsset
	{
		std::shared_ptr<Cubemap> m_cubemap;
		friend class RenderLayer;
		friend class Camera;
	public:
		void Initialize(uint32_t resolution = 32);
		void ConstructFromCubemap(const std::shared_ptr<Cubemap>& targetCubemap);
		[[nodiscard]] std::shared_ptr<Cubemap> GetCubemap() const;
		bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
	};
}