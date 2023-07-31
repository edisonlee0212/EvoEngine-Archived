#pragma once
#include "Cubemap.hpp"

namespace EvoEngine
{
	class LightProbe : public IAsset
	{
		std::unique_ptr<Cubemap> m_irradianceMap;
		size_t m_irradianceMapResolution = 32;
		bool m_ready = false;
	public:
		float m_gamma = 1.0f;
		void ConstructFromCubemap(const std::shared_ptr<Cubemap>& targetCubemap);
	};
}