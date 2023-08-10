#pragma once
#include "IAsset.hpp"

namespace EvoEngine
{
	class Camera;

	struct SSAOSettings
	{
		
	};

	struct BloomSettings
	{
		
	};

	struct SSRSettings
	{
		
	};

	class PostProcessingStack : public IAsset
	{

	public:
		void Process(const std::shared_ptr<Camera>& targetCamera);
		SSAOSettings m_SSAOSettings {};
		BloomSettings m_bloomSettings{};
		SSRSettings m_SSRSettings{};

		bool m_SSAO = false;
		bool m_bloom = true;
		bool m_SSR = true;
	};
}
