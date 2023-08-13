#pragma once
#include "IAsset.hpp"
#include "RenderTexture.hpp"
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
		std::shared_ptr<RenderTexture> m_renderTexture;
		void Resize(const glm::uvec2& size) const;

	public:

		void OnCreate() override;
		void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		void Process(const std::shared_ptr<Camera>& targetCamera);
		SSAOSettings m_SSAOSettings {};
		BloomSettings m_bloomSettings{};
		SSRSettings m_SSRSettings{};

		bool m_SSAO = false;
		bool m_bloom = true;
		bool m_SSR = true;
	};
}
