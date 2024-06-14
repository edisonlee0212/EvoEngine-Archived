#pragma once
#include "IAsset.hpp"
#include "RenderTexture.hpp"
namespace evo_engine
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
		int m_numBinarySearchSteps = 8;
		float m_step = 0.5f;
		float m_minRayStep = 0.1f;
		int m_maxSteps = 16;
	};

	struct SSAOPushConstant
	{

	};

	struct BloomPushConstant
	{

	};

	struct SSRPushConstant
	{
		uint32_t m_cameraIndex = 0;
		int m_numBinarySearchSteps = 8;
		float m_step = 0.5f;
		float m_minRayStep = 0.1f;
		int m_maxSteps = 16;
		float m_reflectionSpecularFalloffExponent = 3.0f;
		int m_horizontal = false;
		float m_weight[5] = { 0.227027f, 0.1945946f, 0.1216216f, 0.054054f, 0.016216f };
	};

	class PostProcessingStack : public IAsset
	{
		friend class Camera;
		std::shared_ptr<RenderTexture> m_renderTexture0;
		std::shared_ptr<RenderTexture> m_renderTexture1;
		std::shared_ptr<RenderTexture> m_renderTexture2;

		std::shared_ptr<DescriptorSet> m_SSRReflectDescriptorSet = VK_NULL_HANDLE;//SSR_REFLECT_LAYOUT: 0, 1, 2, 3
		std::shared_ptr<DescriptorSet> m_SSRBlurHorizontalDescriptorSet = VK_NULL_HANDLE;//RENDER_TEXTURE_PRESENT_LAYOUT: 0
		std::shared_ptr<DescriptorSet> m_SSRBlurVerticalDescriptorSet = VK_NULL_HANDLE;//RENDER_TEXTURE_PRESENT_LAYOUT: 0
		std::shared_ptr<DescriptorSet> m_SSRCombineDescriptorSet = VK_NULL_HANDLE;//SSR_COMBINE: 0, 1
		void Resize(const glm::uvec2& size) const;
	public:

		void OnCreate() override;
		bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		void Process(const std::shared_ptr<Camera>& targetCamera);
		SSAOSettings m_SSAOSettings {};
		BloomSettings m_bloomSettings{};
		SSRSettings m_SSRSettings{};

		bool m_SSAO = false;
		bool m_bloom = false;
		bool m_SSR = false;
	};
}
