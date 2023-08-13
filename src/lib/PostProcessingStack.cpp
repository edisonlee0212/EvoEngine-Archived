#include "PostProcessingStack.hpp"
#include "Camera.hpp"
using namespace EvoEngine;

void PostProcessingStack::Resize(const glm::uvec2& size) const
{
	if (size.x == 0 || size.y == 0) return;
	if (size.x > 16384 || size.y >= 16384) return;
	m_renderTexture->Resize({ size.x, size.y, 1 });
}

void PostProcessingStack::OnCreate()
{
	RenderTextureCreateInfo renderTextureCreateInfo {};
	m_renderTexture = std::make_unique<RenderTexture>(renderTextureCreateInfo);

}

void PostProcessingStack::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	
}

void PostProcessingStack::Process(const std::shared_ptr<Camera>& targetCamera)
{
	const auto size = targetCamera->GetSize();
	m_renderTexture->Resize({ size.x, size.y, 1 });
	if(m_SSAO)
	{
		
	}
	if(m_SSR)
	{
		
	}
	if(m_bloom)
	{
		
	}
}
