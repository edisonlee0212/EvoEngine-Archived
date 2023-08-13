#include "PostProcessingStack.hpp"
#include "Camera.hpp"
using namespace EvoEngine;

void PostProcessingStack::Resize(const glm::uvec2& size) const
{
	if (size.x == 0 || size.y == 0) return;
	if (size.x > 16384 || size.y >= 16384) return;
	m_renderTexture0->Resize({ size.x, size.y, 1 });
	m_renderTexture1->Resize({ size.x, size.y, 1 });
	m_renderTexture2->Resize({ size.x, size.y, 1 });
}

void PostProcessingStack::OnCreate()
{
	RenderTextureCreateInfo renderTextureCreateInfo {};
	renderTextureCreateInfo.m_depth = false;
	m_renderTexture0 = std::make_unique<RenderTexture>(renderTextureCreateInfo);
	m_renderTexture1 = std::make_unique<RenderTexture>(renderTextureCreateInfo);
	m_renderTexture2 = std::make_unique<RenderTexture>(renderTextureCreateInfo);
}

void PostProcessingStack::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	
}

void PostProcessingStack::Process(const std::shared_ptr<Camera>& targetCamera)
{
	const auto size = targetCamera->GetSize();
	m_renderTexture0->Resize({ size.x, size.y, 1 });
	m_renderTexture1->Resize({ size.x, size.y, 1 });
	m_renderTexture2->Resize({ size.x, size.y, 1 });
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
