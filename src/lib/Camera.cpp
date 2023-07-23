#include "Camera.hpp"

using namespace EvoEngine;

std::shared_ptr<RenderTexture> Camera::GetRenderTexture() const
{
	return m_renderTexture;
}

glm::vec2 Camera::GetSize() const
{
	return m_size;
}

void Camera::Resize(const glm::uvec2& size)
{
	m_size = size;
	m_renderTexture->Resize({ m_size.x, m_size.y, 1 });
}

void Camera::OnCreate()
{
	m_size = glm::uvec2(1, 1);
	VkExtent3D extent;
	extent.width = m_size.x;
	extent.height = m_size.y;
	extent.depth = 1;
	m_renderTexture = std::make_unique<RenderTexture>(extent, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT);
}
