#include "Camera.hpp"

using namespace EvoEngine;

std::shared_ptr<Texture2D> Camera::GetTexture() const
{
	return m_colorTexture;
}

std::shared_ptr<Texture2D> Camera::GetDepthStencil() const
{
	return m_depthStencilTexture;
}
