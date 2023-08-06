#pragma once
#include "ISingleton.hpp"
#include "Material.hpp"
#include "Mesh.hpp"

namespace EvoEngine
{
	struct GizmoSettings {
		DrawSettings m_drawSettings;
		enum class ColorMode {
			Default,
			VertexColor,
			NormalColor
		} m_colorMode = ColorMode::Default;
		bool m_depthTest = true;
	};
	class Gizmos : public ISingleton<Gizmos>
	{
	public:
		static void DrawGizmoMesh(
			const std::shared_ptr<Mesh>& mesh,
			const glm::vec4& color = glm::vec4(1.0f),
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});
	};
}
