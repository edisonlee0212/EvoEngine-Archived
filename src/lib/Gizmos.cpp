#include "Gizmos.hpp"
#include "Application.hpp"
#include "RenderLayer.hpp"
#include "EditorLayer.hpp"
using namespace EvoEngine;

void Gizmos::DrawGizmoMesh(const std::shared_ptr<Mesh>& mesh, const glm::vec4& color, const glm::mat4& model,
	const float& size, const GizmoSettings& gizmoSettings)
{
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	if (!renderLayer)
		return;
	const auto editorLayer = Application::GetLayer<EditorLayer>();
	if (!editorLayer)
		return;
	const auto sceneCamera = editorLayer->GetSceneCamera();
	if (sceneCamera && sceneCamera->IsEnabled())
	{
		
	}
}
