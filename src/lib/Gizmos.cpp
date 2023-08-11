#include "Application.hpp"
#include "RenderLayer.hpp"
#include "EditorLayer.hpp"
using namespace EvoEngine;

void GizmoSettings::ApplySettings(GraphicsPipelineStates& globalPipelineState) const
{
	m_drawSettings.ApplySettings(globalPipelineState);
	globalPipelineState.m_depthTest = m_depthTest;
	globalPipelineState.m_depthWrite = m_depthWrite;
}


void EditorLayer::DrawGizmoMesh(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<Camera>& editorCameraComponent,
	const glm::vec4& color, const glm::mat4& model,
	const float& size, const GizmoSettings& gizmoSettings)
{
	if(Application::GetApplicationExecutionStatus() == ApplicationExecutionStatus::LateUpdate)
	{
		EVOENGINE_ERROR("Gizmos command ignored! Submit gizmos command during LateUpdate is now allowed!");
		return;
	}
	m_gizmoMeshTasks.push_back({ mesh, editorCameraComponent, color, model, size, gizmoSettings });
}

void EditorLayer::DrawGizmoStrands(const std::shared_ptr<Strands>& strands, const std::shared_ptr<Camera>& editorCameraComponent,
	const glm::vec4& color, const glm::mat4& model,
	const float& size, const GizmoSettings& gizmoSettings)
{
	if (Application::GetApplicationExecutionStatus() == ApplicationExecutionStatus::LateUpdate)
	{
		EVOENGINE_ERROR("Gizmos command ignored! Submit gizmos command during LateUpdate is now allowed!");
		return;
	}
	m_gizmoStrandsTasks.push_back({ strands, editorCameraComponent, color, model, size, gizmoSettings });
}

void EditorLayer::DrawGizmoMeshInstancedColored(const std::shared_ptr<Mesh>& mesh,
	const std::shared_ptr<Camera>& editorCameraComponent, 
	const std::shared_ptr<ParticleInfoList>& instancedData, const glm::mat4& model, const float& size,
	const GizmoSettings& gizmoSettings)
{
	if (Application::GetApplicationExecutionStatus() == ApplicationExecutionStatus::LateUpdate)
	{
		EVOENGINE_ERROR("Gizmos command ignored! Submit gizmos command during LateUpdate is now allowed!");
		return;
	}
	m_gizmoInstancedMeshTasks.push_back({ mesh, editorCameraComponent, instancedData, model, size, gizmoSettings });
}

void EditorLayer::DrawGizmoMeshInstancedColored(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<ParticleInfoList>& instancedData, const glm::mat4& model, const float& size,
	const GizmoSettings& gizmoSettings)
{
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	if (!renderLayer)
		return;
	const auto sceneCamera = GetSceneCamera();
	DrawGizmoMeshInstancedColored(mesh, sceneCamera, instancedData, model, size, gizmoSettings);
}

void EditorLayer::DrawGizmoMesh(const std::shared_ptr<Mesh>& mesh, const glm::vec4& color, const glm::mat4& model,
                           const float& size, const GizmoSettings& gizmoSettings)
{
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	if (!renderLayer)
		return;
	const auto sceneCamera = GetSceneCamera();
	DrawGizmoMesh(mesh, sceneCamera, color, model, size, gizmoSettings);
}

void EditorLayer::DrawGizmoStrands(const std::shared_ptr<Strands>& strands, const glm::vec4& color, const glm::mat4& model,
	const float& size, const GizmoSettings& gizmoSettings)
{
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	if (!renderLayer)
		return;
	const auto sceneCamera = GetSceneCamera();
	DrawGizmoStrands(strands, sceneCamera, color, model, size, gizmoSettings);
}

void EditorLayer::DrawGizmoCubes(const std::shared_ptr<ParticleInfoList>& instancedData,
                            const glm::mat4& model, const float& size, const GizmoSettings& gizmoSettings)
{
	DrawGizmoMeshInstancedColored(Resources::GetResource<Mesh>("PRIMITIVE_CUBE"), instancedData, model, size, gizmoSettings);
}

void EditorLayer::DrawGizmoCube(const glm::vec4& color, const glm::mat4& model, const float& size,
	const GizmoSettings& gizmoSettings)
{
	DrawGizmoMesh(Resources::GetResource<Mesh>("PRIMITIVE_CUBE"), color, model, size, gizmoSettings);
}

void EditorLayer::DrawGizmoSpheres(const std::shared_ptr<ParticleInfoList>& instancedData,
	const glm::mat4& model, const float& size, const GizmoSettings& gizmoSettings)
{
	DrawGizmoMeshInstancedColored(Resources::GetResource<Mesh>("PRIMITIVE_SPHERE"), instancedData, model, size, gizmoSettings);
}

void EditorLayer::DrawGizmoSphere(const glm::vec4& color, const glm::mat4& model, const float& size,
                             const GizmoSettings& gizmoSettings)
{
	DrawGizmoMesh(Resources::GetResource<Mesh>("PRIMITIVE_SPHERE"), color, model, size, gizmoSettings);
}

void EditorLayer::DrawGizmoCylinders(const std::shared_ptr<ParticleInfoList>& instancedData, const glm::mat4& model, const float& size,
	const GizmoSettings& gizmoSettings)
{
	DrawGizmoMeshInstancedColored(Resources::GetResource<Mesh>("PRIMITIVE_CYLINDER"), instancedData, model, size, gizmoSettings);
}

void EditorLayer::DrawGizmoCylinder(const glm::vec4& color, const glm::mat4& model, const float& size,
                               const GizmoSettings& gizmoSettings)
{
	DrawGizmoMesh(Resources::GetResource<Mesh>("PRIMITIVE_CYLINDER"), color, model, size, gizmoSettings);

}
