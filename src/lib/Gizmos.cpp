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
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	if (!renderLayer)
		return;
	if (m_editorCameras.find(editorCameraComponent->GetHandle()) == m_editorCameras.end())
	{
		EVOENGINE_ERROR("Target camera not registered in editor!");
		return;
	}
	if (editorCameraComponent && editorCameraComponent->IsEnabled())
	{
		Graphics::AppendCommands([&](VkCommandBuffer commandBuffer)
			{
				std::shared_ptr<GraphicsPipeline> gizmosPipeline;
				switch (gizmoSettings.m_colorMode) {
				case GizmoSettings::ColorMode::Default:
				{
					gizmosPipeline = Graphics::GetGraphicsPipeline("GIZMOS");
				}break;
				case GizmoSettings::ColorMode::VertexColor:
				{
					gizmosPipeline = Graphics::GetGraphicsPipeline("GIZMOS_VERTEX_COLORED");
				}break;
				case GizmoSettings::ColorMode::NormalColor:
				{
					gizmosPipeline = Graphics::GetGraphicsPipeline("GIZMOS_NORMAL_COLORED");
				}break;
				}
				editorCameraComponent->GetRenderTexture()->ApplyGraphicsPipelineStates(gizmosPipeline->m_states);
				gizmoSettings.ApplySettings(gizmosPipeline->m_states);

				gizmosPipeline->Bind(commandBuffer);
				gizmosPipeline->BindDescriptorSet(commandBuffer, 0, renderLayer->GetPerFrameDescriptorSet()->GetVkDescriptorSet());

				editorCameraComponent->GetRenderTexture()->BeginRendering(commandBuffer, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
				GizmosPushConstant pushConstant;
				pushConstant.m_model = model;
				pushConstant.m_color = color;
				pushConstant.m_size = size;
				pushConstant.m_cameraIndex = renderLayer->GetCameraIndex(editorCameraComponent->GetHandle());
				gizmosPipeline->PushConstant(commandBuffer, 0, pushConstant);
				mesh->Bind(commandBuffer);
				mesh->DrawIndexed(commandBuffer, gizmosPipeline->m_states, 1, true);
				editorCameraComponent->GetRenderTexture()->EndRendering(commandBuffer);
			});
	}
}

void EditorLayer::DrawGizmoStrands(const std::shared_ptr<Strands>& strands, const std::shared_ptr<Camera>& editorCameraComponent,
	const glm::vec4& color, const glm::mat4& model,
	const float& size, const GizmoSettings& gizmoSettings)
{
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	if (!renderLayer)
		return;
	if (m_editorCameras.find(editorCameraComponent->GetHandle()) == m_editorCameras.end())
	{
		EVOENGINE_ERROR("Target camera not registered in editor!");
		return;
	}
	if (editorCameraComponent && editorCameraComponent->IsEnabled())
	{
		Graphics::AppendCommands([&](VkCommandBuffer commandBuffer)
			{
				std::shared_ptr<GraphicsPipeline> gizmosPipeline;
				switch (gizmoSettings.m_colorMode) {
				case GizmoSettings::ColorMode::Default:
				{
					gizmosPipeline = Graphics::GetGraphicsPipeline("GIZMOS_STRANDS");
				}break;
				case GizmoSettings::ColorMode::VertexColor:
				{
					gizmosPipeline = Graphics::GetGraphicsPipeline("GIZMOS_STRANDS_VERTEX_COLORED");
				}break;
				case GizmoSettings::ColorMode::NormalColor:
				{
					gizmosPipeline = Graphics::GetGraphicsPipeline("GIZMOS_STRANDS_NORMAL_COLORED");
				}break;
				}
				editorCameraComponent->GetRenderTexture()->ApplyGraphicsPipelineStates(gizmosPipeline->m_states);
				gizmoSettings.ApplySettings(gizmosPipeline->m_states);

				gizmosPipeline->Bind(commandBuffer);
				gizmosPipeline->BindDescriptorSet(commandBuffer, 0, renderLayer->GetPerFrameDescriptorSet()->GetVkDescriptorSet());

				editorCameraComponent->GetRenderTexture()->BeginRendering(commandBuffer, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
				GizmosPushConstant pushConstant;
				pushConstant.m_model = model;
				pushConstant.m_color = color;
				pushConstant.m_size = size;
				pushConstant.m_cameraIndex = renderLayer->GetCameraIndex(editorCameraComponent->GetHandle());
				gizmosPipeline->PushConstant(commandBuffer, 0, pushConstant);
				strands->Bind(commandBuffer);
				strands->DrawIndexed(commandBuffer, gizmosPipeline->m_states, 1, true);
				editorCameraComponent->GetRenderTexture()->EndRendering(commandBuffer);
			});
	}
}

void EditorLayer::DrawGizmoMeshInstancedColored(const std::shared_ptr<Mesh>& mesh,
	const std::shared_ptr<Camera>& editorCameraComponent, const std::shared_ptr<ParticleInfoList>& instancedData, const glm::mat4& model, const float& size,
	const GizmoSettings& gizmoSettings)
{
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	if (!renderLayer)
		return;
	if (m_editorCameras.find(editorCameraComponent->GetHandle()) == m_editorCameras.end())
	{
		EVOENGINE_ERROR("Target camera not registered in editor!");
		return;
	}
	if (editorCameraComponent && editorCameraComponent->IsEnabled())
	{
		instancedData->UploadData();
		Graphics::AppendCommands([&](VkCommandBuffer commandBuffer)
			{
				const auto gizmosPipeline = Graphics::GetGraphicsPipeline("GIZMOS_INSTANCED_COLORED");
				editorCameraComponent->GetRenderTexture()->ApplyGraphicsPipelineStates(gizmosPipeline->m_states);
				gizmoSettings.ApplySettings(gizmosPipeline->m_states);

				gizmosPipeline->Bind(commandBuffer);
				gizmosPipeline->BindDescriptorSet(commandBuffer, 0, renderLayer->GetPerFrameDescriptorSet()->GetVkDescriptorSet());
				gizmosPipeline->BindDescriptorSet(commandBuffer, 1, instancedData->GetDescriptorSet()->GetVkDescriptorSet());

				editorCameraComponent->GetRenderTexture()->BeginRendering(commandBuffer, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
				GizmosPushConstant pushConstant;
				pushConstant.m_model = model;
				pushConstant.m_color = glm::vec4(0.0f);
				pushConstant.m_size = size;
				pushConstant.m_cameraIndex = renderLayer->GetCameraIndex(editorCameraComponent->GetHandle());
				gizmosPipeline->PushConstant(commandBuffer, 0, pushConstant);
				mesh->Bind(commandBuffer);
				mesh->DrawIndexed(commandBuffer, gizmosPipeline->m_states, instancedData->m_particleInfos.size(), true);
				editorCameraComponent->GetRenderTexture()->EndRendering(commandBuffer);
			});
	}
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
