#include "Gizmos.hpp"
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

void Gizmos::DrawGizmoMeshInstancedColored(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<InstancedInfoList>& instancedData, const glm::mat4& model, const float& size,
	const GizmoSettings& gizmoSettings)
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
		instancedData->UploadData();
		Graphics::AppendCommands([&](VkCommandBuffer commandBuffer)
			{
				const auto gizmosPipeline = Graphics::GetGraphicsPipeline("GIZMOS_INSTANCED_COLORED");
				sceneCamera->GetRenderTexture()->ApplyGraphicsPipelineStates(gizmosPipeline->m_states);
				gizmoSettings.ApplySettings(gizmosPipeline->m_states);

				gizmosPipeline->Bind(commandBuffer);
				gizmosPipeline->BindDescriptorSet(commandBuffer, 0, renderLayer->GetPerFrameDescriptorSet()->GetVkDescriptorSet());
				gizmosPipeline->BindDescriptorSet(commandBuffer, 1, instancedData->GetDescriptorSet()->GetVkDescriptorSet());

				sceneCamera->GetRenderTexture()->BeginRendering(commandBuffer, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
				GizmosPushConstant pushConstant;
				pushConstant.m_model = model;
				pushConstant.m_color = glm::vec4(0.0f);
				pushConstant.m_size = size;
				pushConstant.m_cameraIndex = renderLayer->GetCameraIndex(sceneCamera->GetHandle());
				gizmosPipeline->PushConstant(commandBuffer, 0, pushConstant);
				mesh->Bind(commandBuffer);
				mesh->DrawIndexed(commandBuffer, gizmosPipeline->m_states);
				sceneCamera->GetRenderTexture()->EndRendering(commandBuffer);
			});
	}
}

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
				sceneCamera->GetRenderTexture()->ApplyGraphicsPipelineStates(gizmosPipeline->m_states);
				gizmoSettings.ApplySettings(gizmosPipeline->m_states);

				gizmosPipeline->Bind(commandBuffer);
				gizmosPipeline->BindDescriptorSet(commandBuffer, 0, renderLayer->GetPerFrameDescriptorSet()->GetVkDescriptorSet());

				sceneCamera->GetRenderTexture()->BeginRendering(commandBuffer, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE);
				GizmosPushConstant pushConstant;
				pushConstant.m_model = model;
				pushConstant.m_color = color;
				pushConstant.m_size = size;
				pushConstant.m_cameraIndex = renderLayer->GetCameraIndex(sceneCamera->GetHandle());
				gizmosPipeline->PushConstant(commandBuffer, 0, pushConstant);
				mesh->Bind(commandBuffer);
				mesh->DrawIndexed(commandBuffer, gizmosPipeline->m_states);
				sceneCamera->GetRenderTexture()->EndRendering(commandBuffer);
			});
	}
}
