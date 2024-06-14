#include "PostProcessingStack.hpp"

#include "Application.hpp"
#include "Camera.hpp"
#include "GeometryStorage.hpp"
#include "Graphics.hpp"
#include "Mesh.hpp"
#include "RenderLayer.hpp"
#include "Resources.hpp"
using namespace evo_engine;

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

	m_SSRReflectDescriptorSet = std::make_shared<DescriptorSet>(Graphics::GetDescriptorSetLayout("SSR_REFLECT_LAYOUT"));
	m_SSRBlurHorizontalDescriptorSet = std::make_shared<DescriptorSet>(Graphics::GetDescriptorSetLayout("RENDER_TEXTURE_PRESENT_LAYOUT"));
	m_SSRBlurVerticalDescriptorSet = std::make_shared<DescriptorSet>(Graphics::GetDescriptorSetLayout("RENDER_TEXTURE_PRESENT_LAYOUT"));
	m_SSRCombineDescriptorSet = std::make_shared<DescriptorSet>(Graphics::GetDescriptorSetLayout("SSR_COMBINE_LAYOUT"));
}

bool PostProcessingStack::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	bool changed = false;
	if(m_SSR && ImGui::TreeNode("SSR Settings"))
	{
		if(ImGui::DragFloat("Step size", &m_SSRSettings.m_step, 0.1, 0.1, 10.0f)) changed = false;
		if (ImGui::DragInt("Max steps", &m_SSRSettings.m_maxSteps, 1, 1, 32)) changed = false;
		if (ImGui::DragInt("Binary search steps", &m_SSRSettings.m_numBinarySearchSteps, 1, 1, 16)) changed = false;
		ImGui::TreePop();
	}

	return changed;
}

void PostProcessingStack::Process(const std::shared_ptr<Camera>& targetCamera)
{
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	const auto size = targetCamera->GetSize();
	m_renderTexture0->Resize({ size.x, size.y, 1 });
	m_renderTexture1->Resize({ size.x, size.y, 1 });
	m_renderTexture2->Resize({ size.x, size.y, 1 });
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	if(m_SSAO)
	{
		
	}
	if (m_SSR)
	{
		{
			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = targetCamera->GetRenderTexture()->GetDepthImageView()->GetVkImageView();
			imageInfo.sampler = targetCamera->GetRenderTexture()->GetDepthSampler()->GetVkSampler();
			m_SSRReflectDescriptorSet->UpdateImageDescriptorBinding(18, imageInfo);
			imageInfo.imageView = targetCamera->m_gBufferNormalView->GetVkImageView();
			imageInfo.sampler = targetCamera->m_gBufferNormalSampler->GetVkSampler();
			m_SSRReflectDescriptorSet->UpdateImageDescriptorBinding(19, imageInfo);
			imageInfo.imageView = targetCamera->GetRenderTexture()->GetColorImageView()->GetVkImageView();
			imageInfo.sampler = targetCamera->GetRenderTexture()->GetColorSampler()->GetVkSampler();
			m_SSRReflectDescriptorSet->UpdateImageDescriptorBinding(20, imageInfo);
			imageInfo.imageView = targetCamera->m_gBufferMaterialView->GetVkImageView();
			imageInfo.sampler = targetCamera->m_gBufferMaterialSampler->GetVkSampler();
			m_SSRReflectDescriptorSet->UpdateImageDescriptorBinding(21, imageInfo);
		}
		{
			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = m_renderTexture1->GetColorImageView()->GetVkImageView();
			imageInfo.sampler = m_renderTexture1->GetColorSampler()->GetVkSampler();
			m_SSRBlurHorizontalDescriptorSet->UpdateImageDescriptorBinding(0, imageInfo);
		}
		{
			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = m_renderTexture2->GetColorImageView()->GetVkImageView();
			imageInfo.sampler = m_renderTexture2->GetColorSampler()->GetVkSampler();
			m_SSRBlurVerticalDescriptorSet->UpdateImageDescriptorBinding(0, imageInfo);
		}
		{
			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = m_renderTexture0->GetColorImageView()->GetVkImageView();
			imageInfo.sampler = m_renderTexture0->GetColorSampler()->GetVkSampler();
			m_SSRCombineDescriptorSet->UpdateImageDescriptorBinding(0, imageInfo);
			imageInfo.imageView = m_renderTexture1->GetColorImageView()->GetVkImageView();
			imageInfo.sampler = m_renderTexture1->GetColorSampler()->GetVkSampler();
			m_SSRCombineDescriptorSet->UpdateImageDescriptorBinding(1, imageInfo);
		}

		Graphics::AppendCommands([&](VkCommandBuffer commandBuffer) {
#pragma region Viewport and scissor
			VkRect2D renderArea;
			renderArea.offset = { 0, 0 };
			renderArea.extent.width = targetCamera->GetSize().x;
			renderArea.extent.height = targetCamera->GetSize().y;
			VkViewport viewport;
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = targetCamera->GetSize().x;
			viewport.height = targetCamera->GetSize().y;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;

			VkRect2D scissor;
			scissor.offset = { 0, 0 };
			scissor.extent.width = targetCamera->GetSize().x;
			scissor.extent.height = targetCamera->GetSize().y;
			VkRenderingInfo renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = renderArea;
			renderInfo.layerCount = 1;
#pragma endregion
			GeometryStorage::BindVertices(commandBuffer);
			{
				SSRPushConstant pushConstant{};
				pushConstant.m_numBinarySearchSteps = m_SSRSettings.m_numBinarySearchSteps;
				pushConstant.m_step = m_SSRSettings.m_step;
				pushConstant.m_maxSteps = m_SSRSettings.m_maxSteps;
				pushConstant.m_cameraIndex = renderLayer->GetCameraIndex(targetCamera->GetHandle());
				const auto mesh = Resources::GetResource<Mesh>("PRIMITIVE_TEX_PASS_THROUGH");
				std::vector<VkRenderingAttachmentInfo> colorAttachmentInfos;
				VkRenderingInfo renderInfo{};
				renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
				renderInfo.renderArea = renderArea;
				renderInfo.layerCount = 1;
				renderInfo.pDepthAttachment = VK_NULL_HANDLE;

				//Input texture
				targetCamera->TransitGBufferImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				targetCamera->m_renderTexture->GetDepthImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				targetCamera->m_renderTexture->GetColorImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				//Attachments
				m_renderTexture0->GetColorImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
				m_renderTexture1->GetColorImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
				m_renderTexture0->AppendColorAttachmentInfos(colorAttachmentInfos, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE);
				m_renderTexture1->AppendColorAttachmentInfos(colorAttachmentInfos, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE);
				renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
				renderInfo.pColorAttachments = colorAttachmentInfos.data();
				
				{
					const auto& ssrReflectPipeline = Graphics::GetGraphicsPipeline("SSR_REFLECT");
					vkCmdBeginRendering(commandBuffer, &renderInfo);
					ssrReflectPipeline->m_states.m_depthTest = false;
					ssrReflectPipeline->m_states.m_colorBlendAttachmentStates.clear();
					ssrReflectPipeline->m_states.m_colorBlendAttachmentStates.resize(colorAttachmentInfos.size());
					for (auto& i : ssrReflectPipeline->m_states.m_colorBlendAttachmentStates)
					{
						i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
						i.blendEnable = VK_FALSE;
					}
					ssrReflectPipeline->Bind(commandBuffer);
					ssrReflectPipeline->BindDescriptorSet(commandBuffer, 0, renderLayer->m_perFrameDescriptorSets[currentFrameIndex]->GetVkDescriptorSet());
					ssrReflectPipeline->BindDescriptorSet(commandBuffer, 1, m_SSRReflectDescriptorSet->GetVkDescriptorSet());
					ssrReflectPipeline->m_states.m_viewPort = viewport;
					ssrReflectPipeline->m_states.m_scissor = scissor;

					ssrReflectPipeline->PushConstant(commandBuffer, 0, pushConstant);
					mesh->DrawIndexed(commandBuffer, ssrReflectPipeline->m_states, 1);
					vkCmdEndRendering(commandBuffer);
				}
				//Input texture
				m_renderTexture1->GetColorImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				//Attachments
				colorAttachmentInfos.clear();
				m_renderTexture2->GetColorImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
				m_renderTexture2->AppendColorAttachmentInfos(colorAttachmentInfos, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
				renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
				renderInfo.pColorAttachments = colorAttachmentInfos.data();
				{
					const auto& ssrBlurPipeline = Graphics::GetGraphicsPipeline("SSR_BLUR");
					vkCmdBeginRendering(commandBuffer, &renderInfo);
					ssrBlurPipeline->m_states.m_depthTest = false;
					ssrBlurPipeline->m_states.m_colorBlendAttachmentStates.clear();
					ssrBlurPipeline->m_states.m_colorBlendAttachmentStates.resize(colorAttachmentInfos.size());
					for (auto& i : ssrBlurPipeline->m_states.m_colorBlendAttachmentStates)
					{
						i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
						i.blendEnable = VK_FALSE;
					}
					ssrBlurPipeline->Bind(commandBuffer);
					ssrBlurPipeline->BindDescriptorSet(commandBuffer, 0, m_SSRBlurHorizontalDescriptorSet->GetVkDescriptorSet());
					ssrBlurPipeline->m_states.m_viewPort = viewport;
					ssrBlurPipeline->m_states.m_scissor = scissor;
					pushConstant.m_horizontal = true;
					ssrBlurPipeline->PushConstant(commandBuffer, 0, pushConstant);
					mesh->DrawIndexed(commandBuffer, ssrBlurPipeline->m_states, 1);
					vkCmdEndRendering(commandBuffer);
				}
				//Input texture
				m_renderTexture2->GetColorImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				//Attachments
				colorAttachmentInfos.clear();
				m_renderTexture1->GetColorImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
				m_renderTexture1->AppendColorAttachmentInfos(colorAttachmentInfos, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
				renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
				renderInfo.pColorAttachments = colorAttachmentInfos.data();
				{
					const auto& ssrBlurPipeline = Graphics::GetGraphicsPipeline("SSR_BLUR");
					vkCmdBeginRendering(commandBuffer, &renderInfo);
					ssrBlurPipeline->m_states.m_depthTest = false;
					ssrBlurPipeline->m_states.m_colorBlendAttachmentStates.clear();
					ssrBlurPipeline->m_states.m_colorBlendAttachmentStates.resize(colorAttachmentInfos.size());
					for (auto& i : ssrBlurPipeline->m_states.m_colorBlendAttachmentStates)
					{
						i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
						i.blendEnable = VK_FALSE;
					}
					ssrBlurPipeline->Bind(commandBuffer);
					ssrBlurPipeline->BindDescriptorSet(commandBuffer, 0, m_SSRBlurVerticalDescriptorSet->GetVkDescriptorSet());
					ssrBlurPipeline->m_states.m_viewPort = viewport;
					ssrBlurPipeline->m_states.m_scissor = scissor;
					pushConstant.m_horizontal = false;
					ssrBlurPipeline->PushConstant(commandBuffer, 0, pushConstant);
					mesh->DrawIndexed(commandBuffer, ssrBlurPipeline->m_states, 1);
					vkCmdEndRendering(commandBuffer);
				}
				//Input texture
				m_renderTexture0->GetColorImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				m_renderTexture1->GetColorImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				//Attachments
				colorAttachmentInfos.clear();
				targetCamera->m_renderTexture->GetColorImage()->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
				targetCamera->m_renderTexture->AppendColorAttachmentInfos(colorAttachmentInfos, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE);
				renderInfo.colorAttachmentCount = colorAttachmentInfos.size();
				renderInfo.pColorAttachments = colorAttachmentInfos.data();
				{
					const auto& ssrCombinePipeline = Graphics::GetGraphicsPipeline("SSR_COMBINE");
					vkCmdBeginRendering(commandBuffer, &renderInfo);
					ssrCombinePipeline->m_states.m_depthTest = false;
					ssrCombinePipeline->m_states.m_colorBlendAttachmentStates.clear();
					ssrCombinePipeline->m_states.m_colorBlendAttachmentStates.resize(colorAttachmentInfos.size());
					for (auto& i : ssrCombinePipeline->m_states.m_colorBlendAttachmentStates)
					{
						i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
						i.blendEnable = VK_FALSE;
					}
					ssrCombinePipeline->Bind(commandBuffer);
					ssrCombinePipeline->BindDescriptorSet(commandBuffer, 0, m_SSRCombineDescriptorSet->GetVkDescriptorSet());
					ssrCombinePipeline->m_states.m_viewPort = viewport;
					ssrCombinePipeline->m_states.m_scissor = scissor;
					ssrCombinePipeline->PushConstant(commandBuffer, 0, pushConstant);
					mesh->DrawIndexed(commandBuffer, ssrCombinePipeline->m_states, 1);
					vkCmdEndRendering(commandBuffer);
				}
			}
			}
		);
	}
	if(m_bloom)
	{
		
	}
}
