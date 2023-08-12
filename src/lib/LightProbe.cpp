#include "LightProbe.hpp"

#include "EditorLayer.hpp"
#include "Mesh.hpp"

using namespace EvoEngine;
struct EquirectangularToCubemapConstant
{
	glm::mat4 m_projectionView = {};
	float m_preset = 0;

};

void LightProbe::Initialize(uint32_t resolution)
{
	m_cubemap = ProjectManager::CreateTemporaryAsset<Cubemap>();
	m_cubemap->Initialize(resolution);

#pragma endregion
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer) {
		m_cubemap->m_image->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
	);
}

void LightProbe::ConstructFromCubemap(const std::shared_ptr<Cubemap>& targetCubemap)
{
	if (!m_cubemap) Initialize();

	if (!targetCubemap->m_image) {
		EVOENGINE_ERROR("Target cubemap doesn't contain any content!");
		return;
	}
#pragma region Create image

#pragma region Depth
	VkImageCreateInfo depthImageInfo{};
	depthImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	depthImageInfo.imageType = VK_IMAGE_TYPE_2D;
	depthImageInfo.extent.width = m_cubemap->m_image->GetExtent().width;
	depthImageInfo.extent.height = m_cubemap->m_image->GetExtent().height;
	depthImageInfo.extent.depth = 1;
	depthImageInfo.mipLevels = 1;
	depthImageInfo.arrayLayers = 1;
	depthImageInfo.format = Graphics::Constants::SHADOW_MAP;
	depthImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	depthImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	depthImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	depthImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	auto depthImage = std::make_shared<Image>(depthImageInfo);
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer)
		{
			depthImage->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
		});


	VkImageViewCreateInfo depthViewInfo{};
	depthViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	depthViewInfo.image = depthImage->GetVkImage();
	depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	depthViewInfo.format = Graphics::Constants::SHADOW_MAP;
	depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	depthViewInfo.subresourceRange.baseMipLevel = 0;
	depthViewInfo.subresourceRange.levelCount = 1;
	depthViewInfo.subresourceRange.baseArrayLayer = 0;
	depthViewInfo.subresourceRange.layerCount = 1;
	auto depthImageView = std::make_shared<ImageView>(depthViewInfo);
#pragma endregion

	std::unique_ptr<DescriptorSet> tempSet = std::make_unique<DescriptorSet>(Graphics::GetDescriptorSetLayout("RENDER_TEXTURE_PRESENT"));
	VkDescriptorImageInfo descriptorImageInfo{};
	descriptorImageInfo.imageView = targetCubemap->GetImageView()->GetVkImageView();
	descriptorImageInfo.imageLayout = targetCubemap->GetImage()->GetLayout();
	descriptorImageInfo.sampler = targetCubemap->GetSampler()->GetVkSampler();

	tempSet->UpdateImageDescriptorBinding(0, descriptorImageInfo);


	glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
	glm::mat4 captureViews[] = {
		glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
		glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
		glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
		glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
		glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
		glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f)) };

	auto irradianceConstruct = Graphics::GetGraphicsPipeline("IRRADIANCE_CONSTRUCT");
	Graphics::ImmediateSubmit([&](VkCommandBuffer commandBuffer) {
		m_cubemap->m_image->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL);
#pragma region Viewport and scissor
		VkRect2D renderArea;
		renderArea.offset = { 0, 0 };
		renderArea.extent.width = m_cubemap->m_image->GetExtent().width;
		renderArea.extent.height = m_cubemap->m_image->GetExtent().height;
		VkViewport viewport;
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = m_cubemap->m_image->GetExtent().width;
		viewport.height = m_cubemap->m_image->GetExtent().height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor;
		scissor.offset = { 0, 0 };
		scissor.extent.width = m_cubemap->m_image->GetExtent().width;
		scissor.extent.height = m_cubemap->m_image->GetExtent().height;
		irradianceConstruct->m_states.m_viewPort = viewport;
		irradianceConstruct->m_states.m_scissor = scissor;
#pragma endregion
		GeometryStorage::BindVertices(commandBuffer);
		for (int i = 0; i < 6; i++) {
#pragma region Lighting pass

			VkRenderingAttachmentInfo attachment{};
			attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

			attachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
			attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

			attachment.clearValue = { 0, 0, 0, 1 };
			attachment.imageView = m_cubemap->m_faceViews[i]->GetVkImageView();

			VkRenderingAttachmentInfo depthAttachment{};
			depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

			depthAttachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
			depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

			depthAttachment.clearValue.depthStencil = { 1, 0 };
			depthAttachment.imageView = depthImageView->GetVkImageView();

			VkRenderingInfo renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = renderArea;
			renderInfo.layerCount = 1;
			renderInfo.colorAttachmentCount = 1;
			renderInfo.pColorAttachments = &attachment;
			renderInfo.pDepthAttachment = &depthAttachment;
			irradianceConstruct->m_states.m_cullMode = VK_CULL_MODE_NONE;
			irradianceConstruct->m_states.m_colorBlendAttachmentStates.clear();
			irradianceConstruct->m_states.m_colorBlendAttachmentStates.resize(1);
			for (auto& i : irradianceConstruct->m_states.m_colorBlendAttachmentStates)
			{
				i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
				i.blendEnable = VK_FALSE;
			}
			vkCmdBeginRendering(commandBuffer, &renderInfo);
			irradianceConstruct->Bind(commandBuffer);
			irradianceConstruct->BindDescriptorSet(commandBuffer, 0, tempSet->GetVkDescriptorSet());
			const auto mesh = Resources::GetResource<Mesh>("PRIMITIVE_RENDERING_CUBE");
			EquirectangularToCubemapConstant constant{};
			constant.m_projectionView = captureProjection * captureViews[i];
			irradianceConstruct->PushConstant(commandBuffer, 0, constant);
			mesh->DrawIndexed(commandBuffer, irradianceConstruct->m_states, 1, false);
			vkCmdEndRendering(commandBuffer);
#pragma endregion

			Graphics::EverythingBarrier(commandBuffer);
		}
		m_cubemap->m_image->TransitImageLayout(commandBuffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
	);

}

void LightProbe::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	if (!m_cubemap->m_imTextureIds.empty()) {
		static float debugSacle = 0.25f;
		ImGui::DragFloat("Scale", &debugSacle, 0.01f, 0.1f, 1.0f);
		debugSacle = glm::clamp(debugSacle, 0.1f, 1.0f);
		for (int i = 0; i < 6; i++) {
			ImGui::Image(m_cubemap->m_imTextureIds[i],
				ImVec2(m_cubemap->m_image->GetExtent().width * debugSacle, m_cubemap->m_image->GetExtent().height * debugSacle),
				ImVec2(0, 1),
				ImVec2(1, 0));
		}
	}
}
