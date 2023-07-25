#include "RenderLayer.hpp"
#include "Application.hpp"
#include "Graphics.hpp"
#include "Utilities.hpp"
#include "WindowLayer.hpp"
#include "ProjectManager.hpp"
#include "EditorLayer.hpp"
using namespace EvoEngine;



glm::vec3 CameraInfoBlock::Project(const glm::vec3& position) const
{
	return m_projection * m_view * glm::vec4(position, 1.0f);
}

glm::vec3 CameraInfoBlock::UnProject(const glm::vec3& position) const
{
	const glm::mat4 inverse = glm::inverse(m_projection * m_view);
	auto start = glm::vec4(position, 1.0f);
	start = inverse * start;
	return start / start.w;
}



void RenderLayer::OnCreate()
{

	CreateRenderPasses();
}

void RenderLayer::OnDestroy()
{
	m_renderPasses.clear();
}

void RenderLayer::PreUpdate()
{
}

void RenderLayer::LateUpdate()
{
	/*
	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer, GlobalPipelineState& globalPipelineState)
		{
			const auto& windowLayer = Application::GetLayer<WindowLayer>();
			if (!windowLayer || windowLayer->m_windowSize.x == 0 || windowLayer->m_windowSize.y == 0) return;

			const auto extent2D = Graphics::GetSwapchain()->GetImageExtent();
			

			VkRenderPassBeginInfo renderPassBeginInfo{};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.renderPass = m_renderPass->GetVkRenderPass();
			renderPassBeginInfo.framebuffer = Graphics::GetSwapchainFramebuffer()->GetVkFrameBuffer();
			renderPassBeginInfo.renderArea.offset = { 0, 0 };
			renderPassBeginInfo.renderArea.extent = extent2D;

			VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
			renderPassBeginInfo.clearValueCount = 1;
			renderPassBeginInfo.pClearValues = &clearColor;

			vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			m_mesh->Bind(commandBuffer);


			VkViewport viewport;
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = static_cast<float>(extent2D.width);
			viewport.height = static_cast<float>(extent2D.height);
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;


			VkRect2D scissor;
			scissor.offset = { 0, 0 };
			scissor.extent = extent2D;


			globalPipelineState.m_viewPort = viewport;
			globalPipelineState.m_scissor = scissor;

			globalPipelineState.ClearShaders();
			globalPipelineState.m_vertexShader = m_vertShader;
			globalPipelineState.m_fragShader = m_fragShader;


			m_mesh->DrawIndexed(commandBuffer, globalPipelineState);

			vkCmdEndRenderPass(commandBuffer);
		});
		*/
}

void RenderLayer::CreateRenderPasses()
{
	auto editorLayer = Application::GetLayer<EditorLayer>();
	if (editorLayer)
	{
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = Graphics::GetSwapchain()->GetImageFormat();
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = VK_FORMAT_D24_UNORM_S8_UINT;
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		const std::vector<VkSubpassDescription> subpasses = { subpass };
		const std::vector<VkSubpassDependency> dependencies = { dependency };
		renderPassInfo.subpassCount = static_cast<uint32_t>(subpasses.size());
		renderPassInfo.pSubpasses = subpasses.data();
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();

		const std::vector attachments = { colorAttachment, depthAttachment };
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		m_renderPasses.insert({ "SCREEN_PRESENT", std::make_unique<RenderPass>(renderPassInfo) });
	}

	{
		VkSubpassDescription geometricSubpass{};
		VkSubpassDescription shadingSubpass{};
		{
			//Subpass 1: To gBuffer.
			VkAttachmentReference gBufferDepthAttachmentRef{};
			gBufferDepthAttachmentRef.attachment = 2;
			gBufferDepthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			VkAttachmentReference gBufferNormalAttachmentRef{};
			gBufferNormalAttachmentRef.attachment = 3;
			gBufferNormalAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			VkAttachmentReference gBufferAlbedoAttachmentRef{};
			gBufferAlbedoAttachmentRef.attachment = 4;
			gBufferAlbedoAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			VkAttachmentReference gBufferMaterialAttachmentRef{};
			gBufferMaterialAttachmentRef.attachment = 5;
			gBufferMaterialAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			std::vector colorReferences {gBufferNormalAttachmentRef, gBufferAlbedoAttachmentRef, gBufferMaterialAttachmentRef};
			
			geometricSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			geometricSubpass.colorAttachmentCount = colorReferences.size();
			geometricSubpass.pColorAttachments = colorReferences.data();
			geometricSubpass.pDepthStencilAttachment = &gBufferDepthAttachmentRef;

		}
		{
			//Subpass 2: To RenderTexture
			VkAttachmentReference depthAttachmentRef{};
			depthAttachmentRef.attachment = 0;
			depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			VkAttachmentReference colorAttachmentRef{};
			colorAttachmentRef.attachment = 1;
			colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			shadingSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			shadingSubpass.colorAttachmentCount = 1;
			shadingSubpass.pColorAttachments = &colorAttachmentRef;
			shadingSubpass.pDepthStencilAttachment = &depthAttachmentRef;
		}

		VkSubpassDependency geometryPassDependency{};
		geometryPassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		geometryPassDependency.dstSubpass = 0;
		geometryPassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		geometryPassDependency.srcAccessMask = 0;
		geometryPassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		geometryPassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		VkSubpassDependency shadingPassDependency{};
		shadingPassDependency.srcSubpass = 0;
		shadingPassDependency.dstSubpass = 1;
		shadingPassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		shadingPassDependency.srcAccessMask = 0;
		shadingPassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		shadingPassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		const std::vector subpasses = { geometricSubpass, shadingSubpass };
		const std::vector dependencies = { geometryPassDependency,  shadingPassDependency };

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.subpassCount = static_cast<uint32_t>(subpasses.size());
		renderPassInfo.pSubpasses = subpasses.data();
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassInfo.pDependencies = dependencies.data();
		auto attachmentDescriptions = Camera::GetAttachmentDescriptions();
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescriptions.size());
		renderPassInfo.pAttachments = attachmentDescriptions.data();
		m_renderPasses.insert({ "CAMERA_DEFERRED_SHADING", std::make_unique<RenderPass>(renderPassInfo) });
	}
}

const std::unique_ptr<RenderPass>& RenderLayer::GetRenderPass(const std::string& name)
{
	return m_renderPasses.at(name);
}
