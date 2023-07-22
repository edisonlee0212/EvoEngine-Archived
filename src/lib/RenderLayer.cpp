#include "RenderLayer.hpp"
#include "Application.hpp"
#include "Graphics.hpp"
#include "Utilities.hpp"
#include "WindowLayer.hpp"
#include "ProjectManager.hpp"
#include "EditorLayer.hpp"
using namespace EvoEngine;
const std::vector<Vertex> vertices = {
	{{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
	{{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
	{{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
	{{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

	{{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
	{{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
	{{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
	{{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
};

const std::vector<unsigned> indices = {
	0, 1, 2, 2, 3, 0,
	4, 5, 6, 6, 7, 4
};
void RenderLayer::OnCreate()
{
#pragma region Descrioptor Layout
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();

	VkDescriptorSetLayoutBinding renderInfoLayoutBinding{};
	renderInfoLayoutBinding.binding = 0;
	renderInfoLayoutBinding.descriptorCount = 1;
	renderInfoLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	renderInfoLayoutBinding.pImmutableSamplers = nullptr;
	renderInfoLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

	VkDescriptorSetLayoutBinding environmentInfoLayoutBinding{};
	environmentInfoLayoutBinding.binding = 1;
	environmentInfoLayoutBinding.descriptorCount = 1;
	environmentInfoLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	environmentInfoLayoutBinding.pImmutableSamplers = nullptr;
	environmentInfoLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

	VkDescriptorSetLayoutBinding cameraInfoLayoutBinding{};
	cameraInfoLayoutBinding.binding = 2;
	cameraInfoLayoutBinding.descriptorCount = 1;
	cameraInfoLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	cameraInfoLayoutBinding.pImmutableSamplers = nullptr;
	cameraInfoLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

	VkDescriptorSetLayoutBinding materialInfoLayoutBinding{};
	materialInfoLayoutBinding.binding = 3;
	materialInfoLayoutBinding.descriptorCount = 1;
	materialInfoLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	materialInfoLayoutBinding.pImmutableSamplers = nullptr;
	materialInfoLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

	const std::vector perFrameBindings = { renderInfoLayoutBinding, environmentInfoLayoutBinding };
	const std::vector perPassBindings = { cameraInfoLayoutBinding };
	const std::vector perObjectGroupBindings = { materialInfoLayoutBinding };

	VkDescriptorSetLayoutCreateInfo perFrameLayoutInfo{};
	perFrameLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	perFrameLayoutInfo.bindingCount = static_cast<uint32_t>(perFrameBindings.size());
	perFrameLayoutInfo.pBindings = perFrameBindings.data();

	m_perFrameLayout = std::make_unique<DescriptorSetLayout>(perFrameLayoutInfo);

	VkDescriptorSetLayoutCreateInfo perPassLayoutInfo{};
	perPassLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	perPassLayoutInfo.bindingCount = static_cast<uint32_t>(perPassBindings.size());
	perPassLayoutInfo.pBindings = perPassBindings.data();

	m_perPassLayout = std::make_unique<DescriptorSetLayout>(perPassLayoutInfo);

	VkDescriptorSetLayoutCreateInfo perObjectGroupLayoutInfo{};
	perObjectGroupLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	perObjectGroupLayoutInfo.bindingCount = static_cast<uint32_t>(perObjectGroupBindings.size());
	perObjectGroupLayoutInfo.pBindings = perObjectGroupBindings.data();

	m_perObjectGroupLayout = std::make_unique<DescriptorSetLayout>(perObjectGroupLayoutInfo);


	std::vector perFrameLayouts(maxFramesInFlight, m_perFrameLayout->GetVkDescriptorSetLayout());
	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = Graphics::GetDescriptorPool()->GetVkDescriptorPool();
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perFrameLayouts.data();
	m_perFrameDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perFrameDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	std::vector perPassLayout(maxFramesInFlight, m_perPassLayout->GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perPassLayout.data();
	m_perPassDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perPassDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	std::vector perObjectGroupLayout(maxFramesInFlight, m_perObjectGroupLayout->GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perObjectGroupLayout.data();
	m_perObjectGroupDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perObjectGroupDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	size_t bindingsSize = perFrameBindings.size() + perPassBindings.size() + perObjectGroupBindings.size();

	VkDescriptorBufferInfo bufferInfos[4] = {};
	bufferInfos[0].offset = 0;
	bufferInfos[0].range = sizeof(RenderInfoBlock);
	bufferInfos[1].offset = 0;
	bufferInfos[1].range = sizeof(EnvironmentInfoBlock);
	bufferInfos[2].offset = 0;
	bufferInfos[2].range = sizeof(CameraInfoBlock);
	bufferInfos[3].offset = 0;
	bufferInfos[3].range = sizeof(MaterialInfoBlock);
	m_descriptorBuffers.clear();

	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo bufferVmaAllocationCreateInfo{};
	bufferVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	bufferVmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

	m_renderInfoBlockMemory.resize(maxFramesInFlight);
	m_environmentalInfoBlockMemory.resize(maxFramesInFlight);
	m_cameraInfoBlockMemory.resize(maxFramesInFlight);
	m_materialInfoBlockMemory.resize(maxFramesInFlight);
	for (size_t i = 0; i < maxFramesInFlight; i++) {
		bufferCreateInfo.size = sizeof(RenderInfoBlock);
		m_descriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(EnvironmentInfoBlock);
		m_descriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(CameraInfoBlock);
		m_descriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		bufferCreateInfo.size = sizeof(MaterialInfoBlock);
		m_descriptorBuffers.emplace_back(std::make_unique<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));

		bufferInfos[0].buffer = m_descriptorBuffers[i * 4 + 0]->GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), m_descriptorBuffers[i * 4 + 0]->GetVmaAllocation(), &m_renderInfoBlockMemory[i]);

		bufferInfos[1].buffer = m_descriptorBuffers[i * 4 + 1]->GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), m_descriptorBuffers[i * 4 + 0]->GetVmaAllocation(), &m_environmentalInfoBlockMemory[i]);

		bufferInfos[2].buffer = m_descriptorBuffers[i * 4 + 2]->GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), m_descriptorBuffers[i * 4 + 0]->GetVmaAllocation(), &m_cameraInfoBlockMemory[i]);

		bufferInfos[3].buffer = m_descriptorBuffers[i * 4 + 3]->GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), m_descriptorBuffers[i * 4 + 0]->GetVmaAllocation(), &m_materialInfoBlockMemory[i]);


		VkWriteDescriptorSet renderInfo{};
		renderInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		renderInfo.dstSet = m_perFrameDescriptorSets[i];
		renderInfo.dstBinding = 0;
		renderInfo.dstArrayElement = 0;
		renderInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		renderInfo.descriptorCount = 1;
		renderInfo.pBufferInfo = &bufferInfos[0];

		VkWriteDescriptorSet environmentInfo{};
		environmentInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		environmentInfo.dstSet = m_perFrameDescriptorSets[i];
		environmentInfo.dstBinding = 1;
		environmentInfo.dstArrayElement = 0;
		environmentInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		environmentInfo.descriptorCount = 1;
		environmentInfo.pBufferInfo = &bufferInfos[1];

		VkWriteDescriptorSet cameraInfo{};
		cameraInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		cameraInfo.dstSet = m_perPassDescriptorSets[i];
		cameraInfo.dstBinding = 2;
		cameraInfo.dstArrayElement = 0;
		cameraInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		cameraInfo.descriptorCount = 1;
		cameraInfo.pBufferInfo = &bufferInfos[2];

		VkWriteDescriptorSet materialInfo{};
		materialInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		materialInfo.dstSet = m_perObjectGroupDescriptorSets[i];
		materialInfo.dstBinding = 3;
		materialInfo.dstArrayElement = 0;
		materialInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		materialInfo.descriptorCount = 1;
		materialInfo.pBufferInfo = &bufferInfos[3];

		std::vector writeInfos = { renderInfo, environmentInfo, cameraInfo, materialInfo };
		vkUpdateDescriptorSets(Graphics::GetVkDevice(), 4, writeInfos.data(), 0, nullptr);
	}
#pragma endregion
	CreateRenderPass();
	UpdateFramebuffers();

	m_mesh = ProjectManager::CreateTemporaryAsset<Mesh>();
	m_mesh->SetVertices({}, vertices, indices);
	const auto vertShaderBinary = ShaderUtils::CompileFile("Shader", shaderc_vertex_shader, FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/shader.vert"));
	VkShaderCreateInfoEXT vertShaderCreateInfo{};
	vertShaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
	vertShaderCreateInfo.pNext = nullptr;
	vertShaderCreateInfo.flags = 0;
	vertShaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderCreateInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	vertShaderCreateInfo.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
	vertShaderCreateInfo.codeSize = vertShaderBinary.size() * sizeof(uint32_t);
	vertShaderCreateInfo.pCode = vertShaderBinary.data();
	vertShaderCreateInfo.pName = "main";
	vertShaderCreateInfo.setLayoutCount = 0;
	vertShaderCreateInfo.pSetLayouts = nullptr;
	vertShaderCreateInfo.pushConstantRangeCount = 0;
	vertShaderCreateInfo.pPushConstantRanges = nullptr;
	vertShaderCreateInfo.pSpecializationInfo = nullptr;
	m_vertShader = std::make_unique<ShaderEXT>(vertShaderCreateInfo);
	const auto fragShaderBinary = ShaderUtils::CompileFile("Shader", shaderc_fragment_shader, FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/shader.frag"));
	VkShaderCreateInfoEXT fragShaderCreateInfo{};
	fragShaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
	fragShaderCreateInfo.pNext = nullptr;
	fragShaderCreateInfo.flags = 0;
	fragShaderCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderCreateInfo.nextStage = 0;
	fragShaderCreateInfo.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT;
	fragShaderCreateInfo.codeSize = fragShaderBinary.size() * sizeof(uint32_t);
	fragShaderCreateInfo.pCode = fragShaderBinary.data();
	fragShaderCreateInfo.pName = "main";
	fragShaderCreateInfo.setLayoutCount = 0;
	fragShaderCreateInfo.pSetLayouts = nullptr;
	fragShaderCreateInfo.pushConstantRangeCount = 0;
	fragShaderCreateInfo.pPushConstantRanges = nullptr;
	fragShaderCreateInfo.pSpecializationInfo = nullptr;
	m_fragShader = std::make_unique<ShaderEXT>(fragShaderCreateInfo);
}

void RenderLayer::OnDestroy()
{
	m_descriptorBuffers.clear();

	m_perObjectGroupLayout.reset();
	m_perPassLayout.reset();
	m_perFrameLayout.reset();


	m_pipelineLayout.reset();
	m_renderPass.reset();
}

void RenderLayer::PreUpdate()
{
	UpdateFramebuffers();
}

void RenderLayer::LateUpdate()
{
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	memcpy(m_renderInfoBlockMemory[currentFrameIndex], &m_renderInfoBlock, sizeof(RenderInfoBlock));

	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer, GlobalPipelineState& globalPipelineState)
		{
			const auto& windowLayer = Application::GetLayer<WindowLayer>();
			if (!windowLayer || windowLayer->m_windowSize.x == 0 || windowLayer->m_windowSize.y == 0) return;

			const auto extent2D = Graphics::GetSwapchain()->GetImageExtent();
			VkRenderPassBeginInfo renderPassBeginInfo{};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.renderPass = m_renderPass->GetVkRenderPass();
			renderPassBeginInfo.framebuffer = m_framebuffers[Graphics::GetNextImageIndex()]->GetVkFrameBuffer();
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
}

bool RenderLayer::UpdateFramebuffers()
{
	const auto currentSwapchainVersion = Graphics::GetSwapchainVersion();
	if (currentSwapchainVersion == m_storedSwapchainVersion) return false;

	m_storedSwapchainVersion = currentSwapchainVersion;
	const auto& swapChain = Graphics::GetSwapchain();
	const auto& swapChainImageViews = swapChain->GetVkImageViews();
	m_framebuffers.clear();
	for (size_t i = 0; i < swapChainImageViews.size(); i++) {
		const VkImageView attachments[] = { swapChainImageViews[i] };
		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = m_renderPass->GetVkRenderPass();
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = attachments;
		framebufferInfo.width = swapChain->GetImageExtent().width;
		framebufferInfo.height = swapChain->GetImageExtent().height;
		framebufferInfo.layers = 1;
		m_framebuffers.emplace_back(std::make_unique<Framebuffer>(framebufferInfo));
	}

	return true;
}

void RenderLayer::CreateRenderPass()
{
	auto editorLayer = Application::GetLayer<EditorLayer>();
	if (editorLayer)
	{

	}
	else {
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = Graphics::GetSwapchain()->GetImageFormat();
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		m_renderPass = std::make_unique<RenderPass>(renderPassInfo);
	}
}