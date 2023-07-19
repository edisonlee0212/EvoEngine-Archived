#include "RenderLayer.hpp"
#include "Application.hpp"
#include "Graphics.hpp"
#include "Utilities.hpp"
#include "WindowLayer.hpp"
#include "ProjectManager.hpp"
using namespace EvoEngine;
const std::vector<Vertex> vertices = {
	{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<glm::uvec3> indices = {
	{0, 1, 2}, {2, 3, 0}
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

	m_perFrameLayout.Create(perFrameLayoutInfo);

	VkDescriptorSetLayoutCreateInfo perPassLayoutInfo{};
	perPassLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	perPassLayoutInfo.bindingCount = static_cast<uint32_t>(perPassBindings.size());
	perPassLayoutInfo.pBindings = perPassBindings.data();

	m_perPassLayout.Create(perPassLayoutInfo);

	VkDescriptorSetLayoutCreateInfo perObjectGroupLayoutInfo{};
	perObjectGroupLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	perObjectGroupLayoutInfo.bindingCount = static_cast<uint32_t>(perObjectGroupBindings.size());
	perObjectGroupLayoutInfo.pBindings = perObjectGroupBindings.data();

	m_perObjectGroupLayout.Create(perObjectGroupLayoutInfo);

	size_t bindingsSize = perFrameBindings.size() + perPassBindings.size() + perObjectGroupBindings.size();
	VkDescriptorPoolSize renderLayerDescriptorPoolSize{};
	renderLayerDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	renderLayerDescriptorPoolSize.descriptorCount =
		static_cast<uint32_t>(maxFramesInFlight * bindingsSize);

	VkDescriptorPoolCreateInfo renderLayerDescriptorPoolInfo{};
	renderLayerDescriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	renderLayerDescriptorPoolInfo.poolSizeCount = 1;
	renderLayerDescriptorPoolInfo.pPoolSizes = &renderLayerDescriptorPoolSize;
	renderLayerDescriptorPoolInfo.maxSets = static_cast<uint32_t>(maxFramesInFlight * 3);
	m_descriptorPool.Create(renderLayerDescriptorPoolInfo);

	std::vector perFrameLayouts(maxFramesInFlight, m_perFrameLayout.GetVkDescriptorSetLayout());
	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = m_descriptorPool.GetVkDescriptorPool();
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perFrameLayouts.data();
	m_perFrameDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perFrameDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	std::vector perPassLayout(maxFramesInFlight, m_perPassLayout.GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perPassLayout.data();
	m_perPassDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perPassDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	std::vector perObjectGroupLayout(maxFramesInFlight, m_perObjectGroupLayout.GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perObjectGroupLayout.data();
	m_perObjectGroupDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perObjectGroupDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	VkDescriptorBufferInfo bufferInfos[4] = {};
	bufferInfos[0].offset = 0;
	bufferInfos[0].range = sizeof(RenderInfoBlock);
	bufferInfos[1].offset = 0;
	bufferInfos[1].range = sizeof(EnvironmentInfoBlock);
	bufferInfos[2].offset = 0;
	bufferInfos[2].range = sizeof(CameraInfoBlock);
	bufferInfos[3].offset = 0;
	bufferInfos[3].range = sizeof(MaterialInfoBlock);
	m_descriptorBuffers.resize(maxFramesInFlight * bindingsSize);

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
		m_descriptorBuffers[i * 4 + 0].Create(bufferCreateInfo, bufferVmaAllocationCreateInfo);
		bufferCreateInfo.size = sizeof(EnvironmentInfoBlock);
		m_descriptorBuffers[i * 4 + 1].Create(bufferCreateInfo, bufferVmaAllocationCreateInfo);
		bufferCreateInfo.size = sizeof(CameraInfoBlock);
		m_descriptorBuffers[i * 4 + 2].Create(bufferCreateInfo, bufferVmaAllocationCreateInfo);
		bufferCreateInfo.size = sizeof(MaterialInfoBlock);
		m_descriptorBuffers[i * 4 + 3].Create(bufferCreateInfo, bufferVmaAllocationCreateInfo);

		bufferInfos[0].buffer = m_descriptorBuffers[i * 4 + 0].GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), m_descriptorBuffers[i * 4 + 0].GetVmaAllocation(), &m_renderInfoBlockMemory[i]);

		bufferInfos[1].buffer = m_descriptorBuffers[i * 4 + 1].GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), m_descriptorBuffers[i * 4 + 0].GetVmaAllocation(), &m_environmentalInfoBlockMemory[i]);

		bufferInfos[2].buffer = m_descriptorBuffers[i * 4 + 2].GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), m_descriptorBuffers[i * 4 + 0].GetVmaAllocation(), &m_cameraInfoBlockMemory[i]);

		bufferInfos[3].buffer = m_descriptorBuffers[i * 4 + 3].GetVkBuffer();
		vmaMapMemory(Graphics::GetVmaAllocator(), m_descriptorBuffers[i * 4 + 0].GetVmaAllocation(), &m_materialInfoBlockMemory[i]);


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
	CreateGraphicsPipeline();
	CreateFramebuffers();

	m_mesh = ProjectManager::CreateTemporaryAsset<Mesh>();
	m_mesh->SetVertices({}, vertices, indices);
}

void RenderLayer::OnDestroy()
{
	m_descriptorPool.Destroy();
	m_perObjectGroupLayout.Destroy();
	m_perPassLayout.Destroy();
	m_perFrameLayout.Destroy();


	m_graphicsPipeline.Destroy();
	m_pipelineLayout.Destroy();
	m_renderPass.Destroy();
}

void RenderLayer::PreUpdate()
{
	const auto currentSwapchainVersion = Graphics::GetSwapchainVersion();
	if(currentSwapchainVersion != m_storedSwapchainVersion)
	{
		CreateFramebuffers();
		m_storedSwapchainVersion = currentSwapchainVersion;
	}
}

void RenderLayer::Update()
{
}

void RenderLayer::LateUpdate()
{
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	memcpy(m_renderInfoBlockMemory[currentFrameIndex], &m_renderInfoBlock, sizeof(EnvironmentInfoBlock));

	const auto& windowLayer = Application::GetLayer<WindowLayer>();
	if (!windowLayer || windowLayer->m_windowSize.x == 0 || windowLayer->m_windowSize.y == 0) return;
	RecordCommandBuffer();
}

void RenderLayer::RecordCommandBuffer()
{
	auto commandBuffer = Graphics::GetCurrentVkCommandBuffer();
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("failed to begin recording command buffer!");
	}
	const auto extent2D = Graphics::GetSwapchain().GetVkExtent2D();
	VkRenderPassBeginInfo renderPassBeginInfo{};
	renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.renderPass = m_renderPass.GetVkRenderPass();
	renderPassBeginInfo.framebuffer = m_framebuffers[Graphics::GetNextImageIndex()].GetVkFrameBuffer();
	renderPassBeginInfo.renderArea.offset = { 0, 0 };
	renderPassBeginInfo.renderArea.extent = extent2D;

	VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
	renderPassBeginInfo.clearValueCount = 1;
	renderPassBeginInfo.pClearValues = &clearColor;

	vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline.GetVkPipeline());

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = static_cast<float>(extent2D.width);
	viewport.height = static_cast<float>(extent2D.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset = { 0, 0 };
	scissor.extent = extent2D;
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

	m_mesh->SubmitDrawIndexed(commandBuffer);

	vkCmdEndRenderPass(commandBuffer);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("failed to record command buffer!");
	}
}


void RenderLayer::CreateRenderPass()
{
#pragma region RenderPass
	
	VkAttachmentDescription colorAttachment{};
	colorAttachment.format = Graphics::GetVkSurfaceFormat().format;
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

	m_renderPass.Create(renderPassInfo);
#pragma endregion
}

void RenderLayer::CreateGraphicsPipeline()
{
#pragma region Pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 0;
	pipelineLayoutInfo.pushConstantRangeCount = 0;
	m_pipelineLayout.Create(pipelineLayoutInfo);
#pragma endregion
#pragma region Graphics Pipeline
	ShaderModule vertShader, fragShader;

	vertShader.Create(shaderc_vertex_shader,
		FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/shader.vert"));
	fragShader.Create(shaderc_fragment_shader,
		FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/shader.frag"));

	VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStageInfo.module = vertShader.GetVkShaderModule();
	vertShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStageInfo.module = fragShader.GetVkShaderModule();
	fragShaderStageInfo.pName = "main";

	std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
	shaderStages.push_back(vertShaderStageInfo);
	shaderStages.push_back(fragShaderStageInfo);

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

	auto bindingDescription = Vertex::GetBindingDescription();
	auto attributeDescriptions = Vertex::GetAttributeDescriptions();

	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;

	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	std::vector<VkDynamicState> dynamicStates = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages.data();
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = m_pipelineLayout.GetVkPipelineLayout();
	pipelineInfo.renderPass = m_renderPass.GetVkRenderPass();
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
	m_graphicsPipeline.Create(pipelineInfo);

	vertShader.Destroy();
	fragShader.Destroy();
#pragma endregion
}

void RenderLayer::CreateFramebuffers()
{
	const auto& swapChain = Graphics::GetSwapchain();
	const auto& swapChainImageViews = swapChain.GetVkImageViews();
	m_framebuffers.resize(swapChainImageViews.size());
	for (size_t i = 0; i < swapChainImageViews.size(); i++) {
		const VkImageView attachments[] = { swapChainImageViews[i] };
		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = m_renderPass.GetVkRenderPass();
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = attachments;
		framebufferInfo.width = swapChain.GetVkExtent2D().width;
		framebufferInfo.height = swapChain.GetVkExtent2D().height;
		framebufferInfo.layers = 1;
		m_framebuffers[i].Create(framebufferInfo);
	}
}