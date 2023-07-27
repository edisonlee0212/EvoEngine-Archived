#include "GraphicsPipeline.hpp"

#include "Shader.hpp"
#include "Console.hpp"
#include "Graphics.hpp"
#include "RenderLayer.hpp"
#include "Utilities.hpp"

using namespace EvoEngine;


void GraphicsPipeline::CheckDescriptorSetsReady()
{
	for (const auto& binding : m_descriptorSetLayoutBindings)
	{
		if (!binding.m_ready)
		{
			m_descriptorSetsReady = false;
		}
	}
	m_descriptorSetsReady = true;
}

void GraphicsPipeline::ClearDescriptorSets()
{
	m_descriptorSetLayoutBindings.clear();

	m_perFrameLayout.reset();
	m_perPassLayout.reset();
	m_perGroupLayout.reset();
	m_perCommandLayout.reset();

	vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perFrameDescriptorSets.size(), m_perFrameDescriptorSets.data());
	vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perPassDescriptorSets.size(), m_perPassDescriptorSets.data());
	vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perGroupDescriptorSets.size(), m_perGroupDescriptorSets.data());
	vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perCommandDescriptorSets.size(), m_perCommandDescriptorSets.data());

	m_perFrameDescriptorSets.clear();
	m_perPassDescriptorSets.clear();
	m_perGroupDescriptorSets.clear();
	m_perCommandDescriptorSets.clear();

	m_containStandardBindings = false;
	m_layoutReady = false;
	m_descriptorSetsReady = false;
}

void GraphicsPipeline::InitializeStandardBindings()
{
	if (m_containStandardBindings)
	{
		EVOENGINE_ERROR("Already contain standard layout!");
		return;
	}
	if (!m_descriptorSetLayoutBindings.empty())
	{
		EVOENGINE_ERROR("Already contain bindings!");
		return;
	}

	PushDescriptorBinding(DescriptorSetType::PerFrame, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	PushDescriptorBinding(DescriptorSetType::PerFrame, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	PushDescriptorBinding(DescriptorSetType::PerPass, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	PushDescriptorBinding(DescriptorSetType::PerGroup, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);
	PushDescriptorBinding(DescriptorSetType::PerCommand, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL);

	m_containStandardBindings = true;
	m_layoutReady = false;
	m_descriptorSetsReady = false;
}

uint32_t GraphicsPipeline::PushDescriptorBinding(const DescriptorSetType setType, const VkDescriptorType type, const VkShaderStageFlags stageFlags)
{
	VkDescriptorSetLayoutBinding binding{};
	binding.binding = m_descriptorSetLayoutBindings.size();
	binding.descriptorCount = 1;
	binding.descriptorType = type;
	binding.pImmutableSamplers = nullptr;
	binding.stageFlags = stageFlags;
	m_descriptorSetLayoutBindings.push_back({ setType, binding, false });

	m_descriptorSetsReady = false;
	m_layoutReady = false;
	return binding.binding;
}

bool GraphicsPipeline::ContainStandardBindings() const
{
	return m_containStandardBindings;
}

void GraphicsPipeline::PrepareLayouts()
{
	m_perFrameLayout.reset();
	m_perPassLayout.reset();
	m_perGroupLayout.reset();
	m_perCommandLayout.reset();

	if (!m_perFrameDescriptorSets.empty()) vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perFrameDescriptorSets.size(), m_perFrameDescriptorSets.data());
	if (!m_perPassDescriptorSets.empty()) vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perPassDescriptorSets.size(), m_perPassDescriptorSets.data());
	if (!m_perGroupDescriptorSets.empty()) vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perGroupDescriptorSets.size(), m_perGroupDescriptorSets.data());
	if (!m_perCommandDescriptorSets.empty()) vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), m_perCommandDescriptorSets.size(), m_perCommandDescriptorSets.data());

	m_perFrameDescriptorSets.clear();
	m_perPassDescriptorSets.clear();
	m_perGroupDescriptorSets.clear();
	m_perCommandDescriptorSets.clear();

	std::vector<VkDescriptorSetLayoutBinding> perFrameBindings;
	std::vector<VkDescriptorSetLayoutBinding> perPassBindings;
	std::vector<VkDescriptorSetLayoutBinding> perGroupBindings;
	std::vector<VkDescriptorSetLayoutBinding> perCommandBindings;

	for (const auto& binding : m_descriptorSetLayoutBindings)
	{
		switch (binding.m_type)
		{
		case DescriptorSetType::PerFrame:
		{
			perFrameBindings.push_back(binding.m_binding);
		}break;
		case DescriptorSetType::PerPass:
		{
			perPassBindings.push_back(binding.m_binding);
		}break;
		case DescriptorSetType::PerGroup:
		{
			perGroupBindings.push_back(binding.m_binding);
		}break;
		case DescriptorSetType::PerCommand:
		{
			perCommandBindings.push_back(binding.m_binding);
		}break;
		}
	}

	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
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

	VkDescriptorSetLayoutCreateInfo perGroupLayoutInfo{};
	perGroupLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	perGroupLayoutInfo.bindingCount = static_cast<uint32_t>(perGroupBindings.size());
	perGroupLayoutInfo.pBindings = perGroupBindings.data();
	m_perGroupLayout = std::make_unique<DescriptorSetLayout>(perGroupLayoutInfo);

	VkDescriptorSetLayoutCreateInfo perCommandLayoutInfo{};
	perCommandLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	perCommandLayoutInfo.bindingCount = static_cast<uint32_t>(perCommandBindings.size());
	perCommandLayoutInfo.pBindings = perCommandBindings.data();
	m_perCommandLayout = std::make_unique<DescriptorSetLayout>(perCommandLayoutInfo);

	std::vector setLayouts = { m_perFrameLayout->GetVkDescriptorSetLayout(), m_perPassLayout->GetVkDescriptorSetLayout(), m_perGroupLayout->GetVkDescriptorSetLayout(), m_perCommandLayout->GetVkDescriptorSetLayout() };
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = setLayouts.size();
	pipelineLayoutInfo.pSetLayouts = setLayouts.data();
	m_pipelineLayout = std::make_unique<PipelineLayout>(pipelineLayoutInfo);

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = Graphics::GetDescriptorPool()->GetVkDescriptorPool();

	const std::vector perFrameLayouts(maxFramesInFlight, m_perFrameLayout->GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perFrameLayouts.data();
	m_perFrameDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perFrameDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	const std::vector perPassLayout(maxFramesInFlight, m_perPassLayout->GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perPassLayout.data();
	m_perPassDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perPassDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	const std::vector perObjectGroupLayout(maxFramesInFlight, m_perGroupLayout->GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perObjectGroupLayout.data();
	m_perGroupDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perGroupDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}

	const std::vector perObjectLayout(maxFramesInFlight, m_perCommandLayout->GetVkDescriptorSetLayout());
	allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	allocInfo.pSetLayouts = perObjectLayout.data();
	m_perCommandDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, m_perCommandDescriptorSets.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor sets!");
	}
	m_layoutReady = true;
}

void GraphicsPipeline::UpdateStandardBindings()
{
	if (!m_containStandardBindings)
	{
		EVOENGINE_ERROR("No standard layout!");
		return;
	}
	auto& graphics = Graphics::GetInstance();
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	VkDescriptorBufferInfo bufferInfos[5] = {};
	bufferInfos[0].offset = 0;
	bufferInfos[0].range = VK_WHOLE_SIZE;
	bufferInfos[1].offset = 0;
	bufferInfos[1].range = VK_WHOLE_SIZE;
	bufferInfos[2].offset = 0;
	bufferInfos[2].range = VK_WHOLE_SIZE;
	bufferInfos[3].offset = 0;
	bufferInfos[3].range = VK_WHOLE_SIZE;
	bufferInfos[4].offset = 0;
	bufferInfos[4].range = VK_WHOLE_SIZE;
	for (size_t i = 0; i < maxFramesInFlight; i++) {
		bufferInfos[0].buffer = graphics.m_standardDescriptorBuffers[i * 4 + 0]->GetVkBuffer();
		bufferInfos[1].buffer = graphics.m_standardDescriptorBuffers[i * 4 + 1]->GetVkBuffer();
		bufferInfos[2].buffer = graphics.m_standardDescriptorBuffers[i * 4 + 2]->GetVkBuffer();
		bufferInfos[3].buffer = graphics.m_standardDescriptorBuffers[i * 4 + 3]->GetVkBuffer();
		bufferInfos[4].buffer = graphics.m_standardDescriptorBuffers[i * 4 + 4]->GetVkBuffer();

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
		materialInfo.dstSet = m_perGroupDescriptorSets[i];
		materialInfo.dstBinding = 3;
		materialInfo.dstArrayElement = 0;
		materialInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		materialInfo.descriptorCount = 1;
		materialInfo.pBufferInfo = &bufferInfos[3];

		VkWriteDescriptorSet objectInfo{};
		objectInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		objectInfo.dstSet = m_perCommandDescriptorSets[i];
		objectInfo.dstBinding = 4;
		objectInfo.dstArrayElement = 0;
		objectInfo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		objectInfo.descriptorCount = 1;
		objectInfo.pBufferInfo = &bufferInfos[4];

		std::vector writeInfos = { renderInfo, environmentInfo, cameraInfo, materialInfo, objectInfo };
		vkUpdateDescriptorSets(Graphics::GetVkDevice(), writeInfos.size(), writeInfos.data(), 0, nullptr);
	}
	m_descriptorSetLayoutBindings[0].m_ready = true;
	m_descriptorSetLayoutBindings[1].m_ready = true;
	m_descriptorSetLayoutBindings[2].m_ready = true;
	m_descriptorSetLayoutBindings[3].m_ready = true;
	m_descriptorSetLayoutBindings[4].m_ready = true;
	CheckDescriptorSetsReady();
}

void GraphicsPipeline::UpdateBinding(const uint32_t binding, const VkDescriptorBufferInfo& bufferInfo)
{
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		const auto& descriptorBinding = m_descriptorSetLayoutBindings[binding];
		VkWriteDescriptorSet writeInfo{};
		writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		switch (descriptorBinding.m_type)
		{
		case DescriptorSetType::PerFrame:
		{
			writeInfo.dstSet = m_perFrameDescriptorSets[i];
		}break;
		case DescriptorSetType::PerPass:
		{
			writeInfo.dstSet = m_perPassDescriptorSets[i];
		}break;
		case DescriptorSetType::PerGroup:
		{
			writeInfo.dstSet = m_perGroupDescriptorSets[i];
		}break;
		case DescriptorSetType::PerCommand:
		{
			writeInfo.dstSet = m_perCommandDescriptorSets[i];
		}break;
		default: break;
		}
		writeInfo.dstBinding = binding;
		writeInfo.dstArrayElement = 0;
		writeInfo.descriptorType = descriptorBinding.m_binding.descriptorType;
		writeInfo.descriptorCount = descriptorBinding.m_binding.descriptorCount;
		writeInfo.pBufferInfo = &bufferInfo;
		vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
	}
	m_descriptorSetLayoutBindings[binding].m_ready = true;
	CheckDescriptorSetsReady();
}

void GraphicsPipeline::UpdateBinding(const uint32_t binding, const VkDescriptorImageInfo& imageInfo)
{
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		const auto& descriptorBinding = m_descriptorSetLayoutBindings[binding];
		VkWriteDescriptorSet writeInfo{};
		writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		switch (descriptorBinding.m_type)
		{
		case DescriptorSetType::PerFrame:
		{
			writeInfo.dstSet = m_perFrameDescriptorSets[i];
		}break;
		case DescriptorSetType::PerPass:
		{
			writeInfo.dstSet = m_perPassDescriptorSets[i];
		}break;
		case DescriptorSetType::PerGroup:
		{
			writeInfo.dstSet = m_perGroupDescriptorSets[i];
		}break;
		case DescriptorSetType::PerCommand:
		{
			writeInfo.dstSet = m_perCommandDescriptorSets[i];
		}break;
		default: break;
		}
		writeInfo.dstBinding = binding;
		writeInfo.dstArrayElement = 0;
		writeInfo.descriptorType = descriptorBinding.m_binding.descriptorType;
		writeInfo.descriptorCount = descriptorBinding.m_binding.descriptorCount;
		writeInfo.pImageInfo = &imageInfo;
		vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
	}
	m_descriptorSetLayoutBindings[binding].m_ready = true;
	CheckDescriptorSetsReady();
}

bool GraphicsPipeline::LayoutSetsReady() const
{
	return m_layoutReady;
}

bool GraphicsPipeline::DescriptorSetsReady() const
{
	return m_descriptorSetsReady;
}

void GraphicsPipeline::PreparePipeline()
{
	std::vector<VkPipelineShaderStageCreateInfo> shaderStages {};
	if (m_vertexShader && m_vertexShader->m_shaderType == ShaderType::Vertex && m_vertexShader->Compiled())
	{
		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = m_vertexShader->m_shaderModule->GetVkShaderModule();
		vertShaderStageInfo.pName = "main";
		shaderStages.emplace_back(vertShaderStageInfo);
	}
	if (m_tessellationControlShader && m_tessellationControlShader->m_shaderType == ShaderType::TessellationControl && m_tessellationControlShader->Compiled())
	{
		VkPipelineShaderStageCreateInfo tessControlShaderStageInfo{};
		tessControlShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		tessControlShaderStageInfo.stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
		tessControlShaderStageInfo.module = m_tessellationControlShader->m_shaderModule->GetVkShaderModule();
		tessControlShaderStageInfo.pName = "main";
		shaderStages.emplace_back(tessControlShaderStageInfo);
	}
	if (m_tessellationEvaluationShader && m_tessellationEvaluationShader->m_shaderType == ShaderType::TessellationEvaluation && m_tessellationEvaluationShader->Compiled())
	{
		VkPipelineShaderStageCreateInfo tessEvaluationShaderStageInfo{};
		tessEvaluationShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		tessEvaluationShaderStageInfo.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
		tessEvaluationShaderStageInfo.module = m_tessellationEvaluationShader->m_shaderModule->GetVkShaderModule();
		tessEvaluationShaderStageInfo.pName = "main";
		shaderStages.emplace_back(tessEvaluationShaderStageInfo);
	}
	if (m_geometryShader && m_geometryShader->m_shaderType == ShaderType::Geometry && m_geometryShader->Compiled())
	{
		VkPipelineShaderStageCreateInfo geometryShaderStageInfo{};
		geometryShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		geometryShaderStageInfo.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
		geometryShaderStageInfo.module = m_geometryShader->m_shaderModule->GetVkShaderModule();
		geometryShaderStageInfo.pName = "main";
		shaderStages.emplace_back(geometryShaderStageInfo);
	}
	if (m_fragmentShader && m_fragmentShader->m_shaderType == ShaderType::Fragment && m_fragmentShader->Compiled())
	{
		VkPipelineShaderStageCreateInfo fragmentShaderStageInfo{};
		fragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragmentShaderStageInfo.module = m_fragmentShader->m_shaderModule->GetVkShaderModule();
		fragmentShaderStageInfo.pName = "main";
		shaderStages.emplace_back(fragmentShaderStageInfo);
	}

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	auto& bindingDescription = IGeometry::GetVertexBindingDescriptions(m_geometryType);
	auto& attributeDescriptions = IGeometry::GetVertexAttributeDescriptions(m_geometryType);
	vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescription.size());;
	vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
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
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	/*
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
	*/
	std::vector<VkDynamicState> dynamicStates = {
		VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT,

		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,

		VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT,
		VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
		VK_DYNAMIC_STATE_POLYGON_MODE_EXT,
		VK_DYNAMIC_STATE_CULL_MODE,
		VK_DYNAMIC_STATE_FRONT_FACE,
		VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
		VK_DYNAMIC_STATE_DEPTH_BIAS,
		VK_DYNAMIC_STATE_LINE_WIDTH,

		VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
		VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
		VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
		VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
		VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
		VK_DYNAMIC_STATE_STENCIL_OP,
		VK_DYNAMIC_STATE_DEPTH_BOUNDS,

		
		VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT,
		VK_DYNAMIC_STATE_LOGIC_OP_EXT,
		VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
		VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
		VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,
		VK_DYNAMIC_STATE_BLEND_CONSTANTS
	};
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = shaderStages.size();
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pStages = shaderStages.data();
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pMultisampleState = &multisampling;
	//pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = m_pipelineLayout->GetVkPipelineLayout();
	//pipelineInfo.renderPass = renderPass->GetVkRenderPass();
	//pipelineInfo.subpass = subpassIndex;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

	Graphics::CheckVk(vkCreateGraphicsPipelines(Graphics::GetVkDevice(), VK_NULL_HANDLE, 1,
		&pipelineInfo, nullptr, &m_vkGraphicsPipeline));
}

bool GraphicsPipeline::PipelineReady() const
{
	return m_vkGraphicsPipeline != VK_NULL_HANDLE;;
}

#pragma region Pipeline Data
void PipelineShaderStage::Apply(const VkPipelineShaderStageCreateInfo& vkPipelineShaderStageCreateInfo)
{
	m_flags = vkPipelineShaderStageCreateInfo.flags;
	m_stage = vkPipelineShaderStageCreateInfo.stage;
	m_module = vkPipelineShaderStageCreateInfo.module;
	m_name = vkPipelineShaderStageCreateInfo.pName;
	if (vkPipelineShaderStageCreateInfo.pSpecializationInfo != VK_NULL_HANDLE)
	{
		m_specializationInfo = *vkPipelineShaderStageCreateInfo.pSpecializationInfo;
	}
}

void PipelineVertexInputState::Apply(const VkPipelineVertexInputStateCreateInfo& vkPipelineShaderStageCreateInfo)
{
	m_flags = vkPipelineShaderStageCreateInfo.flags;
	IGraphicsResource::ApplyVector(m_vertexBindingDescriptions, vkPipelineShaderStageCreateInfo.vertexBindingDescriptionCount, vkPipelineShaderStageCreateInfo.pVertexBindingDescriptions);
	IGraphicsResource::ApplyVector(m_vertexAttributeDescriptions, vkPipelineShaderStageCreateInfo.vertexAttributeDescriptionCount, vkPipelineShaderStageCreateInfo.pVertexAttributeDescriptions);
}

void PipelineInputAssemblyState::Apply(
	const VkPipelineInputAssemblyStateCreateInfo& vkPipelineInputAssemblyStateCreateInfo)
{
	m_flags = vkPipelineInputAssemblyStateCreateInfo.flags;
	m_topology = vkPipelineInputAssemblyStateCreateInfo.topology;
	m_primitiveRestartEnable = vkPipelineInputAssemblyStateCreateInfo.primitiveRestartEnable;
}

void PipelineTessellationState::Apply(const VkPipelineTessellationStateCreateInfo& vkPipelineTessellationStateCreateInfo)
{
	m_flags = vkPipelineTessellationStateCreateInfo.flags;
	m_patchControlPoints = vkPipelineTessellationStateCreateInfo.patchControlPoints;
}

void PipelineViewportState::Apply(const VkPipelineViewportStateCreateInfo& vkPipelineViewportStateCreateInfo)
{
	m_flags = vkPipelineViewportStateCreateInfo.flags;
	IGraphicsResource::ApplyVector(m_viewports, vkPipelineViewportStateCreateInfo.viewportCount, vkPipelineViewportStateCreateInfo.pViewports);
	IGraphicsResource::ApplyVector(m_scissors, vkPipelineViewportStateCreateInfo.scissorCount, vkPipelineViewportStateCreateInfo.pScissors);
}

void PipelineRasterizationState::Apply(
	const VkPipelineRasterizationStateCreateInfo& vkPipelineRasterizationStateCreateInfo)
{
	m_flags = vkPipelineRasterizationStateCreateInfo.flags;
	m_depthClampEnable = vkPipelineRasterizationStateCreateInfo.depthClampEnable;
	m_rasterizerDiscardEnable = vkPipelineRasterizationStateCreateInfo.rasterizerDiscardEnable;
	m_polygonMode = vkPipelineRasterizationStateCreateInfo.polygonMode;
	m_cullMode = vkPipelineRasterizationStateCreateInfo.cullMode;
	m_frontFace = vkPipelineRasterizationStateCreateInfo.frontFace;
	m_depthBiasEnable = vkPipelineRasterizationStateCreateInfo.depthBiasEnable;
	m_depthBiasConstantFactor = vkPipelineRasterizationStateCreateInfo.depthBiasConstantFactor;
	m_depthBiasClamp = vkPipelineRasterizationStateCreateInfo.depthBiasClamp;
	m_depthBiasSlopeFactor = vkPipelineRasterizationStateCreateInfo.depthBiasSlopeFactor;
	m_lineWidth = vkPipelineRasterizationStateCreateInfo.lineWidth;
}

void PipelineMultisampleState::Apply(const VkPipelineMultisampleStateCreateInfo& vkPipelineMultisampleStateCreateInfo)
{
	m_flags = vkPipelineMultisampleStateCreateInfo.flags;
	m_rasterizationSamples = vkPipelineMultisampleStateCreateInfo.rasterizationSamples;
	m_sampleShadingEnable = vkPipelineMultisampleStateCreateInfo.sampleShadingEnable;
	m_minSampleShading = vkPipelineMultisampleStateCreateInfo.minSampleShading;
	if (vkPipelineMultisampleStateCreateInfo.pSampleMask) m_sampleMask = *vkPipelineMultisampleStateCreateInfo.pSampleMask;
	m_alphaToCoverageEnable = vkPipelineMultisampleStateCreateInfo.alphaToCoverageEnable;
	m_alphaToOneEnable = vkPipelineMultisampleStateCreateInfo.alphaToOneEnable;
}

void PipelineDepthStencilState::Apply(
	const VkPipelineDepthStencilStateCreateInfo& vkPipelineDepthStencilStateCreateInfo)
{
	m_flags = vkPipelineDepthStencilStateCreateInfo.flags;
	m_depthTestEnable = vkPipelineDepthStencilStateCreateInfo.depthTestEnable;
	m_depthWriteEnable = vkPipelineDepthStencilStateCreateInfo.depthWriteEnable;
	m_depthCompareOp = vkPipelineDepthStencilStateCreateInfo.depthCompareOp;
	m_depthBoundsTestEnable = vkPipelineDepthStencilStateCreateInfo.depthBoundsTestEnable;
	m_stencilTestEnable = vkPipelineDepthStencilStateCreateInfo.stencilTestEnable;
	m_front = vkPipelineDepthStencilStateCreateInfo.front;
	m_back = vkPipelineDepthStencilStateCreateInfo.back;
	m_minDepthBounds = vkPipelineDepthStencilStateCreateInfo.minDepthBounds;
	m_maxDepthBounds = vkPipelineDepthStencilStateCreateInfo.maxDepthBounds;
}

void PipelineColorBlendState::Apply(const VkPipelineColorBlendStateCreateInfo& vkPipelineColorBlendStateCreateInfo)
{
	m_flags = vkPipelineColorBlendStateCreateInfo.flags;
	m_logicOpEnable = vkPipelineColorBlendStateCreateInfo.logicOpEnable;
	m_logicOp = vkPipelineColorBlendStateCreateInfo.logicOp;
	IGraphicsResource::ApplyVector(m_attachments, vkPipelineColorBlendStateCreateInfo.attachmentCount, vkPipelineColorBlendStateCreateInfo.pAttachments);
	m_blendConstants[0] = vkPipelineColorBlendStateCreateInfo.blendConstants[0];
	m_blendConstants[1] = vkPipelineColorBlendStateCreateInfo.blendConstants[1];
	m_blendConstants[2] = vkPipelineColorBlendStateCreateInfo.blendConstants[2];
	m_blendConstants[3] = vkPipelineColorBlendStateCreateInfo.blendConstants[3];
}

void PipelineDynamicState::Apply(const VkPipelineDynamicStateCreateInfo& vkPipelineDynamicStateCreateInfo)
{
	m_flags = vkPipelineDynamicStateCreateInfo.flags;
	IGraphicsResource::ApplyVector(m_dynamicStates, vkPipelineDynamicStateCreateInfo.dynamicStateCount, vkPipelineDynamicStateCreateInfo.pDynamicStates);
}
#pragma endregion