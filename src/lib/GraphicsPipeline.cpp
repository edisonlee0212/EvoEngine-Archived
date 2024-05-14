#include "GraphicsPipeline.hpp"

#include "Application.hpp"
#include "Shader.hpp"
#include "Console.hpp"
#include "Graphics.hpp"
#include "RenderLayer.hpp"
#include "Utilities.hpp"

using namespace EvoEngine;

void GraphicsPipeline::PreparePipeline()
{
	const auto renderLayer = Application::GetLayer<RenderLayer>();

	std::vector<VkDescriptorSetLayout> setLayouts = {};
	for(const auto& i : m_descriptorSetLayouts)
	{
		setLayouts.push_back(i->GetVkDescriptorSetLayout());
	}
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = setLayouts.size();
	pipelineLayoutInfo.pSetLayouts = setLayouts.data();
	pipelineLayoutInfo.pushConstantRangeCount = m_pushConstantRanges.size();
	pipelineLayoutInfo.pPushConstantRanges = m_pushConstantRanges.data();

	m_pipelineLayout = std::make_unique<PipelineLayout>(pipelineLayoutInfo);

	std::vector<VkPipelineShaderStageCreateInfo> shaderStages {};

	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

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
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
	}
	if (m_tessellationEvaluationShader && m_tessellationEvaluationShader->m_shaderType == ShaderType::TessellationEvaluation && m_tessellationEvaluationShader->Compiled())
	{
		VkPipelineShaderStageCreateInfo tessEvaluationShaderStageInfo{};
		tessEvaluationShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		tessEvaluationShaderStageInfo.stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
		tessEvaluationShaderStageInfo.module = m_tessellationEvaluationShader->m_shaderModule->GetVkShaderModule();
		tessEvaluationShaderStageInfo.pName = "main";
		shaderStages.emplace_back(tessEvaluationShaderStageInfo);
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
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
	if (m_taskShader && m_taskShader->m_shaderType == ShaderType::Task && m_taskShader->Compiled())
	{
		VkPipelineShaderStageCreateInfo taskShaderStageInfo{};
		taskShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		taskShaderStageInfo.stage = VK_SHADER_STAGE_TASK_BIT_EXT;
		taskShaderStageInfo.module = m_taskShader->m_shaderModule->GetVkShaderModule();
		taskShaderStageInfo.pName = "main";
		shaderStages.emplace_back(taskShaderStageInfo);
	}
	if (m_meshShader && m_meshShader->m_shaderType == ShaderType::Mesh && m_meshShader->Compiled())
	{
		VkPipelineShaderStageCreateInfo meshShaderStageInfo{};
		meshShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		meshShaderStageInfo.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
		meshShaderStageInfo.module = m_meshShader->m_shaderModule->GetVkShaderModule();
		meshShaderStageInfo.pName = "main";
		shaderStages.emplace_back(meshShaderStageInfo);
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

	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	VkPipelineTessellationStateCreateInfo tessInfo{};
	tessInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
	tessInfo.patchControlPoints = m_tessellationPatchControlPoints;
	
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


	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;
	std::vector colorBlendAttachments = { m_colorAttachmentFormats.size(), colorBlendAttachment };
	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = colorBlendAttachments.size();
	colorBlending.pAttachments = colorBlendAttachments.data();
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.stencilTestEnable = VK_FALSE;
	std::vector dynamicStates = {
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

	VkPipelineRenderingCreateInfo renderingCreateInfo{};
	renderingCreateInfo.viewMask = m_viewMask;
	renderingCreateInfo.colorAttachmentCount = m_colorAttachmentFormats.size();
	renderingCreateInfo.pColorAttachmentFormats = m_colorAttachmentFormats.data();
	renderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
	renderingCreateInfo.depthAttachmentFormat = m_depthAttachmentFormat;
	renderingCreateInfo.stencilAttachmentFormat = m_stencilAttachmentFormat;

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.stageCount = shaderStages.size();
	pipelineInfo.pTessellationState = &tessInfo;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pStages = shaderStages.data();
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	if (m_meshShader) pipelineInfo.pVertexInputState = nullptr;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = m_pipelineLayout->GetVkPipelineLayout();
	pipelineInfo.pNext = &renderingCreateInfo;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
	Graphics::CheckVk(vkCreateGraphicsPipelines(Graphics::GetVkDevice(), VK_NULL_HANDLE, 1,
		&pipelineInfo, nullptr, &m_vkGraphicsPipeline));
	m_states.ResetAllStates(1);
}


bool GraphicsPipeline::PipelineReady() const
{
	return m_vkGraphicsPipeline != VK_NULL_HANDLE;
}

void GraphicsPipeline::Bind(const VkCommandBuffer commandBuffer)
{
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_vkGraphicsPipeline);
	m_states.ApplyAllStates(commandBuffer, true);
}

void GraphicsPipeline::BindDescriptorSet(VkCommandBuffer commandBuffer, uint32_t firstSet, VkDescriptorSet descriptorSet) const
{
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout->GetVkPipelineLayout(), firstSet, 1, &descriptorSet, 0, nullptr);
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