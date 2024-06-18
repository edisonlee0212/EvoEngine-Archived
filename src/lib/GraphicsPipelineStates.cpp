#include "GraphicsPipelineStates.hpp"

using namespace EvoEngine;



void GraphicsPipelineStates::ResetAllStates(const size_t colorAttachmentSize)
{
	m_viewPort = {};
	m_viewPort.width = 1;
	m_viewPort.height = 1;
	m_scissor = {};
	m_depthClamp = false;
	m_rasterizerDiscard = false;
	m_polygonMode = VK_POLYGON_MODE_FILL;
	m_cullMode = VK_CULL_MODE_BACK_BIT;
	m_frontFace = VK_FRONT_FACE_CLOCKWISE;
	m_depthBias = false;
	m_depthBiasConstantClampSlope = glm::vec3(0.0f);
	m_lineWidth = 1.0f;
	m_depthTest = true;
	m_depthWrite = true;
	m_depthCompare = VK_COMPARE_OP_LESS;
	m_depthBoundTest = false;
	m_minMaxDepthBound = glm::vec2(0.0f, 1.0f);
	m_stencilTest = false;
	m_stencilFaceMask = VK_STENCIL_FACE_FRONT_BIT;
	m_stencilFailOp = VK_STENCIL_OP_ZERO;
	m_stencilPassOp = VK_STENCIL_OP_ZERO;
	m_stencilDepthFailOp = VK_STENCIL_OP_ZERO;
	m_stencilCompareOp = VK_COMPARE_OP_LESS;

	m_logicOpEnable = VK_FALSE;
	m_logicOp = VK_LOGIC_OP_COPY;
	m_colorBlendAttachmentStates.clear();
	m_colorBlendAttachmentStates.resize(std::max(static_cast<size_t>(1), colorAttachmentSize));
	for (auto& i : m_colorBlendAttachmentStates)
	{
		i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		i.blendEnable = VK_FALSE;
		i.colorBlendOp = i.alphaBlendOp = VK_BLEND_OP_ADD;

		i.srcColorBlendFactor = i.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		i.dstColorBlendFactor = i.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	}
	m_blendConstants[0] = 0;
	m_blendConstants[1] = 0;
	m_blendConstants[2] = 0;
	m_blendConstants[3] = 0;
}

void GraphicsPipelineStates::ApplyAllStates(const VkCommandBuffer commandBuffer, const bool forceSet)
{
	if (forceSet
		|| m_viewPortApplied.x != m_viewPort.x
		|| m_viewPortApplied.y != m_viewPort.y
		|| m_viewPortApplied.width != m_viewPort.width
		|| m_viewPortApplied.height != m_viewPort.height
		|| m_viewPortApplied.maxDepth != m_viewPort.maxDepth
		|| m_viewPortApplied.minDepth != m_viewPort.minDepth) {
		m_viewPortApplied = m_viewPort;
		m_viewPortApplied.width = m_viewPort.width = glm::max(1.0f, m_viewPort.width);
		m_viewPortApplied.height = m_viewPort.height = glm::max(1.0f, m_viewPort.height);
		vkCmdSetViewport(commandBuffer, 0, 1, &m_viewPortApplied);

	}
	if (forceSet
		|| m_scissorApplied.extent.height != m_scissor.extent.height
		|| m_scissorApplied.extent.width != m_scissor.extent.width
		|| m_scissorApplied.offset.x != m_scissor.offset.x
		|| m_scissorApplied.offset.y != m_scissor.offset.y) {
		m_scissorApplied = m_scissor;
		vkCmdSetScissor(commandBuffer, 0, 1, &m_scissorApplied);
	}

	if (forceSet || m_depthClampApplied != m_depthClamp) {
		m_depthClampApplied = m_depthClamp;
		vkCmdSetDepthClampEnableEXT(commandBuffer, m_depthClampApplied);
	}
	if (forceSet || m_rasterizerDiscardApplied != m_rasterizerDiscard) {
		m_rasterizerDiscardApplied = m_rasterizerDiscard;
		vkCmdSetRasterizerDiscardEnable(commandBuffer, m_rasterizerDiscardApplied);
	}
	if (forceSet || m_polygonModeApplied != m_polygonMode) {
		m_polygonModeApplied = m_polygonMode;
		vkCmdSetPolygonModeEXT(commandBuffer, m_polygonModeApplied);
	}
	if (forceSet || m_cullModeApplied != m_cullMode) {
		m_cullModeApplied = m_cullMode;
		vkCmdSetCullModeEXT(commandBuffer, m_cullModeApplied);
	}
	if (forceSet || m_frontFaceApplied != m_frontFace) {
		m_frontFaceApplied = m_frontFace;
		vkCmdSetFrontFace(commandBuffer, m_frontFaceApplied);
	}
	if (forceSet || m_depthBiasApplied != m_depthBias) {
		m_depthBiasApplied = m_depthBias;
		vkCmdSetDepthBiasEnable(commandBuffer, m_depthBiasApplied);
	}
	if (forceSet || m_depthBiasConstantClampSlopeApplied != m_depthBiasConstantClampSlope) {
		m_depthBiasConstantClampSlopeApplied = m_depthBiasConstantClampSlope;
		vkCmdSetDepthBias(commandBuffer, m_depthBiasConstantClampSlopeApplied.x, m_depthBiasConstantClampSlopeApplied.y, m_depthBiasConstantClampSlopeApplied.z);
	}
	if (forceSet || m_lineWidthApplied != m_lineWidth) {
		m_lineWidthApplied = m_lineWidth;
		vkCmdSetLineWidth(commandBuffer, m_lineWidthApplied);
	}
	if (forceSet || m_depthTestApplied != m_depthTest) {
		m_depthTestApplied = m_depthTest;
		vkCmdSetDepthTestEnableEXT(commandBuffer, m_depthTestApplied);
	}
	if (forceSet || m_depthWriteApplied != m_depthWrite) {
		m_depthWriteApplied = m_depthWrite;
		vkCmdSetDepthWriteEnableEXT(commandBuffer, m_depthWriteApplied);
	}
	if (forceSet || m_depthCompareApplied != m_depthCompare) {
		m_depthCompareApplied = m_depthCompare;
		vkCmdSetDepthCompareOpEXT(commandBuffer, m_depthCompareApplied);
	}
	if (forceSet || m_depthBoundTestApplied != m_depthBoundTest) {
		m_depthBoundTestApplied = m_depthBoundTest;
		vkCmdSetDepthBoundsTestEnableEXT(commandBuffer, m_depthBoundTestApplied);
	}
	if (forceSet || m_minMaxDepthBoundApplied != m_minMaxDepthBound) {
		m_minMaxDepthBoundApplied = m_minMaxDepthBound;
		vkCmdSetDepthBounds(commandBuffer, m_minMaxDepthBoundApplied.x, m_minMaxDepthBoundApplied.y);
	}
	if (forceSet || m_stencilTestApplied != m_stencilTest) {
		m_stencilTestApplied = m_stencilTest;
		vkCmdSetStencilTestEnableEXT(commandBuffer, m_stencilTestApplied);
	}
	if (forceSet ||
		m_frontFaceApplied != m_frontFace
		|| m_stencilFailOpApplied != m_stencilFailOp
		|| m_stencilPassOpApplied != m_stencilPassOp
		|| m_stencilDepthFailOpApplied != m_stencilDepthFailOp
		|| m_stencilCompareOpApplied != m_stencilCompareOp) {
		m_stencilFaceMaskApplied = m_stencilFaceMask;
		m_stencilFailOpApplied = m_stencilFailOp;
		m_stencilPassOpApplied = m_stencilPassOp;
		m_stencilDepthFailOpApplied = m_stencilDepthFailOp;
		m_stencilCompareOpApplied = m_stencilCompareOp;
		vkCmdSetStencilOpEXT(commandBuffer, m_stencilFaceMaskApplied, m_stencilFailOpApplied, m_stencilPassOpApplied, m_stencilDepthFailOpApplied, m_stencilCompareOpApplied);
	}

	if (forceSet || m_logicOpEnableApplied != m_logicOpEnable)
	{
		vkCmdSetLogicOpEnableEXT(commandBuffer, m_logicOpEnableApplied);
	}

	if (forceSet || m_logicOpApplied != m_logicOp)
	{
		vkCmdSetLogicOpEXT(commandBuffer, m_logicOpApplied);
	}
	if(m_colorBlendAttachmentStates.empty())
	{
		m_colorBlendAttachmentStates.emplace_back();
	}
	std::vector<VkBool32> colorWriteMasks = {};
	colorWriteMasks.reserve(m_colorBlendAttachmentStates.size());
	for (const auto& i : m_colorBlendAttachmentStates)
	{
		colorWriteMasks.push_back(i.colorWriteMask);
	}
	if (!colorWriteMasks.empty()) vkCmdSetColorWriteMaskEXT(commandBuffer, 0, colorWriteMasks.size(), colorWriteMasks.data());

	std::vector<VkBool32> colorBlendEnable = {};
	colorBlendEnable.reserve(m_colorBlendAttachmentStates.size());
	for (const auto& i : m_colorBlendAttachmentStates)
	{
		colorBlendEnable.push_back(i.blendEnable);
	}
	if (!colorBlendEnable.empty()) vkCmdSetColorBlendEnableEXT(commandBuffer, 0, colorBlendEnable.size(), colorBlendEnable.data());

	std::vector<VkColorBlendEquationEXT> equations{};
	for (const auto& i : m_colorBlendAttachmentStates)
	{
		VkColorBlendEquationEXT equation;
		equation.srcColorBlendFactor = i.srcColorBlendFactor;
		equation.dstColorBlendFactor = i.dstColorBlendFactor;
		equation.colorBlendOp = i.colorBlendOp;
		equation.srcAlphaBlendFactor = i.srcAlphaBlendFactor;
		equation.dstAlphaBlendFactor = i.dstAlphaBlendFactor;
		equation.alphaBlendOp = i.alphaBlendOp;
		equations.emplace_back(equation);
	}
	if (!equations.empty()) vkCmdSetColorBlendEquationEXT(commandBuffer, 0, equations.size(), equations.data());
	else
	{
		int a = 0;
	}
	vkCmdSetBlendConstants(commandBuffer, m_blendConstantsApplied);
}
