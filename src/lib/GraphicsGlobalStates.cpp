#include "GraphicsGlobalStates.hpp"

using namespace EvoEngine;



void GraphicsGlobalStates::ResetAllStates(const VkCommandBuffer commandBuffer, size_t colorAttachmentSize)
{
	m_viewPort = {};
	m_viewPort.width = 1;
	m_viewPort.height = 1;
	m_scissor = {};
	m_patchControlPoints = 1;
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
	m_colorBlendAttachmentStates.resize(colorAttachmentSize);
	for(auto& i : m_colorBlendAttachmentStates)
	{
		i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		i.blendEnable = VK_FALSE;
	}
	m_blendConstants[0] = 0;
	m_blendConstants[1] = 0;
	m_blendConstants[2] = 0;
	m_blendConstants[3] = 0;
}

void GraphicsGlobalStates::ApplyAllStates(const VkCommandBuffer commandBuffer, const bool forceSet)
{
	m_viewPortApplied = m_viewPort;
	m_viewPortApplied.width = glm::max(1.0f, m_viewPort.width);
	m_viewPortApplied.height = glm::max(1.0f, m_viewPort.height);
	m_scissorApplied = m_scissor;
	vkCmdSetViewport(commandBuffer, 0, 1, &m_viewPortApplied);
	vkCmdSetScissor(commandBuffer, 0, 1, &m_scissorApplied);
	if (forceSet || m_patchControlPointsApplied != m_patchControlPoints) {
		m_patchControlPointsApplied = m_patchControlPoints;
		vkCmdSetPatchControlPointsEXT(commandBuffer, m_patchControlPointsApplied);
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

	bool updateBlendingStates = false;
	if (m_colorBlendAttachmentStates.size() != m_colorBlendAttachmentStatesApplied.size())
	{
		updateBlendingStates = true;
	}
	else
	{
		for (int i = 0; i < m_colorBlendAttachmentStates.size(); i++)
		{
			if (m_colorBlendAttachmentStates[i].blendEnable != m_colorBlendAttachmentStatesApplied[i].blendEnable
				|| m_colorBlendAttachmentStates[i].srcColorBlendFactor != m_colorBlendAttachmentStatesApplied[i].srcColorBlendFactor
				|| m_colorBlendAttachmentStates[i].dstColorBlendFactor != m_colorBlendAttachmentStatesApplied[i].dstColorBlendFactor
				|| m_colorBlendAttachmentStates[i].colorBlendOp != m_colorBlendAttachmentStatesApplied[i].colorBlendOp
				|| m_colorBlendAttachmentStates[i].srcAlphaBlendFactor != m_colorBlendAttachmentStatesApplied[i].srcAlphaBlendFactor
				|| m_colorBlendAttachmentStates[i].dstAlphaBlendFactor != m_colorBlendAttachmentStatesApplied[i].dstAlphaBlendFactor
				|| m_colorBlendAttachmentStates[i].alphaBlendOp != m_colorBlendAttachmentStatesApplied[i].alphaBlendOp
				|| m_colorBlendAttachmentStates[i].colorWriteMask != m_colorBlendAttachmentStatesApplied[i].colorWriteMask)
			{
				updateBlendingStates = true;
				break;
			}
		}
	}
	if (forceSet || updateBlendingStates)
	{
		m_colorBlendAttachmentStatesApplied = m_colorBlendAttachmentStates;
	}
	if (forceSet || updateBlendingStates)
	{
		std::vector<VkBool32> colorWriteMasks = {};
		colorWriteMasks.reserve(m_colorBlendAttachmentStatesApplied.size());
		for (const auto& i : m_colorBlendAttachmentStatesApplied)
		{
			colorWriteMasks.push_back(i.colorWriteMask);
		}
		if(!colorWriteMasks.empty()) vkCmdSetColorWriteMaskEXT(commandBuffer, 0, colorWriteMasks.size(), colorWriteMasks.data());
	}

	if (forceSet || updateBlendingStates)
	{
		std::vector<VkBool32> colorBlendEnable = {};
		colorBlendEnable.reserve(m_colorBlendAttachmentStatesApplied.size());
		for (const auto& i : m_colorBlendAttachmentStatesApplied)
		{
			colorBlendEnable.push_back(i.blendEnable);
		}
		if (!colorBlendEnable.empty()) vkCmdSetColorBlendEnableEXT(commandBuffer, 0, colorBlendEnable.size(), colorBlendEnable.data());
	}
	if (forceSet || updateBlendingStates)
	{
		std::vector<VkColorBlendEquationEXT> equations{};
		for (const auto& i : m_colorBlendAttachmentStatesApplied)
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
	}

	if (forceSet
		|| m_blendConstantsApplied[0] != m_blendConstants[0]
		|| m_blendConstantsApplied[1] != m_blendConstants[1]
		|| m_blendConstantsApplied[2] != m_blendConstants[2]
		|| m_blendConstantsApplied[3] != m_blendConstants[3])
	{
		vkCmdSetBlendConstants(commandBuffer, m_blendConstantsApplied);
	}
}
