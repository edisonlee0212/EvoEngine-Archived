#include "GraphicsGlobalStates.hpp"

using namespace EvoEngine;



void GraphicsGlobalStates::ResetAllStates(VkCommandBuffer commandBuffer)
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
	ApplyAllStates(commandBuffer, true);
}

void GraphicsGlobalStates::ApplyAllStates(const VkCommandBuffer commandBuffer, const bool forceSet)
{
	m_viewPortApplied = m_viewPort;
	m_viewPort.width = glm::max(1.0f, m_viewPort.width);
	m_viewPort.height = glm::max(1.0f, m_viewPort.height);
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
}
