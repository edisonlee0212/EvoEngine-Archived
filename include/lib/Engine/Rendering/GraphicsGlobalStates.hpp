#pragma once
#include "GraphicsPipeline.hpp"
#include "Transform.hpp"
#include "Vertex.hpp"
namespace EvoEngine
{
	class GraphicsGlobalStates
	{
		friend class Graphics;
		VkViewport m_viewPortApplied = {};
		VkRect2D m_scissorApplied = {};
		uint32_t m_patchControlPointsApplied = 1;
		bool m_depthClampApplied = false;
		bool m_rasterizerDiscardApplied = false;
		VkPolygonMode m_polygonModeApplied = VK_POLYGON_MODE_FILL;
		VkCullModeFlags m_cullModeApplied = VK_CULL_MODE_BACK_BIT;
		VkFrontFace m_frontFaceApplied = VK_FRONT_FACE_CLOCKWISE;
		bool m_depthBiasApplied = false;
		glm::vec3 m_depthBiasConstantClampSlopeApplied = glm::vec3(0.0f);
		float m_lineWidthApplied = 1.0f;
		bool m_depthTestApplied = true;
		bool m_depthWriteApplied = true;
		VkCompareOp m_depthCompareApplied = VK_COMPARE_OP_LESS;
		bool m_depthBoundTestApplied = false;
		glm::vec2 m_minMaxDepthBoundApplied = glm::vec2(-1.0f, 1.0f);
		bool m_stencilTestApplied = false;
		VkStencilFaceFlags m_stencilFaceMaskApplied = VK_STENCIL_FACE_FRONT_BIT;
		VkStencilOp m_stencilFailOpApplied = VK_STENCIL_OP_ZERO;
		VkStencilOp m_stencilPassOpApplied = VK_STENCIL_OP_ZERO;
		VkStencilOp m_stencilDepthFailOpApplied = VK_STENCIL_OP_ZERO;
		VkCompareOp m_stencilCompareOpApplied = VK_COMPARE_OP_LESS;
		void ResetAllStates(VkCommandBuffer commandBuffer);

		std::shared_ptr<RenderPass> m_renderPassApplied = {};
		std::shared_ptr<GraphicsPipeline> m_graphicsPipeline = {};
		uint32_t m_subpassIndexApplied = 0;
	public:
		VkViewport m_viewPort = {};
		VkRect2D m_scissor = {};
		uint32_t m_patchControlPoints = 1;
		bool m_depthClamp = false;
		bool m_rasterizerDiscard = false;
		VkPolygonMode m_polygonMode = VK_POLYGON_MODE_FILL;
		VkCullModeFlags m_cullMode = VK_CULL_MODE_BACK_BIT;
		VkFrontFace m_frontFace = VK_FRONT_FACE_CLOCKWISE;
		bool m_depthBias = false;
		glm::vec3 m_depthBiasConstantClampSlope = glm::vec3(0.0f);
		float m_lineWidth = 1.0f;
		bool m_depthTest = true;
		bool m_depthWrite = true;
		VkCompareOp m_depthCompare = VK_COMPARE_OP_LESS;
		bool m_depthBoundTest = false;
		glm::vec2 m_minMaxDepthBound = glm::vec2(0.0f, 1.0f);
		bool m_stencilTest = false;
		VkStencilFaceFlags m_stencilFaceMask = VK_STENCIL_FACE_FRONT_BIT;
		VkStencilOp m_stencilFailOp = VK_STENCIL_OP_ZERO;
		VkStencilOp m_stencilPassOp = VK_STENCIL_OP_ZERO;
		VkStencilOp m_stencilDepthFailOp = VK_STENCIL_OP_ZERO;
		VkCompareOp m_stencilCompareOp = VK_COMPARE_OP_LESS;
		void ApplyAllStates(VkCommandBuffer commandBuffer, bool forceSet = false);

		void BeginRenderPass(VkCommandBuffer commandBuffer, const std::shared_ptr<RenderPass>& targetRenderPass, const std::shared_ptr<Framebuffer>& targetFramebuffer);
		void NextSubpass(VkCommandBuffer commandBuffer, VkSubpassContents subpassContents = VK_SUBPASS_CONTENTS_INLINE);
		void EndRenderPass(VkCommandBuffer commandBuffer);

		void BindGraphicsPipeline(VkCommandBuffer commandBuffer, const std::shared_ptr<GraphicsPipeline>& graphicsPipeline);
	};
}