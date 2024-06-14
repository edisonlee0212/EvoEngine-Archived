#pragma once
namespace evo_engine
{
	class GraphicsPipelineStates
	{
		friend class Graphics;
		VkViewport m_viewPortApplied = {};
		VkRect2D m_scissorApplied = {};

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

		bool m_logicOpEnableApplied = VK_FALSE;
		VkLogicOp m_logicOpApplied = VK_LOGIC_OP_COPY;
		
		float m_blendConstantsApplied[4] = { 0, 0, 0, 0 };
	public:
		void ResetAllStates(size_t colorAttachmentSize);
		VkViewport m_viewPort = {};
		VkRect2D m_scissor = {};
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

		bool m_logicOpEnable = VK_FALSE;
		VkLogicOp m_logicOp = VK_LOGIC_OP_COPY;
		std::vector<VkPipelineColorBlendAttachmentState> m_colorBlendAttachmentStates = {};
		float m_blendConstants[4] = {0, 0, 0, 0};
		void ApplyAllStates(VkCommandBuffer commandBuffer, bool forceSet = false);
	};
}