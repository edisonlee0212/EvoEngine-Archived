#pragma once
#include "Camera.hpp"
#include "ILayer.hpp"
#include "Material.hpp"
#include "Mesh.hpp"

namespace EvoEngine
{
	struct RenderInfoBlock {
		float m_splitDistance[4] = {};
		int m_pcfSampleAmount = 64;
		int m_blockerSearchAmount = 1;
		float m_seamFixRatio = 0.05f;
		float m_gamma = 2.2f;

		float m_strandsSubdivisionXFactor = 50.0f;
		float m_strandsSubdivisionYFactor = 50.0f;
		int m_strandsSubdivisionMaxX = 15;
		int m_strandsSubdivisionMaxY = 8;
	};

	struct EnvironmentInfoBlock {
		glm::vec4 m_backgroundColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
		float m_environmentalMapGamma = 1.0f;
		float m_environmentalLightingIntensity = 1.0f;
		float m_backgroundIntensity = 1.0f;
		float m_environmentalPadding2 = 0.0f;
	};

	struct CameraInfoBlock
	{
		glm::mat4 m_projection = {};
		glm::mat4 m_view = {};
		glm::mat4 m_projectionView = {};
		glm::mat4 m_inverseProjection = {};
		glm::mat4 m_inverseView = {};
		glm::mat4 m_inverseProjectionView = {};
		glm::vec4 m_clearColor = {};
		glm::vec4 m_reservedParameters1 = {};
		glm::vec4 m_reservedParameters2 = {};
	};

	struct MaterialInfoBlock {
		int m_albedoEnabled = 0;
		int m_normalEnabled = 0;
		int m_metallicEnabled = 0;
		int m_roughnessEnabled = 0;

		int m_aoEnabled = 0;
		int m_castShadow = true;
		int m_receiveShadow = true;
		int m_enableShadow = true;

		glm::vec4 m_albedoColorVal = glm::vec4(1.0f);
		glm::vec4 m_subsurfaceColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
		glm::vec4 m_subsurfaceRadius = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
		float m_metallicVal = 0.5f;
		float m_roughnessVal = 0.5f;
		float m_aoVal = 1.0f;
		float m_emissionVal = 0.0f;
	};

	class RenderLayer : public ILayer {
		DescriptorSetLayout m_perFrameLayout = {};
		DescriptorSetLayout m_perPassLayout = {};
		DescriptorSetLayout m_perObjectGroupLayout = {};

		std::vector<VkDescriptorSet> m_perFrameDescriptorSets = {};
		std::vector<VkDescriptorSet> m_perPassDescriptorSets = {};
		std::vector<VkDescriptorSet> m_perObjectGroupDescriptorSets = {};

		std::vector<void*> m_renderInfoBlockMemory;
		std::vector<void*> m_environmentalInfoBlockMemory;
		std::vector<void*> m_cameraInfoBlockMemory;
		std::vector<void*> m_materialInfoBlockMemory;

		DescriptorPool m_descriptorPool = {};
		std::vector<Buffer> m_descriptorBuffers = {};

		PipelineLayout m_pipelineLayout = {};
		RenderPass m_renderPass = {};
		GraphicsPipeline m_graphicsPipeline = {};

		std::vector<Framebuffer> m_framebuffers = {};

		void RecordCommandBuffer();

		void CreateRenderPass();
		void CreateGraphicsPipeline();
		void CreateFramebuffers();


		void OnCreate() override;
		void OnDestroy() override;
		void PreUpdate() override;
		void Update() override;
		void LateUpdate() override;

		unsigned m_storedSwapchainVersion = 0;
	public:
		EnvironmentInfoBlock m_environmentInfoBlock = {};
		RenderInfoBlock m_renderInfoBlock = {};
		MaterialInfoBlock m_materialInfoBlock = {};
		CameraInfoBlock m_cameraInfoBlock = {};

		std::shared_ptr<Mesh> m_mesh;
	};
}