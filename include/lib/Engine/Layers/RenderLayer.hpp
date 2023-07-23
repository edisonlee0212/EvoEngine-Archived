#pragma once
#include "Camera.hpp"
#include "ILayer.hpp"
#include "Material.hpp"
#include "Mesh.hpp"

namespace EvoEngine
{
	struct RenderInfoBlock {
		glm::vec4 m_splitDistances = {};
		alignas(4) int m_pcfSampleAmount = 64;
		alignas(4) int m_blockerSearchAmount = 1;
		alignas(4) float m_seamFixRatio = 0.05f;
		alignas(4) float m_gamma = 2.2f;

		alignas(4) float m_strandsSubdivisionXFactor = 50.0f;
		alignas(4) float m_strandsSubdivisionYFactor = 50.0f;
		alignas(4) int m_strandsSubdivisionMaxX = 15;
		alignas(4) int m_strandsSubdivisionMaxY = 8;
	};

	struct EnvironmentInfoBlock {
		glm::vec4 m_backgroundColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
		alignas(4) float m_environmentalMapGamma = 1.0f;
		alignas(4) float m_environmentalLightingIntensity = 1.0f;
		alignas(4) float m_backgroundIntensity = 1.0f;
		alignas(4) float m_environmentalPadding2 = 0.0f;
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
		alignas(4) bool m_albedoEnabled = false;
		alignas(4) bool m_normalEnabled = false;
		alignas(4) bool m_metallicEnabled = false;
		alignas(4) bool m_roughnessEnabled = false;

		alignas(4) bool m_aoEnabled = false;
		alignas(4) bool m_castShadow = true;
		alignas(4) bool m_receiveShadow = true;
		alignas(4) bool m_enableShadow = true;

		glm::vec4 m_albedoColorVal = glm::vec4(1.0f);
		glm::vec4 m_subsurfaceColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
		glm::vec4 m_subsurfaceRadius = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);

		alignas(4) float m_metallicVal = 0.5f;
		alignas(4) float m_roughnessVal = 0.5f;
		alignas(4) float m_aoVal = 1.0f;
		alignas(4) float m_emissionVal = 0.0f;
	};

	class RenderLayer : public ILayer {
		std::unique_ptr<DescriptorSetLayout> m_perFrameLayout = {};
		std::unique_ptr<DescriptorSetLayout> m_perPassLayout = {};
		std::unique_ptr<DescriptorSetLayout> m_perObjectGroupLayout = {};

		std::vector<VkDescriptorSet> m_perFrameDescriptorSets = {};
		std::vector<VkDescriptorSet> m_perPassDescriptorSets = {};
		std::vector<VkDescriptorSet> m_perObjectGroupDescriptorSets = {};

		std::vector<void*> m_renderInfoBlockMemory;
		std::vector<void*> m_environmentalInfoBlockMemory;
		std::vector<void*> m_cameraInfoBlockMemory;
		std::vector<void*> m_materialInfoBlockMemory;

		
		std::vector<std::unique_ptr<Buffer>> m_descriptorBuffers = {};

		std::unique_ptr<PipelineLayout> m_pipelineLayout = {};

		std::shared_ptr<ShaderEXT> m_vertShader;
		std::shared_ptr<ShaderEXT> m_fragShader;
		
		void OnCreate() override;
		void OnDestroy() override;
		void PreUpdate() override;
		void LateUpdate() override;
		void CreateRenderPass();
		std::unique_ptr<RenderPass> m_renderPass = {};
	public:
		EnvironmentInfoBlock m_environmentInfoBlock = {};
		RenderInfoBlock m_renderInfoBlock = {};
		MaterialInfoBlock m_materialInfoBlock = {};
		CameraInfoBlock m_cameraInfoBlock = {};

		std::shared_ptr<Mesh> m_mesh;
	};
}