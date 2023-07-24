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

		[[nodiscard]] glm::vec3 Project(const glm::vec3& position) const;
		[[nodiscard]] glm::vec3 UnProject(const glm::vec3& position) const;
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
		friend class Resources;
		class ShaderIncludes
		{
			friend class Resources;
		public:
			static std::unique_ptr<std::string> GENERAL_INCLUDES;
			constexpr static size_t MAX_BONE_AMOUNT = 65536;
			constexpr static size_t MAX_MATERIAL_AMOUNT = 1;
			constexpr static size_t MAX_KERNEL_AMOUNT = 64;
			constexpr static size_t MAX_DIRECTIONAL_LIGHT_AMOUNT = 128;
			constexpr static size_t MAX_POINT_LIGHT_AMOUNT = 128;
			constexpr static size_t MAX_SPOT_LIGHT_AMOUNT = 128;
			constexpr static size_t SHADOW_CASCADE_SIZE = 4;
		};

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

		std::shared_ptr<ShaderEXT> m_vertShader = {};
		std::shared_ptr<ShaderEXT> m_fragShader = {};
		
		void OnCreate() override;
		void OnDestroy() override;
		void PreUpdate() override;
		void LateUpdate() override;
		void CreateRenderPasses();
		std::unordered_map<std::string, std::unique_ptr<RenderPass>> m_renderPasses;
	public:
		[[nodiscard]] const std::unique_ptr<RenderPass>& GetRenderPass(const std::string& name);

		bool m_allowAutoResize = true;

		EnvironmentInfoBlock m_environmentInfoBlock = {};
		RenderInfoBlock m_renderInfoBlock = {};
		MaterialInfoBlock m_materialInfoBlock = {};
		CameraInfoBlock m_cameraInfoBlock = {};

		std::shared_ptr<Mesh> m_mesh = {};
	};
}