#pragma once
#include "AssetRef.hpp"
#include "Graphics.hpp"
#include "IAsset.hpp"
#include "MaterialProperties.hpp"
namespace EvoEngine
{
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

	struct DrawSettings {
		float m_lineWidth = 1.0f;
		VkCullModeFlags m_cullMode = VK_CULL_MODE_BACK_BIT;
		VkPolygonMode m_polygonMode = VK_POLYGON_MODE_FILL;

		bool m_blending = false;
		VkBlendFactor m_blendingSrcFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		VkBlendFactor m_blendingDstFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		bool OnInspect();
		void ApplySettings(GraphicsGlobalStates& globalPipelineState) const;

		void Save(const std::string& name, YAML::Emitter& out) const;
		void Load(const std::string& name, const YAML::Node& in);
	};

	class Material final : public IAsset
	{
		friend class RenderLayer;
		VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
	public:
		void OnCreate() override;
		~Material() override;
		AssetRef m_albedoTexture;
		AssetRef m_normalTexture;
		AssetRef m_metallicTexture;
		AssetRef m_roughnessTexture;
		AssetRef m_aoTexture;

		bool m_vertexColorOnly = false;
		MaterialProperties m_materialProperties;
		DrawSettings m_drawSettings;

		void UpdateMaterialInfoBlock(MaterialInfoBlock& materialInfoBlock);
		void UpdateDescriptorBindings();
		void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		void CollectAssetRef(std::vector<AssetRef>& list) override;
		void Serialize(YAML::Emitter& out) override;
		void Deserialize(const YAML::Node& in) override;
	};
}
