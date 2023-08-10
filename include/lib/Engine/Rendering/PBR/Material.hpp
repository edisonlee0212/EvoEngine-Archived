#pragma once
#include "AssetRef.hpp"
#include "Graphics.hpp"
#include "IAsset.hpp"
#include "MaterialProperties.hpp"
#include "Texture2D.hpp"
namespace EvoEngine
{
	struct MaterialInfoBlock {
		alignas(4) int m_albedoTextureIndex = -1;
		alignas(4) int m_normalTextureIndex = -1;
		alignas(4) int m_metallicTextureIndex = -1;
		alignas(4) int m_roughnessTextureIndex = -1;

		alignas(4) int m_aoTextureIndex = -1;
		alignas(4) int m_castShadow = true;
		alignas(4) int m_receiveShadow = true;
		alignas(4) int m_enableShadow = true;

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
		VkBlendOp m_blendOp = VK_BLEND_OP_ADD;

		VkBlendFactor m_blendingSrcFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		VkBlendFactor m_blendingDstFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		bool OnInspect();
		void ApplySettings(GraphicsPipelineStates& globalPipelineState) const;

		void Save(const std::string& name, YAML::Emitter& out) const;
		void Load(const std::string& name, const YAML::Node& in);
	};

	class Material final : public IAsset
	{
		friend class RenderLayer;
		bool m_needUpdate = true;
		AssetRef m_albedoTexture;
		AssetRef m_normalTexture;
		AssetRef m_metallicTexture;
		AssetRef m_roughnessTexture;
		AssetRef m_aoTexture;
	public:
		void SetAlbedoTexture(const std::shared_ptr<Texture2D>& texture);
		void SetNormalTexture(const std::shared_ptr<Texture2D>& texture);
		void SetMetallicTexture(const std::shared_ptr<Texture2D>& texture);
		void SetRoughnessTexture(const std::shared_ptr<Texture2D>& texture);
		void SetAOTexture(const std::shared_ptr<Texture2D>& texture);
		[[nodiscard]] std::shared_ptr<Texture2D> GetAlbedoTexture();
		[[nodiscard]] std::shared_ptr<Texture2D> GetNormalTexture();
		[[nodiscard]] std::shared_ptr<Texture2D> GetMetallicTexture();
		[[nodiscard]] std::shared_ptr<Texture2D> GetRoughnessTexture();
		[[nodiscard]] std::shared_ptr<Texture2D> GetAoTexture();
		void OnCreate() override;
		~Material() override;
		

		bool m_vertexColorOnly = false;
		MaterialProperties m_materialProperties;
		DrawSettings m_drawSettings;

		void UpdateMaterialInfoBlock(MaterialInfoBlock& materialInfoBlock);
		void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		void CollectAssetRef(std::vector<AssetRef>& list) override;
		void Serialize(YAML::Emitter& out) override;
		void Deserialize(const YAML::Node& in) override;
	};
}
