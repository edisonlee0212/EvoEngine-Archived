#pragma once
#include "AssetRef.hpp"
#include "Graphics.hpp"
#include "IAsset.hpp"
#include "MaterialProperties.hpp"
namespace EvoEngine
{
	struct DrawSettings {
		float m_lineWidth = 1.0f;
		VkCullModeFlags m_cullMode = VK_CULL_MODE_BACK_BIT;
		VkPolygonMode m_polygonMode = VK_POLYGON_MODE_FILL;

		bool m_blending = false;
		VkBlendFactor m_blendingSrcFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		VkBlendFactor m_blendingDstFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		bool OnInspect();
		void ApplySettings(GlobalPipelineState& globalPipelineState) const;

		void Save(const std::string& name, YAML::Emitter& out) const;
		void Load(const std::string& name, const YAML::Node& in);
	};

	class Material final : public IAsset
	{
	public:
		AssetRef m_albedoTexture;
		AssetRef m_normalTexture;
		AssetRef m_metallicTexture;
		AssetRef m_roughnessTexture;
		AssetRef m_aoTexture;

		bool m_vertexColorOnly = false;
		MaterialProperties m_materialProperties;
		DrawSettings m_drawSettings;

		void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		void CollectAssetRef(std::vector<AssetRef>& list) override;
		void Serialize(YAML::Emitter& out) override;
		void Deserialize(const YAML::Node& in) override;
	};
}
