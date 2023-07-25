#pragma once
#include "AssetRef.hpp"
#include "GraphicsResources.hpp"
#include "IAsset.hpp"

namespace EvoEngine
{
	enum class ShaderType
	{
		Vertex,
		TessellationControl,
		TessellationEvaluation,
		Geometry,
		Fragment,
		Compute,
		Unknown
	};
	class Shader final : public IAsset
	{
		friend class ShaderProgram;
		std::unique_ptr<ShaderEXT> m_shaderEXT;
		std::string m_code = {};
		ShaderType m_shaderType = ShaderType::Unknown;
	public:
		void Set(ShaderType shaderType, const std::string& shaderCode);
		[[nodiscard]] const std::unique_ptr<ShaderEXT>& GetShaderEXT() const;
	};

	enum class DescriptorSetType
	{
		PerFrame,
		PerPass,
		PerGroup,
		PerCommand,
		Unknown
	};

	struct DescriptorSetLayoutBinding
	{
		DescriptorSetType m_type = DescriptorSetType::Unknown;
		VkDescriptorSetLayoutBinding m_binding = {};
		bool m_ready = false;
	};
	class ShaderProgram final : public IAsset
	{
		std::vector<DescriptorSetLayoutBinding> m_descriptorSetLayoutBindings;
		bool m_containStandardBindings = false;

		std::unique_ptr<DescriptorSetLayout> m_perFrameLayout = {};
		std::unique_ptr<DescriptorSetLayout> m_perPassLayout = {};
		std::unique_ptr<DescriptorSetLayout> m_perGroupLayout = {};
		std::unique_ptr<DescriptorSetLayout> m_perCommandLayout = {};

		std::unique_ptr<PipelineLayout> m_pipelineLayout = {};

		/**
		 * \brief You can link the shader with the layouts ready.
		 * The actual descriptor set can be set later.
		 */
		bool m_layoutReady = false;

		std::vector<VkDescriptorSet> m_perFrameDescriptorSets = {};
		std::vector<VkDescriptorSet> m_perPassDescriptorSets = {};
		std::vector<VkDescriptorSet> m_perGroupDescriptorSets = {};
		std::vector<VkDescriptorSet> m_perCommandDescriptorSets = {};

		/**
		 * \brief You can only bind shaders after the descriptor sets are ready.
		 */
		bool m_descriptorSetsReady = false;
		void CheckDescriptorSetsReady();
	public:
		AssetRef m_vertexShader;
		AssetRef m_tessellationControlShader;
		AssetRef m_tessellationEvaluationShader;
		AssetRef m_geometryShader;
		AssetRef m_fragmentShader;
		AssetRef m_computeShader;

		void ClearDescriptorSets();
		void InitializeStandardBindings();
		[[nodiscard]] bool ContainStandardBindings() const;

		[[maybe_unused]] uint32_t PushDescriptorBinding(DescriptorSetType setType, VkDescriptorType type, VkShaderStageFlags stageFlags);

		void PrepareLayouts();
		[[nodiscard]] bool LayoutSetsReady() const;
		void LinkShaders();

		void UpdateStandardBindings();
		void UpdateBinding(uint32_t binding, const VkDescriptorBufferInfo& bufferInfo);
		void UpdateBinding(uint32_t binding, const VkDescriptorImageInfo& imageInfo);
		[[nodiscard]] bool DescriptorSetsReady() const;

		void BindDescriptorSet(VkCommandBuffer commandBuffer, DescriptorSetType setType) const;
		void BindAllDescriptorSets(VkCommandBuffer commandBuffer) const;
		void BindShaders(VkCommandBuffer commandBuffer);
	};
}
