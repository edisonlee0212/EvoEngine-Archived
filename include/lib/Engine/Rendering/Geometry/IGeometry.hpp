#pragma once
#include "Transform.hpp"
#include "Vertex.hpp"
namespace EvoEngine
{
	class GraphicsPipelineStates;

	enum class GeometryType
	{
		Mesh,
		SkinnedMesh,
		Strands
	};
	class IGeometry
	{
	public:
		virtual void Bind(VkCommandBuffer vkCommandBuffer) const = 0;
		virtual void DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState, bool enableMetrics = true) const = 0;
		//virtual void DrawInstanced(const std::vector<glm::mat4>& matrices) const {}
		//virtual void DrawInstanced(const std::vector<GlobalTransform>& matrices) const {}
		//virtual void DrawInstanced(const std::shared_ptr<ParticleMatrices>& matrices) const {}
		//virtual void DrawInstancedColored(const std::vector<glm::vec4>& colors, const std::vector<glm::mat4>& matrices) const {}
		//virtual void DrawInstancedColored(const std::vector<glm::vec4>& colors, const std::vector<GlobalTransform>& matrices) const {}
		static const std::vector<VkVertexInputBindingDescription>& GetVertexBindingDescriptions(GeometryType geometryType);
		static const std::vector<VkVertexInputAttributeDescription>& GetVertexAttributeDescriptions(GeometryType geometryType);
	};
}