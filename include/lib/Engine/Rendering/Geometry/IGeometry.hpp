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
		virtual void Bind(VkCommandBuffer vkCommandBuffer) {};
		virtual void DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState, int instanceCount) const = 0;
		static const std::vector<VkVertexInputBindingDescription>& GetVertexBindingDescriptions(GeometryType geometryType);
		static const std::vector<VkVertexInputAttributeDescription>& GetVertexAttributeDescriptions(GeometryType geometryType);
	};
}