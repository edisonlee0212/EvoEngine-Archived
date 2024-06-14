#pragma once
#include "Transform.hpp"
#include "Vertex.hpp"
namespace evo_engine
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
		virtual void DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState, int instanceCount) const = 0;
		static const std::vector<VkVertexInputBindingDescription>& GetVertexBindingDescriptions(GeometryType geometryType);
		static const std::vector<VkVertexInputAttributeDescription>& GetVertexAttributeDescriptions(GeometryType geometryType);
	};
}