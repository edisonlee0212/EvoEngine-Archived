#pragma once
#include "Transform.hpp"
#include "Vertex.hpp"
namespace EvoEngine
{
	class GraphicsGlobalStates;

	enum class GeometryType
	{
		Mesh
	};
	class IGeometry
	{
	public:
		virtual void DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsGlobalStates& globalPipelineState) const = 0;
		//virtual void DrawInstanced(const std::vector<glm::mat4>& matrices) const {}
		//virtual void DrawInstanced(const std::vector<GlobalTransform>& matrices) const {}
		//virtual void DrawInstanced(const std::shared_ptr<ParticleMatrices>& matrices) const {}
		//virtual void DrawInstancedColored(const std::vector<glm::vec4>& colors, const std::vector<glm::mat4>& matrices) const {}
		//virtual void DrawInstancedColored(const std::vector<glm::vec4>& colors, const std::vector<GlobalTransform>& matrices) const {}

		static const std::vector<VkVertexInputBindingDescription>& GetVertexBindingDescriptions(GeometryType geometryType);
		static const std::vector<VkVertexInputAttributeDescription>& GetVertexAttributeDescriptions(GeometryType geometryType);
	};
}