#pragma once
#include "Bound.hpp"
#include "GraphicsResources.hpp"
#include "IAsset.hpp"
#include "Vertex.hpp"
namespace EvoEngine
{
	struct VertexAttributes
	{
		bool m_position = true;
		bool m_normal = false;
		bool m_tangent = false;
		bool m_texCoord = false;
		bool m_color = false;

		void Serialize(YAML::Emitter& out) const;
		void Deserialize(const YAML::Node& in);
	};

	class Mesh final : public IAsset
	{
		size_t m_version = 0;
		Bound m_bound = {};

		std::vector<Vertex> m_vertices;
		std::vector<glm::uvec3> m_triangles;

		VertexAttributes m_vertexAttributes;

		Buffer m_verticesBuffer;
		Buffer m_trianglesBuffer;
	public:
		void SubmitDrawIndexed(VkCommandBuffer vkCommandBuffer);

		void UploadData();
		void SetVertices(const VertexAttributes& vertexAttributes, std::vector<Vertex>& vertices, const std::vector<unsigned>& indices);
		void SetVertices(const VertexAttributes& vertexAttributes, const std::vector<Vertex>& vertices, const std::vector<glm::uvec3>& triangles);
		[[nodiscard]] size_t GetVerticesAmount() const;
		[[nodiscard]] size_t GetTriangleAmount() const;

		void RecalculateNormal();
		void RecalculateTangent();

		[[nodiscard]] size_t& GetVersion();
		[[nodiscard]] std::vector<Vertex>& UnsafeGetVertices();
		[[nodiscard]] std::vector<glm::uvec3>& UnsafeGetTriangles();

		void Serialize(YAML::Emitter& out) override;
		void Deserialize(const YAML::Node& in) override;
	};
}