#pragma once
#include "Bound.hpp"
#include "Graphics.hpp"
#include "GraphicsResources.hpp"
#include "IAsset.hpp"
#include "IGeometry.hpp"
#include "Vertex.hpp"
namespace EvoEngine
{
	struct VertexAttributes
	{
		bool m_normal = false;
		bool m_tangent = false;
		bool m_texCoord = false;
		bool m_color = false;

		void Serialize(YAML::Emitter& out) const;
		void Deserialize(const YAML::Node& in);
	};

	struct ParticleInfo
	{
		Transform m_instanceMatrix = {};
		glm::vec4 m_instanceColor = glm::vec4(1.0f);

	};

	class ParticleInfoList : public IAsset
	{
		std::vector<std::shared_ptr<Buffer>> m_buffer;
		std::vector<std::shared_ptr<DescriptorSet>> m_descriptorSet;

		std::vector<bool> m_pendingUpdate;
	public:
		void Serialize(YAML::Emitter& out) override;
		void Deserialize(const YAML::Node& in) override;
		void UploadData(bool force = false);
		void SetPendingUpdate();
		ParticleInfoList();
		void ApplyRays(const std::vector<Ray>& rays, const glm::vec4& color, float rayWidth);
		void ApplyRays(const std::vector<Ray>& rays, const std::vector<glm::vec4>& colors, float rayWidth);
		void ApplyConnections(const std::vector<glm::vec3>& starts,
			const std::vector<glm::vec3>& ends, const glm::vec4& color, float rayWidth);
		void ApplyConnections(const std::vector<glm::vec3>& starts,
			const std::vector<glm::vec3>& ends, const std::vector<glm::vec4>& colors, float rayWidth);
;		std::vector<ParticleInfo> m_particleInfos;
		[[nodiscard]] const std::shared_ptr<DescriptorSet>& GetDescriptorSet() const;
	};

	class Mesh final : public IAsset, public IGeometry
	{
		Bound m_bound = {};

		std::vector<Vertex> m_vertices;
		std::vector<glm::uvec3> m_triangles;

		VertexAttributes m_vertexAttributes = {};

		std::vector<glm::uvec3> m_geometryStorageTriangles;
		std::vector<std::unique_ptr<Buffer>> m_trianglesBuffer = {};
		std::vector<bool> m_uploadPending;
		std::vector<uint32_t> m_meshletIndices;
	public:
		[[nodiscard]] const std::vector<uint32_t>& PeekMeshletIndices() const;
		void OnCreate() override;
		~Mesh() override;
		void DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState, int instancesCount, bool enableMetrics) const override;

		void Bind(VkCommandBuffer vkCommandBuffer) override;

		void UploadData();
		void SetVertices(const VertexAttributes& vertexAttributes, const std::vector<Vertex>& vertices, const std::vector<unsigned>& indices);
		void SetVertices(const VertexAttributes& vertexAttributes, const std::vector<Vertex>& vertices, const std::vector<glm::uvec3>& triangles);
		[[nodiscard]] size_t GetVerticesAmount() const;
		[[nodiscard]] size_t GetTriangleAmount() const;

		void RecalculateNormal();
		void RecalculateTangent();

		
		[[nodiscard]] std::vector<Vertex>& UnsafeGetVertices();
		[[nodiscard]] std::vector<glm::uvec3>& UnsafeGetTriangles();
		[[nodiscard]] Bound GetBound() const;
		
		void Serialize(YAML::Emitter& out) override;
		void Deserialize(const YAML::Node& in) override;
	};
}