#pragma once
#include "Graphics.hpp"
#include "GraphicsResources.hpp"
#include "ISingleton.hpp"
#include "Vertex.hpp"

namespace EvoEngine
{
	struct VertexDataChunk
	{
		Vertex m_vertexData[Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE] = {};
	};
	struct Meshlet
	{
		uint32_t m_vertexIndices[Graphics::Constants::MESHLET_MAX_VERTICES_SIZE] = {};
		glm::uvec3 m_triangles[Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE] = {}; // up to 126 triangles
		uint32_t m_verticesSize = 0;
		uint32_t m_triangleSize = 0;
	};

	struct SkinnedVertexDataChunk
	{
		SkinnedVertex m_skinnedVertexData[Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE] = {};
	};
	struct SkinnedMeshlet
	{
		uint32_t m_skinnedVertexIndices[Graphics::Constants::MESHLET_MAX_VERTICES_SIZE] = {};
		glm::uvec3 m_triangles[Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE] = {}; // up to 126 triangles
		uint32_t m_skinnedVerticesSize = 0;
		uint32_t m_triangleSize = 0;
	};

	struct StrandPointDataChunk
	{
		StrandPoint m_strandPointData[Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE] = {};
	};
	struct StrandMeshlet
	{
		uint32_t m_strandPointIndices[Graphics::Constants::MESHLET_MAX_VERTICES_SIZE] = {};
		glm::uvec4 m_segments[Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE] = {}; // up to 126 triangles
		uint32_t m_strandPointsSize = 0;
		uint32_t m_segmentSize = 0;
	};

	class GeometryStorage : public ISingleton<GeometryStorage>
	{
		std::vector<VertexDataChunk> m_vertexDataChunks = {};
		uint32_t m_verticesCount = 0;
		std::vector<Meshlet> m_meshlets = {};
		std::queue<uint32_t> m_vertexDataVertexPool = {};
		std::queue<uint32_t> m_meshletPool = {};
		std::vector<std::unique_ptr<Buffer>> m_vertexBuffer = {};
		std::vector<std::unique_ptr<Buffer>> m_meshletBuffer = {};
		std::vector<bool> m_requireMeshDataDeviceUpdate = {};

		std::vector<SkinnedVertexDataChunk> m_skinnedVertexDataChunks = {};
		uint32_t m_skinnedVerticesCount = 0;
		std::vector<SkinnedMeshlet> m_skinnedMeshlets = {};
		std::queue<uint32_t> m_skinnedVertexDataPool = {};
		std::queue<uint32_t> m_skinnedMeshletPool = {};
		std::vector<std::unique_ptr<Buffer>> m_skinnedVertexBuffer = {};
		std::vector<std::unique_ptr<Buffer>> m_skinnedMeshletBuffer = {};
		std::vector<bool> m_requireSkinnedMeshDataDeviceUpdate = {};

		std::vector<StrandPointDataChunk> m_strandPointDataChunks = {};
		uint32_t m_strandPointsCount = 0;
		std::vector<StrandMeshlet> m_strandMeshlets = {};
		std::queue<uint32_t> m_strandPointDataPool = {};
		std::queue<uint32_t> m_strandMeshletPool = {};
		std::vector<std::unique_ptr<Buffer>> m_strandPointBuffer = {};
		std::vector<std::unique_ptr<Buffer>> m_strandMeshletBuffer = {};
		std::vector<bool> m_requireStrandMeshDataDeviceUpdate = {};

		void UploadData();
		friend class Graphics;
		friend class Resources;
		static void PreUpdate();
		static void Initialize();
	public:
		static const std::unique_ptr<Buffer>& GetVertexBuffer();
		static const std::unique_ptr<Buffer>& GetMeshletBuffer();

		static const std::unique_ptr<Buffer>& GetSkinnedVertexBuffer();
		static const std::unique_ptr<Buffer>& GetSkinnedMeshletBuffer();

		static const std::unique_ptr<Buffer>& GetStrandPointBuffer();
		static const std::unique_ptr<Buffer>& GetStrandMeshletBuffer();

		static void BindVertices(VkCommandBuffer commandBuffer);
		static void BindSkinnedVertices(VkCommandBuffer commandBuffer);
		static void BindStrandPoints(VkCommandBuffer commandBuffer);

		[[nodiscard]] static const Vertex& PeekVertex(size_t vertexIndex);
		[[nodiscard]] static const SkinnedVertex& PeekSkinnedVertex(size_t skinnedVertexIndex);
		[[nodiscard]] static const StrandPoint& PeekStrandPoint(size_t strandPointIndex);

		static void AllocateMesh(const std::vector<Vertex>& vertices, const std::vector<glm::uvec3>& triangles,
			std::vector<uint32_t>& meshletIndices);
		static void AllocateSkinnedMesh(const std::vector<SkinnedVertex>& skinnedVertices, const std::vector<glm::uvec3>& triangles,
			std::vector<uint32_t>& skinnedMeshletIndices);
		static void AllocateStrands(const std::vector<StrandPoint>& strandPoints, const std::vector<glm::uvec4>& segments,
			std::vector<uint32_t>& strandMeshletIndices);

		static void FreeMesh(const std::vector<uint32_t>& meshletIndices);
		static void FreeSkinnedMesh(const std::vector<uint32_t>& skinnedMeshletIndices);
		static void FreeStrands(const std::vector<uint32_t>& strandMeshletIndices);

		[[nodiscard]] static const Meshlet& PeekMeshlet(uint32_t meshletIndex);
		[[nodiscard]] static const SkinnedMeshlet& PeekSkinnedMeshlet(uint32_t skinnedMeshletIndex);
		[[nodiscard]] static const StrandMeshlet& PeekStrandMeshlet(uint32_t strandMeshletIndex);
	};
}
