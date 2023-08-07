#pragma once
#include "GraphicsResources.hpp"
#include "ISingleton.hpp"
#include "Vertex.hpp"

namespace EvoEngine
{
#define VERTEX_CHUNK_VERTICES_SIZE 64
#define MESHLET_MAX_VERTICES_SIZE 64
#define MESHLET_MAX_TRIANGLES_SIZE 126

	struct VertexDataChunk
	{
		Vertex m_vertexData[VERTEX_CHUNK_VERTICES_SIZE] = {};
	};

	struct Meshlet
	{
		uint32_t m_vertexIndices[MESHLET_MAX_VERTICES_SIZE] = {};
		glm::u8vec3 m_triangles[MESHLET_MAX_TRIANGLES_SIZE] = {}; // up to 126 triangles
		uint8_t m_verticesSize = 0;
		uint8_t m_triangleSize = 0;
	};
	class MeshStorage : public ISingleton<MeshStorage>
	{
		std::vector<VertexDataChunk> m_vertexDataChunks = {};
		std::vector<Meshlet> m_meshlets = {};
		std::queue<uint32_t> m_vertexDataChunkPool = {};
		std::queue<uint32_t> m_meshletPool = {};

		std::unique_ptr<Buffer> m_vertexBuffer = {};
		std::unique_ptr<Buffer> m_meshletBuffer = {};
		bool m_requireDeviceUpdate = false;
		void UploadData();
		friend class Graphics;
		static void PreUpdate();
		static void Initialize();
	public:
		static void Bind(VkCommandBuffer commandBuffer);
		[[nodiscard]] static const Vertex& PeekVertex(size_t vertexIndex);
		static void Allocate(const std::vector<Vertex>& vertices, const std::vector<glm::uvec3>& triangles,
			std::vector<uint32_t>& meshletIndices);
		static void Free(const std::vector<uint32_t>& meshletIndices);

		[[nodiscard]] static const Meshlet& PeekMeshlet(uint32_t meshletIndex);
	};
}
