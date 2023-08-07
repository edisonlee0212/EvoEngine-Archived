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
	class MeshStorage : public ISingleton<MeshStorage>
	{
		std::vector<VertexDataChunk> m_vertexDataChunks = {};
		std::vector<Meshlet> m_meshlets = {};
		std::queue<uint32_t> m_vertexDataChunkPool = {};
		std::queue<uint32_t> m_meshletPool = {};

		std::vector<std::unique_ptr<Buffer>> m_vertexBuffer = {};
		std::vector < std::unique_ptr<Buffer>> m_meshletBuffer = {};
		std::vector<bool> m_requireDeviceUpdate = {};

		void UploadData();
		friend class Graphics;
		friend class Resources;
		static void PreUpdate();
		static void Initialize();
	public:
		static const std::unique_ptr<Buffer>& GetVertexBuffer();
		static const std::unique_ptr<Buffer>& GetMeshletBuffer();
		static void Bind(VkCommandBuffer commandBuffer);
		[[nodiscard]] static const Vertex& PeekVertex(size_t vertexIndex);
		static void Allocate(const std::vector<Vertex>& vertices, const std::vector<glm::uvec3>& triangles,
			std::vector<uint32_t>& meshletIndices);
		static void Free(const std::vector<uint32_t>& meshletIndices);

		[[nodiscard]] static const Meshlet& PeekMeshlet(uint32_t meshletIndex);
	};
}
