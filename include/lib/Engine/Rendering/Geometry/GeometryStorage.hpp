#pragma once
#include "Graphics.hpp"
#include "GraphicsResources.hpp"
#include "ISingleton.hpp"
#include "Vertex.hpp"

namespace EvoEngine
{
	struct VertexDataChunk
	{
		Vertex m_vertexData[Graphics::Constants::MESHLET_MAX_VERTICES_SIZE] = {};
	};

	struct Meshlet
	{
		glm::u8vec3 m_triangles[Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE] = {}; // up to 126 triangles
		uint32_t m_verticesSize = 0;
		uint32_t m_triangleSize = 0;
		uint32_t m_vertexChunkIndex = 0;
	};

	struct SkinnedVertexDataChunk
	{
		SkinnedVertex m_skinnedVertexData[Graphics::Constants::MESHLET_MAX_VERTICES_SIZE] = {};
	};
	struct SkinnedMeshlet
	{
		glm::u8vec3 m_skinnedTriangles[Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE] = {}; // up to 126 triangles
		uint32_t m_skinnedVerticesSize = 0;
		uint32_t m_skinnedTriangleSize = 0;
		uint32_t m_skinnedVertexChunkIndex = 0;
	};

	struct StrandPointDataChunk
	{
		StrandPoint m_strandPointData[Graphics::Constants::MESHLET_MAX_VERTICES_SIZE] = {};
	};
	struct StrandMeshlet
	{
		glm::u8vec4 m_segments[Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE] = {}; // up to 126 triangles
		uint32_t m_strandPointsSize = 0;
		uint32_t m_segmentSize = 0;
		uint32_t m_strandPointChunkIndex = 0;
	};
	class RangeDescriptor
	{
		friend class GeometryStorage;
		Handle m_handle;
	public:
		uint32_t m_offset;
		/**
		 * \brief When used to record meshlet range. This records the newest number of meshlet for this geometry.
		 * When used to record triangles, this records the newest number of triangles, including the size of the empty fillers.
		 */
		uint32_t m_range;

		uint32_t m_prevFrameOffset;
		uint32_t m_indexCount;
		uint32_t m_prevFrameIndexCount;
	};

	struct ParticleInfo
	{
		Transform m_instanceMatrix = {};
		glm::vec4 m_instanceColor = glm::vec4(1.0f);
	};

	enum class ParticleInfoListDataStatus
	{
		Updated,
		UpdatePending,
		Removed
	};
	struct ParticleInfoListData
	{
		std::shared_ptr<Buffer> m_buffer;
		std::shared_ptr<DescriptorSet> m_descriptorSet;
		std::vector<ParticleInfo> m_particleInfoList;
		ParticleInfoListDataStatus m_status = ParticleInfoListDataStatus::Updated;
		std::shared_ptr<RangeDescriptor> m_rangeDescriptor;
	};

	class GeometryStorage : public ISingleton<GeometryStorage>
	{
		std::vector<VertexDataChunk> m_vertexDataChunks = {};
		std::vector<Meshlet> m_meshlets = {};
		std::vector<std::shared_ptr<RangeDescriptor>> m_meshletRangeDescriptor;
		std::vector<glm::uvec3> m_triangles;
		std::vector<std::shared_ptr<RangeDescriptor>> m_triangleRangeDescriptor;

		std::unique_ptr<Buffer> m_vertexBuffer = {};
		std::unique_ptr<Buffer> m_meshletBuffer = {};
		std::unique_ptr<Buffer> m_triangleBuffer = {};
		bool m_requireMeshDataDeviceUpdate = {};

		std::vector<SkinnedVertexDataChunk> m_skinnedVertexDataChunks = {};
		std::vector<SkinnedMeshlet> m_skinnedMeshlets = {};
		std::vector<std::shared_ptr<RangeDescriptor>> m_skinnedMeshletRangeDescriptor;
		std::vector<glm::uvec3> m_skinnedTriangles;
		std::vector<std::shared_ptr<RangeDescriptor>> m_skinnedTriangleRangeDescriptor;

		std::unique_ptr<Buffer> m_skinnedVertexBuffer = {};
		std::unique_ptr<Buffer> m_skinnedMeshletBuffer = {};
		std::unique_ptr<Buffer> m_skinnedTriangleBuffer = {};
		bool m_requireSkinnedMeshDataDeviceUpdate = {};

		std::vector<StrandPointDataChunk> m_strandPointDataChunks = {};
		std::vector<StrandMeshlet> m_strandMeshlets = {};
		std::vector<std::shared_ptr<RangeDescriptor>> m_strandMeshletRangeDescriptor;
		std::vector<glm::uvec4> m_segments;
		std::vector<std::shared_ptr<RangeDescriptor>> m_segmentRangeDescriptor;

		std::unique_ptr<Buffer> m_strandPointBuffer = {};
		std::unique_ptr<Buffer> m_strandMeshletBuffer = {};
		std::unique_ptr<Buffer> m_segmentBuffer = {};
		bool m_requireStrandMeshDataDeviceUpdate = {};

		void UploadData();
		friend class RenderLayer;
		friend class Resources;
		friend class Graphics;
		static void DeviceSync();
		static void Initialize();

		std::vector<ParticleInfoListData> m_particleInfoListDataList;
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

		static void AllocateMesh(const Handle& handle, const std::vector<Vertex>& vertices, const std::vector<glm::uvec3>& triangles,
			const std::shared_ptr<RangeDescriptor>& targetMeshletRange, const std::shared_ptr<RangeDescriptor>& targetTriangleRange);
		static void AllocateSkinnedMesh(const Handle& handle, const std::vector<SkinnedVertex>& skinnedVertices, const std::vector<glm::uvec3>& skinnedTriangles,
			const std::shared_ptr<RangeDescriptor>& targetSkinnedMeshletRange, const std::shared_ptr<RangeDescriptor>& targetSkinnedTriangleRange);
		static void AllocateStrands(const Handle& handle, const std::vector<StrandPoint>& strandPoints, const std::vector<glm::uvec4>& segments,
			const std::shared_ptr<RangeDescriptor>& targetStrandMeshletRange, const std::shared_ptr<RangeDescriptor>& targetSegmentRange);

		static void FreeMesh(const Handle& handle);
		static void FreeSkinnedMesh(const Handle& handle);
		static void FreeStrands(const Handle& handle);

		static void AllocateParticleInfo(const Handle& handle, const std::shared_ptr<RangeDescriptor>& rangeDescriptor);
		static void UpdateParticleInfo(const std::shared_ptr<RangeDescriptor>& rangeDescriptor, const std::vector<ParticleInfo>& particleInfos);
		static void FreeParticleInfo(const std::shared_ptr<RangeDescriptor>& rangeDescriptor);
		[[nodiscard]] static const std::vector<ParticleInfo>& PeekParticleInfoList(const std::shared_ptr<RangeDescriptor>& rangeDescriptor);
		[[nodiscard]] static const std::shared_ptr<DescriptorSet>& PeekDescriptorSet(const std::shared_ptr<RangeDescriptor>& rangeDescriptor);

		[[nodiscard]] static const Meshlet& PeekMeshlet(uint32_t meshletIndex);
		[[nodiscard]] static const SkinnedMeshlet& PeekSkinnedMeshlet(uint32_t skinnedMeshletIndex);
		[[nodiscard]] static const StrandMeshlet& PeekStrandMeshlet(uint32_t strandMeshletIndex);
	};
}
