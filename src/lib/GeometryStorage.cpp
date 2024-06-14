#include "GeometryStorage.hpp"
#include "meshoptimizer.h"
using namespace evo_engine;

void GeometryStorage::UploadData()
{
	if (m_requireMeshDataDeviceUpdate) {
		m_vertexBuffer->UploadVector(m_vertexDataChunks);
		m_meshletBuffer->UploadVector(m_meshlets);
		m_triangleBuffer->UploadVector(m_triangles);
		m_requireMeshDataDeviceUpdate = false;
	}
	if (m_requireSkinnedMeshDataDeviceUpdate) {
		m_skinnedVertexBuffer->UploadVector(m_skinnedVertexDataChunks);
		m_skinnedMeshletBuffer->UploadVector(m_skinnedMeshlets);
		m_skinnedTriangleBuffer->UploadVector(m_skinnedTriangles);
		m_requireSkinnedMeshDataDeviceUpdate = false;
	}
	if (m_requireStrandMeshDataDeviceUpdate) {
		m_strandPointBuffer->UploadVector(m_strandPointDataChunks);
		m_strandMeshletBuffer->UploadVector(m_strandMeshlets);
		m_segmentBuffer->UploadVector(m_segments);
		m_requireStrandMeshDataDeviceUpdate = false;
	}

	for (int index = 0; index < m_particleInfoListDataList.size(); index++)
	{
		auto& particleInfoListData = m_particleInfoListDataList.at(index);
		if (particleInfoListData.m_status == ParticleInfoListDataStatus::Removed)
		{
			m_particleInfoListDataList.at(index) = m_particleInfoListDataList.back();
			m_particleInfoListDataList.pop_back();
			index--;
		}
		else if (particleInfoListData.m_status == ParticleInfoListDataStatus::UpdatePending)
		{
			particleInfoListData.m_buffer->UploadVector(particleInfoListData.m_particleInfoList);
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.offset = 0;
			bufferInfo.range = VK_WHOLE_SIZE;
			bufferInfo.buffer = particleInfoListData.m_buffer->GetVkBuffer();
			particleInfoListData.m_descriptorSet->UpdateBufferDescriptorBinding(18, bufferInfo, 0);

			particleInfoListData.m_status = ParticleInfoListDataStatus::Updated;
		}
	}
	for (int index = 0; index < m_particleInfoListDataList.size(); index++)
	{
		const auto& particleInfoListData = m_particleInfoListDataList.at(index);
		particleInfoListData.m_rangeDescriptor->m_offset = index;
	}
	const auto& storage = GetInstance();
	for (const auto& triangleRange : storage.m_triangleRangeDescriptor)
	{
		triangleRange->m_prevFrameIndexCount = triangleRange->m_indexCount;
		triangleRange->m_prevFrameOffset = triangleRange->m_offset;
	}

	for (const auto& triangleRange : storage.m_skinnedTriangleRangeDescriptor)
	{
		triangleRange->m_prevFrameIndexCount = triangleRange->m_indexCount;
		triangleRange->m_prevFrameOffset = triangleRange->m_offset;
	}

	for (const auto& triangleRange : storage.m_segmentRangeDescriptor)
	{
		triangleRange->m_prevFrameIndexCount = triangleRange->m_indexCount;
		triangleRange->m_prevFrameOffset = triangleRange->m_offset;
	}
}

void GeometryStorage::DeviceSync()
{
	auto& storage = GetInstance();
	storage.UploadData();
}

void GeometryStorage::Initialize()
{
	auto& storage = GetInstance();
	VkBufferCreateInfo storageBufferCreateInfo{};
	storageBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	storageBufferCreateInfo.size = 1;

	storageBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo verticesVmaAllocationCreateInfo{};
	verticesVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	storageBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	storage.m_vertexBuffer = std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo);
	storageBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	storage.m_meshletBuffer = std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo);

	storageBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	storage.m_triangleBuffer = std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo);

	storage.m_requireMeshDataDeviceUpdate = false;

	storageBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	storage.m_skinnedVertexBuffer = std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo);
	storageBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	storage.m_skinnedMeshletBuffer = std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo);

	storageBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	storage.m_skinnedTriangleBuffer = std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo);

	storage.m_requireSkinnedMeshDataDeviceUpdate = false;

	storageBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	storage.m_strandPointBuffer = std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo);
	storageBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	storage.m_strandMeshletBuffer = std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo);

	storageBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	storage.m_segmentBuffer = std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo);

	storage.m_requireStrandMeshDataDeviceUpdate = false;
}

const std::unique_ptr<Buffer>& GeometryStorage::GetVertexBuffer()
{
	const auto& storage = GetInstance();
	return storage.m_vertexBuffer;
}

const std::unique_ptr<Buffer>& GeometryStorage::GetMeshletBuffer()
{
	const auto& storage = GetInstance();
	return storage.m_meshletBuffer;
}

const std::unique_ptr<Buffer>& GeometryStorage::GetSkinnedVertexBuffer()
{
	const auto& storage = GetInstance();
	return storage.m_skinnedVertexBuffer;
}

const std::unique_ptr<Buffer>& GeometryStorage::GetSkinnedMeshletBuffer()
{
	const auto& storage = GetInstance();
	return storage.m_skinnedMeshletBuffer;
}

const std::unique_ptr<Buffer>& GeometryStorage::GetStrandPointBuffer()
{
	const auto& storage = GetInstance();
	return storage.m_strandPointBuffer;
}

const std::unique_ptr<Buffer>& GeometryStorage::GetStrandMeshletBuffer()
{
	const auto& storage = GetInstance();
	return storage.m_strandMeshletBuffer;
}

void GeometryStorage::BindVertices(const VkCommandBuffer commandBuffer)
{
	const auto& storage = GetInstance();
	constexpr VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &storage.m_vertexBuffer->GetVkBuffer(), offsets);
	vkCmdBindIndexBuffer(commandBuffer, storage.m_triangleBuffer->GetVkBuffer(), 0, VK_INDEX_TYPE_UINT32);
}

void GeometryStorage::BindSkinnedVertices(const VkCommandBuffer commandBuffer)
{
	const auto& storage = GetInstance();
	constexpr VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &storage.m_skinnedVertexBuffer->GetVkBuffer(), offsets);
	vkCmdBindIndexBuffer(commandBuffer, storage.m_skinnedTriangleBuffer->GetVkBuffer(), 0, VK_INDEX_TYPE_UINT32);
}

void GeometryStorage::BindStrandPoints(const VkCommandBuffer commandBuffer)
{
	const auto& storage = GetInstance();
	constexpr VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &storage.m_strandPointBuffer->GetVkBuffer(), offsets);
	vkCmdBindIndexBuffer(commandBuffer, storage.m_segmentBuffer->GetVkBuffer(), 0, VK_INDEX_TYPE_UINT32);
}

const Vertex& GeometryStorage::PeekVertex(const size_t vertexIndex)
{
	const auto& storage = GetInstance();
	return storage.m_vertexDataChunks[vertexIndex / Graphics::Constants::MESHLET_MAX_VERTICES_SIZE].m_vertexData[vertexIndex % Graphics::Constants::MESHLET_MAX_VERTICES_SIZE];
}

const SkinnedVertex& GeometryStorage::PeekSkinnedVertex(const size_t skinnedVertexIndex)
{
	const auto& storage = GetInstance();
	return storage.m_skinnedVertexDataChunks[skinnedVertexIndex / Graphics::Constants::MESHLET_MAX_VERTICES_SIZE].m_skinnedVertexData[skinnedVertexIndex % Graphics::Constants::MESHLET_MAX_VERTICES_SIZE];
}

const StrandPoint& GeometryStorage::PeekStrandPoint(const size_t strandPointIndex)
{
	const auto& storage = GetInstance();
	return storage.m_strandPointDataChunks[strandPointIndex / Graphics::Constants::MESHLET_MAX_VERTICES_SIZE].m_strandPointData[strandPointIndex % Graphics::Constants::MESHLET_MAX_VERTICES_SIZE];
}

void GeometryStorage::AllocateMesh(const Handle& handle, const std::vector<Vertex>& vertices, const std::vector<glm::uvec3>& triangles,
	const std::shared_ptr<RangeDescriptor>& targetMeshletRange, const std::shared_ptr<RangeDescriptor>& targetTriangleRange)
{
	if (vertices.empty() || triangles.empty())
	{
		throw std::runtime_error("Empty vertices or triangles!");
	}
	auto& storage = GetInstance();

	//const auto meshletRange = std::make_shared<RangeDescriptor>();
	targetMeshletRange->m_handle = handle;
	targetMeshletRange->m_offset = storage.m_meshlets.size();
	targetMeshletRange->m_range = 0;

	//const auto triangleRange = std::make_shared<RangeDescriptor>();
	targetTriangleRange->m_handle = handle;
	targetTriangleRange->m_offset = storage.m_triangles.size();
	targetTriangleRange->m_range = 0;
	targetTriangleRange->m_indexCount = triangles.size();

	std::vector<meshopt_Meshlet> meshletsResults;
	std::vector<unsigned> meshletResultVertices;
	std::vector<unsigned char> meshletResultTriangles;
	const auto maxMeshlets =
		meshopt_buildMeshletsBound(triangles.size() * 3, Graphics::Constants::MESHLET_MAX_VERTICES_SIZE, Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE);
	meshletsResults.resize(maxMeshlets);
	meshletResultVertices.resize(maxMeshlets * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE);
	meshletResultTriangles.resize(maxMeshlets * Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE * 3);
	const auto meshletSize = meshopt_buildMeshlets(
		meshletsResults.data(), meshletResultVertices.data(), meshletResultTriangles.data(),
		&triangles.at(0).x, triangles.size() * 3, &vertices.at(0).m_position.x, vertices.size(), sizeof(Vertex),
		Graphics::Constants::MESHLET_MAX_VERTICES_SIZE, Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE, 0);

	targetMeshletRange->m_range = meshletSize;
	for (size_t meshletIndex = 0; meshletIndex < meshletSize; meshletIndex++)
	{
		const uint32_t currentMeshletIndex = storage.m_meshlets.size();
		storage.m_meshlets.emplace_back();
		auto& currentMeshlet = storage.m_meshlets[currentMeshletIndex];

		currentMeshlet.m_vertexChunkIndex = storage.m_vertexDataChunks.size();
		storage.m_vertexDataChunks.emplace_back();
		auto& currentChunk = storage.m_vertexDataChunks[currentMeshlet.m_vertexChunkIndex];

		const auto& meshletResult = meshletsResults.at(meshletIndex);
		for (unsigned vi = 0; vi < meshletResult.vertex_count; vi++)
		{
			currentChunk.m_vertexData[vi] = vertices[meshletResultVertices.at(meshletResult.vertex_offset + vi)];
		}
		currentMeshlet.m_verticesSize = meshletResult.vertex_count;
		currentMeshlet.m_triangleSize = meshletResult.triangle_count;
		for (unsigned ti = 0; ti < meshletResult.triangle_count; ti++)
		{
			auto& currentMeshletTriangle = currentMeshlet.m_triangles[ti];
			currentMeshletTriangle = glm::u8vec3(
				meshletResultTriangles[ti * 3 + meshletResult.triangle_offset],
				meshletResultTriangles[ti * 3 + meshletResult.triangle_offset + 1],
				meshletResultTriangles[ti * 3 + meshletResult.triangle_offset + 2]);

			auto& globalTriangle = storage.m_triangles.emplace_back();
			globalTriangle.x = currentMeshletTriangle.x + currentMeshlet.m_vertexChunkIndex * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
			globalTriangle.y = currentMeshletTriangle.y + currentMeshlet.m_vertexChunkIndex * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
			globalTriangle.z = currentMeshletTriangle.z + currentMeshlet.m_vertexChunkIndex * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
		}
		targetTriangleRange->m_range += currentMeshlet.m_triangleSize;
	}

	storage.m_meshletRangeDescriptor.push_back(targetMeshletRange);
	storage.m_triangleRangeDescriptor.push_back(targetTriangleRange);
	storage.m_requireMeshDataDeviceUpdate = true;
}

void GeometryStorage::AllocateSkinnedMesh(const Handle& handle, const std::vector<SkinnedVertex>& skinnedVertices,
	const std::vector<glm::uvec3>& skinnedTriangles,
	const std::shared_ptr<RangeDescriptor>& targetSkinnedMeshletRange,
	const std::shared_ptr<RangeDescriptor>& targetSkinnedTriangleRange)
{
	if (skinnedVertices.empty() || skinnedTriangles.empty()) {
		throw std::runtime_error("Empty skinned vertices or skinnedTriangles!");
	}
	auto& storage = GetInstance();

	targetSkinnedMeshletRange->m_handle = handle;
	targetSkinnedMeshletRange->m_offset = storage.m_skinnedMeshlets.size();
	targetSkinnedMeshletRange->m_range = 0;

	targetSkinnedTriangleRange->m_handle = handle;
	targetSkinnedTriangleRange->m_offset = storage.m_skinnedTriangles.size();
	targetSkinnedTriangleRange->m_range = 0;
	targetSkinnedTriangleRange->m_indexCount = skinnedTriangles.size();
	std::vector<meshopt_Meshlet> skinnedMeshletsResults{};
	std::vector<unsigned> skinnedMeshletResultVertices{};
	std::vector<unsigned char> skinnedMeshletResultTriangles{};
	const auto maxMeshlets =
		meshopt_buildMeshletsBound(skinnedTriangles.size() * 3, Graphics::Constants::MESHLET_MAX_VERTICES_SIZE, Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE);
	skinnedMeshletsResults.resize(maxMeshlets);
	skinnedMeshletResultVertices.resize(maxMeshlets * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE);
	skinnedMeshletResultTriangles.resize(maxMeshlets * Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE * 3);
	const auto skinnedMeshletSize = meshopt_buildMeshlets(
		skinnedMeshletsResults.data(), skinnedMeshletResultVertices.data(), skinnedMeshletResultTriangles.data(),
		&skinnedTriangles.at(0).x, skinnedTriangles.size() * 3, &skinnedVertices.at(0).m_position.x, skinnedVertices.size(), sizeof(SkinnedVertex),
		Graphics::Constants::MESHLET_MAX_VERTICES_SIZE, Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE, 0);

	targetSkinnedMeshletRange->m_range = skinnedMeshletSize;
	for (size_t skinnedMeshletIndex = 0; skinnedMeshletIndex < skinnedMeshletSize; skinnedMeshletIndex++)
	{
		storage.m_skinnedMeshlets.emplace_back();
		auto& currentSkinnedMeshlet = storage.m_skinnedMeshlets.back();

		currentSkinnedMeshlet.m_skinnedVertexChunkIndex = storage.m_skinnedVertexDataChunks.size();
		storage.m_skinnedVertexDataChunks.emplace_back();
		auto& currentSkinnedChunk = storage.m_skinnedVertexDataChunks.back();

		const auto& skinnedMeshletResult = skinnedMeshletsResults.at(skinnedMeshletIndex);
		for (unsigned vi = 0; vi < skinnedMeshletResult.vertex_count; vi++)
		{
			currentSkinnedChunk.m_skinnedVertexData[vi] = skinnedVertices[skinnedMeshletResultVertices.at(skinnedMeshletResult.vertex_offset + vi)];
		}
		currentSkinnedMeshlet.m_skinnedVerticesSize = skinnedMeshletResult.vertex_count;
		currentSkinnedMeshlet.m_skinnedTriangleSize = skinnedMeshletResult.triangle_count;
		for (unsigned ti = 0; ti < skinnedMeshletResult.triangle_count; ti++)
		{
			auto& currentMeshletTriangle = currentSkinnedMeshlet.m_skinnedTriangles[ti];
			currentMeshletTriangle = glm::u8vec3(
				skinnedMeshletResultTriangles[ti * 3 + skinnedMeshletResult.triangle_offset],
				skinnedMeshletResultTriangles[ti * 3 + skinnedMeshletResult.triangle_offset + 1],
				skinnedMeshletResultTriangles[ti * 3 + skinnedMeshletResult.triangle_offset + 2]);

			storage.m_skinnedTriangles.emplace_back();
			auto& globalTriangle = storage.m_skinnedTriangles.back();
			globalTriangle.x = currentMeshletTriangle.x + currentSkinnedMeshlet.m_skinnedVertexChunkIndex * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
			globalTriangle.y = currentMeshletTriangle.y + currentSkinnedMeshlet.m_skinnedVertexChunkIndex * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
			globalTriangle.z = currentMeshletTriangle.z + currentSkinnedMeshlet.m_skinnedVertexChunkIndex * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
		}
		targetSkinnedTriangleRange->m_range += currentSkinnedMeshlet.m_skinnedTriangleSize;
	}
	storage.m_skinnedMeshletRangeDescriptor.push_back(targetSkinnedMeshletRange);
	storage.m_skinnedTriangleRangeDescriptor.push_back(targetSkinnedTriangleRange);
	storage.m_requireSkinnedMeshDataDeviceUpdate = true;

}

void GeometryStorage::AllocateStrands(const Handle& handle, const std::vector<StrandPoint>& strandPoints,
	const std::vector<glm::uvec4>& segments,
	const std::shared_ptr<RangeDescriptor>& targetStrandMeshletRange,
	const std::shared_ptr<RangeDescriptor>& targetSegmentRange)
{
	if (strandPoints.empty() || segments.empty()) {
		throw std::runtime_error("Empty strand points or strand segments!");
	}
	auto& storage = GetInstance();

	uint32_t currentSegmentIndex = 0;
	targetStrandMeshletRange->m_handle = handle;
	targetStrandMeshletRange->m_offset = storage.m_strandMeshlets.size();
	targetStrandMeshletRange->m_range = 0;

	targetSegmentRange->m_handle = handle;
	targetSegmentRange->m_offset = storage.m_segments.size();
	targetSegmentRange->m_range = 0;
	targetSegmentRange->m_indexCount = segments.size();

	while (currentSegmentIndex < segments.size())
	{
		targetStrandMeshletRange->m_range++;
		const uint32_t currentStrandMeshletIndex = storage.m_strandMeshlets.size();
		storage.m_strandMeshlets.emplace_back();
		auto& currentStrandMeshlet = storage.m_strandMeshlets[currentStrandMeshletIndex];

		currentStrandMeshlet.m_strandPointChunkIndex = storage.m_strandPointDataChunks.size();
		storage.m_strandPointDataChunks.emplace_back();
		auto& currentChunk = storage.m_strandPointDataChunks[currentStrandMeshlet.m_strandPointChunkIndex];

		currentStrandMeshlet.m_strandPointsSize = currentStrandMeshlet.m_segmentSize = 0;

		std::unordered_map<uint32_t, uint32_t> assignedStrandPoints{};
		while (currentStrandMeshlet.m_segmentSize < Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE && currentSegmentIndex < segments.size())
		{
			const auto& currentSegment = segments[currentSegmentIndex];
			uint32_t newStrandPointsAmount = 0;
			auto searchX = assignedStrandPoints.find(currentSegment.x);
			if (searchX == assignedStrandPoints.end()) newStrandPointsAmount++;

			auto searchY = assignedStrandPoints.find(currentSegment.y);
			if (searchY == assignedStrandPoints.end()) newStrandPointsAmount++;

			auto searchZ = assignedStrandPoints.find(currentSegment.z);
			if (searchZ == assignedStrandPoints.end()) newStrandPointsAmount++;

			auto searchW = assignedStrandPoints.find(currentSegment.w);
			if (searchW == assignedStrandPoints.end()) newStrandPointsAmount++;

			if (currentStrandMeshlet.m_strandPointsSize + newStrandPointsAmount > Graphics::Constants::MESHLET_MAX_VERTICES_SIZE)
			{
				break;
			}
			auto& currentStrandMeshletSegment = currentStrandMeshlet.m_segments[currentStrandMeshlet.m_segmentSize];

			if (searchX != assignedStrandPoints.end())
			{
				currentStrandMeshletSegment.x = searchX->second;
			}
			else
			{
				//Add current strandPoint index into the map.
				assignedStrandPoints[currentSegment.x] = currentStrandMeshlet.m_strandPointsSize;

				//Assign new strandPoint in strandMeshlet, and retrieve actual strandPoint index in strandPoint data chunks.
				currentChunk.m_strandPointData[currentStrandMeshlet.m_strandPointsSize] = strandPoints[currentSegment.x];
				currentStrandMeshletSegment.x = currentStrandMeshlet.m_strandPointsSize;
				currentStrandMeshlet.m_strandPointsSize++;
			}

			searchY = assignedStrandPoints.find(currentSegment.y);
			if (searchY != assignedStrandPoints.end())
			{
				currentStrandMeshletSegment.y = searchY->second;
			}
			else
			{
				//Add current strandPoint index into the map.
				assignedStrandPoints[currentSegment.y] = currentStrandMeshlet.m_strandPointsSize;

				//Assign new strandPoint in strandMeshlet, and retrieve actual strandPoint index in strandPoint data chunks.
				currentChunk.m_strandPointData[currentStrandMeshlet.m_strandPointsSize] = strandPoints[currentSegment.y];
				currentStrandMeshletSegment.y = currentStrandMeshlet.m_strandPointsSize;
				currentStrandMeshlet.m_strandPointsSize++;
			}

			searchZ = assignedStrandPoints.find(currentSegment.z);
			if (searchZ != assignedStrandPoints.end())
			{
				currentStrandMeshletSegment.z = searchZ->second;
			}
			else
			{
				//Add current strandPoint index into the map.
				assignedStrandPoints[currentSegment.z] = currentStrandMeshlet.m_strandPointsSize;

				//Assign new strandPoint in strandMeshlet, and retrieve actual strandPoint index in strandPoint data chunks.
				currentChunk.m_strandPointData[currentStrandMeshlet.m_strandPointsSize] = strandPoints[currentSegment.z];
				currentStrandMeshletSegment.z = currentStrandMeshlet.m_strandPointsSize;
				currentStrandMeshlet.m_strandPointsSize++;
			}

			searchW = assignedStrandPoints.find(currentSegment.w);
			if (searchW != assignedStrandPoints.end())
			{
				currentStrandMeshletSegment.w = searchW->second;
			}
			else
			{
				//Add current strandPoint index into the map.
				assignedStrandPoints[currentSegment.w] = currentStrandMeshlet.m_strandPointsSize;

				//Assign new strandPoint in strandMeshlet, and retrieve actual strandPoint index in strandPoint data chunks.
				currentChunk.m_strandPointData[currentStrandMeshlet.m_strandPointsSize] = strandPoints[currentSegment.w];
				currentStrandMeshletSegment.w = currentStrandMeshlet.m_strandPointsSize;
				currentStrandMeshlet.m_strandPointsSize++;
			}
			currentStrandMeshlet.m_segmentSize++;
			currentSegmentIndex++;

			auto& globalSegment = storage.m_segments.emplace_back();
			globalSegment.x = currentStrandMeshletSegment.x + currentStrandMeshlet.m_strandPointChunkIndex * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
			globalSegment.y = currentStrandMeshletSegment.y + currentStrandMeshlet.m_strandPointChunkIndex * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
			globalSegment.z = currentStrandMeshletSegment.z + currentStrandMeshlet.m_strandPointChunkIndex * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
			globalSegment.w = currentStrandMeshletSegment.w + currentStrandMeshlet.m_strandPointChunkIndex * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
			targetSegmentRange->m_range++;
		}
	}

	storage.m_strandMeshletRangeDescriptor.push_back(targetStrandMeshletRange);
	storage.m_segmentRangeDescriptor.push_back(targetSegmentRange);
	storage.m_requireStrandMeshDataDeviceUpdate = true;

}

void GeometryStorage::FreeMesh(const Handle& handle)
{
	auto& storage = GetInstance();
	uint32_t meshletRangeDescriptorIndex = UINT_MAX;
	for (int i = 0; i < storage.m_meshletRangeDescriptor.size(); i++)
	{
		if (storage.m_meshletRangeDescriptor[i]->m_handle == handle)
		{
			meshletRangeDescriptorIndex = i;
			break;
		}
	}
	if (meshletRangeDescriptorIndex == UINT_MAX)
	{
		return;
	}
	const auto& meshletRangeDescriptor = storage.m_meshletRangeDescriptor[meshletRangeDescriptorIndex];
	const uint32_t removeChunkSize = meshletRangeDescriptor->m_range;
	storage.m_meshlets.erase(storage.m_meshlets.begin() + meshletRangeDescriptor->m_offset, storage.m_meshlets.begin() + meshletRangeDescriptor->m_offset + removeChunkSize);
	storage.m_vertexDataChunks.erase(storage.m_vertexDataChunks.begin() + meshletRangeDescriptor->m_offset,
		storage.m_vertexDataChunks.begin() + meshletRangeDescriptor->m_offset + removeChunkSize);
	for (uint32_t i = meshletRangeDescriptorIndex; i < storage.m_meshlets.size(); i++)
	{
		storage.m_meshlets[i].m_vertexChunkIndex = i;
	}
	for (uint32_t i = meshletRangeDescriptorIndex + 1; i < storage.m_meshletRangeDescriptor.size(); i++)
	{
		assert(storage.m_meshletRangeDescriptor[i]->m_offset >= meshletRangeDescriptor->m_range);
		storage.m_meshletRangeDescriptor[i]->m_offset -= meshletRangeDescriptor->m_range;
	}
	storage.m_meshletRangeDescriptor.erase(storage.m_meshletRangeDescriptor.begin() + meshletRangeDescriptorIndex);

	uint32_t triangleRangeDescriptorIndex = UINT_MAX;
	for (uint32_t i = 0; i < storage.m_triangleRangeDescriptor.size(); i++)
	{
		if (storage.m_triangleRangeDescriptor[i]->m_handle == handle)
		{
			triangleRangeDescriptorIndex = i;
			break;
		}
	}
	if (triangleRangeDescriptorIndex == UINT_MAX)
	{
		return;
	}
	const auto& triangleRangeDescriptor = storage.m_triangleRangeDescriptor[triangleRangeDescriptorIndex];
	storage.m_triangles.erase(storage.m_triangles.begin() + triangleRangeDescriptor->m_offset,
		storage.m_triangles.begin() + triangleRangeDescriptor->m_offset + triangleRangeDescriptor->m_range);
	for (uint32_t i = triangleRangeDescriptorIndex + 1; i < storage.m_triangleRangeDescriptor.size(); i++)
	{
		assert(storage.m_triangleRangeDescriptor[i]->m_offset >= triangleRangeDescriptor->m_range);
		storage.m_triangleRangeDescriptor[i]->m_offset -= triangleRangeDescriptor->m_range;
	}

	for (uint32_t i = triangleRangeDescriptor->m_offset; i < storage.m_triangles.size(); i++)
	{
		storage.m_triangles[i].x -= removeChunkSize * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
		storage.m_triangles[i].y -= removeChunkSize * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
		storage.m_triangles[i].z -= removeChunkSize * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
	}
	storage.m_triangleRangeDescriptor.erase(storage.m_triangleRangeDescriptor.begin() + triangleRangeDescriptorIndex);
	storage.m_requireMeshDataDeviceUpdate = true;
}

void GeometryStorage::FreeSkinnedMesh(const Handle& handle)
{
	auto& storage = GetInstance();
	uint32_t skinnedMeshletRangeDescriptorIndex = UINT_MAX;
	for (int i = 0; i < storage.m_skinnedMeshletRangeDescriptor.size(); i++)
	{
		if (storage.m_skinnedMeshletRangeDescriptor[i]->m_handle == handle)
		{
			skinnedMeshletRangeDescriptorIndex = i;
			break;
		}
	}
	if (skinnedMeshletRangeDescriptorIndex == UINT_MAX)
	{
		return;
	};
	const auto& skinnedMeshletRangeDescriptor = storage.m_skinnedMeshletRangeDescriptor[skinnedMeshletRangeDescriptorIndex];
	const uint32_t removeChunkSize = skinnedMeshletRangeDescriptor->m_range;
	storage.m_skinnedMeshlets.erase(storage.m_skinnedMeshlets.begin() + skinnedMeshletRangeDescriptor->m_offset, storage.m_skinnedMeshlets.begin() + removeChunkSize);
	storage.m_skinnedVertexDataChunks.erase(storage.m_skinnedVertexDataChunks.begin() + skinnedMeshletRangeDescriptor->m_offset,
		storage.m_skinnedVertexDataChunks.begin() + skinnedMeshletRangeDescriptor->m_offset + removeChunkSize);
	for (uint32_t i = skinnedMeshletRangeDescriptorIndex; i < storage.m_skinnedMeshlets.size(); i++)
	{
		storage.m_skinnedMeshlets[i].m_skinnedVertexChunkIndex = i;
	}
	for (uint32_t i = skinnedMeshletRangeDescriptorIndex + 1; i < storage.m_skinnedMeshletRangeDescriptor.size(); i++)
	{
		assert(storage.m_skinnedMeshletRangeDescriptor[i]->m_offset >= skinnedMeshletRangeDescriptor->m_range);
		storage.m_skinnedMeshletRangeDescriptor[i]->m_offset -= skinnedMeshletRangeDescriptor->m_range;
	}
	storage.m_skinnedMeshletRangeDescriptor.erase(storage.m_skinnedMeshletRangeDescriptor.begin() + skinnedMeshletRangeDescriptorIndex);

	uint32_t skinnedTriangleRangeDescriptorIndex = UINT_MAX;
	for (uint32_t i = 0; i < storage.m_skinnedTriangleRangeDescriptor.size(); i++)
	{
		if (storage.m_skinnedTriangleRangeDescriptor[i]->m_handle == handle)
		{
			skinnedTriangleRangeDescriptorIndex = i;
			break;
		}
	}
	if (skinnedTriangleRangeDescriptorIndex == UINT_MAX)
	{
		return;
	}
	const auto& skinnedTriangleRangeDescriptor = storage.m_skinnedTriangleRangeDescriptor[skinnedTriangleRangeDescriptorIndex];
	storage.m_skinnedTriangles.erase(storage.m_skinnedTriangles.begin() + skinnedTriangleRangeDescriptor->m_offset,
		storage.m_skinnedTriangles.begin() + skinnedTriangleRangeDescriptor->m_offset + skinnedTriangleRangeDescriptor->m_range);
	for (uint32_t i = skinnedTriangleRangeDescriptorIndex + 1; i < storage.m_skinnedTriangleRangeDescriptor.size(); i++)
	{
		assert(storage.m_skinnedTriangleRangeDescriptor[i]->m_offset >= skinnedTriangleRangeDescriptor->m_range);
		storage.m_skinnedTriangleRangeDescriptor[i]->m_offset -= skinnedTriangleRangeDescriptor->m_range;
	}

	for (uint32_t i = skinnedTriangleRangeDescriptor->m_offset; i < storage.m_skinnedTriangles.size(); i++)
	{
		storage.m_skinnedTriangles[i].x -= removeChunkSize * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
		storage.m_skinnedTriangles[i].y -= removeChunkSize * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
		storage.m_skinnedTriangles[i].z -= removeChunkSize * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
	}

	storage.m_skinnedTriangleRangeDescriptor.erase(storage.m_skinnedTriangleRangeDescriptor.begin() + skinnedTriangleRangeDescriptorIndex);
	storage.m_requireSkinnedMeshDataDeviceUpdate = true;
}

void GeometryStorage::FreeStrands(const Handle& handle)
{
	auto& storage = GetInstance();
	uint32_t strandMeshletRangeDescriptorIndex = UINT_MAX;
	for (int i = 0; i < storage.m_strandMeshletRangeDescriptor.size(); i++)
	{
		if (storage.m_strandMeshletRangeDescriptor[i]->m_handle == handle)
		{
			strandMeshletRangeDescriptorIndex = i;
			break;
		}
	}
	if (strandMeshletRangeDescriptorIndex == UINT_MAX)
	{
		return;
	}
	const auto& strandMeshletRangeDescriptor = storage.m_strandMeshletRangeDescriptor[strandMeshletRangeDescriptorIndex];
	const uint32_t removeChunkSize = strandMeshletRangeDescriptor->m_range;
	storage.m_strandMeshlets.erase(storage.m_strandMeshlets.begin() + strandMeshletRangeDescriptor->m_offset, storage.m_strandMeshlets.begin() + strandMeshletRangeDescriptor->m_offset + removeChunkSize);
	storage.m_strandPointDataChunks.erase(storage.m_strandPointDataChunks.begin() + strandMeshletRangeDescriptor->m_offset,
		storage.m_strandPointDataChunks.begin() + strandMeshletRangeDescriptor->m_offset + removeChunkSize);
	for (uint32_t i = strandMeshletRangeDescriptorIndex; i < storage.m_strandMeshlets.size(); i++)
	{
		storage.m_strandMeshlets[i].m_strandPointChunkIndex = i;
	}
	for (uint32_t i = strandMeshletRangeDescriptorIndex + 1; i < storage.m_strandMeshletRangeDescriptor.size(); i++)
	{
		assert(storage.m_strandMeshletRangeDescriptor[i]->m_offset >= strandMeshletRangeDescriptor->m_range);
		storage.m_strandMeshletRangeDescriptor[i]->m_offset -= strandMeshletRangeDescriptor->m_range;
	}
	storage.m_strandMeshletRangeDescriptor.erase(storage.m_strandMeshletRangeDescriptor.begin() + strandMeshletRangeDescriptorIndex);

	uint32_t segmentRangeDescriptorIndex = UINT_MAX;
	for (uint32_t i = 0; i < storage.m_segmentRangeDescriptor.size(); i++)
	{
		if (storage.m_segmentRangeDescriptor[i]->m_handle == handle)
		{
			segmentRangeDescriptorIndex = i;
			break;
		}
	}
	if (segmentRangeDescriptorIndex == UINT_MAX)
	{
		return;
	}
	const auto& segmentRangeDescriptor = storage.m_segmentRangeDescriptor[segmentRangeDescriptorIndex];
	storage.m_segments.erase(storage.m_segments.begin() + segmentRangeDescriptor->m_offset,
		storage.m_segments.begin() + segmentRangeDescriptor->m_offset + segmentRangeDescriptor->m_range);
	for (uint32_t i = segmentRangeDescriptorIndex + 1; i < storage.m_segmentRangeDescriptor.size(); i++)
	{
		assert(storage.m_segmentRangeDescriptor[i]->m_offset >= segmentRangeDescriptor->m_range);
		storage.m_segmentRangeDescriptor[i]->m_offset -= segmentRangeDescriptor->m_range;
	}

	for (uint32_t i = segmentRangeDescriptor->m_offset; i < storage.m_segments.size(); i++)
	{
		storage.m_segments[i].x -= removeChunkSize * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
		storage.m_segments[i].y -= removeChunkSize * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
		storage.m_segments[i].z -= removeChunkSize * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
		storage.m_segments[i].w -= removeChunkSize * Graphics::Constants::MESHLET_MAX_VERTICES_SIZE;
	}
	storage.m_segmentRangeDescriptor.erase(storage.m_segmentRangeDescriptor.begin() + segmentRangeDescriptorIndex);

	storage.m_requireStrandMeshDataDeviceUpdate = true;
}

void GeometryStorage::AllocateParticleInfo(const Handle& handle,
	const std::shared_ptr<RangeDescriptor>& rangeDescriptor)
{
	auto& storage = GetInstance();
	storage.m_particleInfoListDataList.emplace_back();
	auto& infoData = storage.m_particleInfoListDataList.back();
	infoData.m_rangeDescriptor = rangeDescriptor;
	infoData.m_rangeDescriptor->m_offset = storage.m_particleInfoListDataList.size() - 1;
	infoData.m_rangeDescriptor->m_handle = handle;
	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.size = sizeof(ParticleInfo);
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VmaAllocationCreateInfo bufferVmaAllocationCreateInfo{};
	bufferVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	infoData.m_buffer = std::make_shared<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo);
	infoData.m_descriptorSet = std::make_shared<DescriptorSet>(Graphics::GetDescriptorSetLayout("INSTANCED_DATA_LAYOUT"));
	infoData.m_status = ParticleInfoListDataStatus::UpdatePending;
}

void GeometryStorage::UpdateParticleInfo(const std::shared_ptr<RangeDescriptor>& rangeDescriptor,
	const std::vector<ParticleInfo>& particleInfos)
{
	auto& storage = GetInstance();
	assert(rangeDescriptor->m_offset < storage.m_particleInfoListDataList.size());
	auto& infoData = storage.m_particleInfoListDataList.at(rangeDescriptor->m_offset);
	assert(infoData.m_status != ParticleInfoListDataStatus::Removed);
	infoData.m_particleInfoList = particleInfos;
	infoData.m_status = ParticleInfoListDataStatus::UpdatePending;
}

void GeometryStorage::FreeParticleInfo(const std::shared_ptr<RangeDescriptor>& rangeDescriptor)
{
	auto& storage = GetInstance();
	assert(rangeDescriptor->m_offset < storage.m_particleInfoListDataList.size());
	auto& infoData = storage.m_particleInfoListDataList.at(rangeDescriptor->m_offset);
	assert(infoData.m_status != ParticleInfoListDataStatus::Removed);
	infoData.m_status = ParticleInfoListDataStatus::Removed;
}

const std::vector<ParticleInfo>& GeometryStorage::PeekParticleInfoList(const std::shared_ptr<RangeDescriptor>& rangeDescriptor)
{
	const auto& storage = GetInstance();
	return storage.m_particleInfoListDataList[rangeDescriptor->m_offset].m_particleInfoList;
}

const std::shared_ptr<DescriptorSet>& GeometryStorage::PeekDescriptorSet(const std::shared_ptr<RangeDescriptor>& rangeDescriptor)
{
	const auto& storage = GetInstance();
	return storage.m_particleInfoListDataList[rangeDescriptor->m_offset].m_descriptorSet;
}

const Meshlet& GeometryStorage::PeekMeshlet(const uint32_t meshletIndex)
{
	const auto& storage = GetInstance();
	return storage.m_meshlets[meshletIndex];
}

const SkinnedMeshlet& GeometryStorage::PeekSkinnedMeshlet(const uint32_t skinnedMeshletIndex)
{
	const auto& storage = GetInstance();
	return storage.m_skinnedMeshlets[skinnedMeshletIndex];
}

const StrandMeshlet& GeometryStorage::PeekStrandMeshlet(const uint32_t strandMeshletIndex)
{
	const auto& storage = GetInstance();
	return storage.m_strandMeshlets[strandMeshletIndex];
}
