#include "GeometryStorage.hpp"
using namespace EvoEngine;

void GeometryStorage::UploadData()
{
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	if (m_requireMeshDataDeviceUpdate[currentFrameIndex]) {
		m_vertexBuffer[currentFrameIndex]->UploadVector(m_vertexDataChunks);
		m_meshletBuffer[currentFrameIndex]->UploadVector(m_meshlets);
		m_requireMeshDataDeviceUpdate[currentFrameIndex] = false;
	}
	if (m_requireSkinnedMeshDataDeviceUpdate[currentFrameIndex]) {
		m_skinnedVertexBuffer[currentFrameIndex]->UploadVector(m_skinnedVertexDataChunks);
		m_skinnedMeshletBuffer[currentFrameIndex]->UploadVector(m_skinnedMeshlets);
		m_requireSkinnedMeshDataDeviceUpdate[currentFrameIndex] = false;
	}
	if (m_requireStrandMeshDataDeviceUpdate[currentFrameIndex]) {
		m_strandPointBuffer[currentFrameIndex]->UploadVector(m_strandPointDataChunks);
		m_strandMeshletBuffer[currentFrameIndex]->UploadVector(m_strandMeshlets);
		m_requireStrandMeshDataDeviceUpdate[currentFrameIndex] = false;
	}
}

void GeometryStorage::PreUpdate()
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
	storageBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	storageBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo verticesVmaAllocationCreateInfo{};
	verticesVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	const auto maxFrameInFlight = Graphics::GetMaxFramesInFlight();
	for (int i = 0; i < maxFrameInFlight; i++) {
		storage.m_vertexBuffer.emplace_back(std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo));
		storage.m_meshletBuffer.emplace_back(std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo));
	}
	storage.m_requireMeshDataDeviceUpdate.resize(maxFrameInFlight);

	for (int i = 0; i < maxFrameInFlight; i++) {
		storage.m_skinnedVertexBuffer.emplace_back(std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo));
		storage.m_skinnedMeshletBuffer.emplace_back(std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo));
	}
	storage.m_requireSkinnedMeshDataDeviceUpdate.resize(maxFrameInFlight);

	for (int i = 0; i < maxFrameInFlight; i++) {
		storage.m_strandPointBuffer.emplace_back(std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo));
		storage.m_strandMeshletBuffer.emplace_back(std::make_unique<Buffer>(storageBufferCreateInfo, verticesVmaAllocationCreateInfo));
	}
	storage.m_requireStrandMeshDataDeviceUpdate.resize(maxFrameInFlight);
}

const std::unique_ptr<Buffer>& GeometryStorage::GetVertexBuffer()
{
	const auto& storage = GetInstance();
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	return storage.m_vertexBuffer[currentFrameIndex];
}

const std::unique_ptr<Buffer>& GeometryStorage::GetMeshletBuffer()
{
	const auto& storage = GetInstance();
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	return storage.m_meshletBuffer[currentFrameIndex];
}

const std::unique_ptr<Buffer>& GeometryStorage::GetSkinnedVertexBuffer()
{
	const auto& storage = GetInstance();
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	return storage.m_skinnedVertexBuffer[currentFrameIndex];
}

const std::unique_ptr<Buffer>& GeometryStorage::GetSkinnedMeshletBuffer()
{
	const auto& storage = GetInstance();
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	return storage.m_skinnedMeshletBuffer[currentFrameIndex];
}

const std::unique_ptr<Buffer>& GeometryStorage::GetStrandPointBuffer()
{
	const auto& storage = GetInstance();
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	return storage.m_strandPointBuffer[currentFrameIndex];
}

const std::unique_ptr<Buffer>& GeometryStorage::GetStrandMeshletBuffer()
{
	const auto& storage = GetInstance();
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	return storage.m_strandMeshletBuffer[currentFrameIndex];
}

void GeometryStorage::BindVertices(const VkCommandBuffer commandBuffer)
{
	const auto& storage = GetInstance();
	constexpr VkDeviceSize offsets[] = { 0 };
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &storage.m_vertexBuffer[currentFrameIndex]->GetVkBuffer(), offsets);
}

void GeometryStorage::BindSkinnedVertices(const VkCommandBuffer commandBuffer)
{
	const auto& storage = GetInstance();
	constexpr VkDeviceSize offsets[] = { 0 };
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &storage.m_skinnedVertexBuffer[currentFrameIndex]->GetVkBuffer(), offsets);
}

void GeometryStorage::BindStrandPoints(const VkCommandBuffer commandBuffer)
{
	const auto& storage = GetInstance();
	constexpr VkDeviceSize offsets[] = { 0 };
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &storage.m_strandPointBuffer[currentFrameIndex]->GetVkBuffer(), offsets);
}

const Vertex& GeometryStorage::PeekVertex(const size_t vertexIndex)
{
	const auto& storage = GetInstance();
	return storage.m_vertexDataChunks[vertexIndex / Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE].m_vertexData[vertexIndex % Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE];
}

const SkinnedVertex& GeometryStorage::PeekSkinnedVertex(const size_t skinnedVertexIndex)
{
	const auto& storage = GetInstance();
	return storage.m_skinnedVertexDataChunks[skinnedVertexIndex / Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE].m_skinnedVertexData[skinnedVertexIndex % Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE];
}

const StrandPoint& GeometryStorage::PeekStrandPoint(const size_t strandPointIndex)
{
	const auto& storage = GetInstance();
	return storage.m_strandPointDataChunks[strandPointIndex / Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE].m_strandPointData[strandPointIndex % Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE];
}

void GeometryStorage::AllocateMesh(const std::vector<Vertex>& vertices, const std::vector<glm::uvec3>& triangles,
	std::vector<uint32_t>& meshletIndices)
{
	if (vertices.empty() || triangles.empty()) return;
	auto& storage = GetInstance();
	/* From mesh vertices to vertex data indices
	 */
	std::vector<uint32_t> vertexIndices;
	vertexIndices.resize(vertices.size());
	
	for (uint32_t i = 0; i < vertexIndices.size(); i++)
	{
		size_t currentVertexIndex;
		if (!storage.m_vertexDataVertexPool.empty())
		{
			currentVertexIndex = storage.m_vertexDataVertexPool.front();
			storage.m_vertexDataVertexPool.pop();
		}
		else
		{
			currentVertexIndex = storage.m_verticesCount;
			storage.m_verticesCount++;
		}

		vertexIndices[i] = currentVertexIndex;
		if (currentVertexIndex / Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE >= storage.m_vertexDataChunks.size())
		{
			storage.m_vertexDataChunks.emplace_back();
		}
		storage.m_vertexDataChunks[currentVertexIndex / Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE].m_vertexData[currentVertexIndex % Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE] = vertices[i];
	}

	meshletIndices.clear();
	uint32_t currentTriangleIndex = 0;
	while (currentTriangleIndex < triangles.size())
	{
		uint32_t currentMeshletIndex;
		if (!storage.m_meshletPool.empty())
		{
			currentMeshletIndex = storage.m_meshletPool.front();
			storage.m_meshletPool.pop();
		}
		else
		{
			currentMeshletIndex = storage.m_meshlets.size();
			storage.m_meshlets.emplace_back();
		}
		auto& currentMeshlet = storage.m_meshlets[currentMeshletIndex];
		currentMeshlet.m_verticesSize = currentMeshlet.m_triangleSize = 0;

		std::unordered_map<uint32_t, uint32_t> assignedVertices{};
		while (currentMeshlet.m_triangleSize < Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE && currentTriangleIndex < triangles.size())
		{
			const auto& currentTriangle = triangles[currentTriangleIndex];
			uint32_t newVerticesAmount = 0;
			auto searchX = assignedVertices.find(currentTriangle.x);
			if (searchX == assignedVertices.end()) newVerticesAmount++;

			auto searchY = assignedVertices.find(currentTriangle.y);
			if (searchY == assignedVertices.end()) newVerticesAmount++;

			auto searchZ = assignedVertices.find(currentTriangle.z);
			if (searchZ == assignedVertices.end()) newVerticesAmount++;

			if (currentMeshlet.m_verticesSize + newVerticesAmount > Graphics::Constants::MESHLET_MAX_VERTICES_SIZE)
			{
				break;
			}
			auto& currentMeshletTriangle = currentMeshlet.m_triangles[currentMeshlet.m_triangleSize];

			if (searchX != assignedVertices.end())
			{
				currentMeshletTriangle.x = searchX->second;
			}
			else
			{
				//Add current vertex index into the map.
				assignedVertices[currentTriangle.x] = currentMeshlet.m_verticesSize;

				//Assign new vertex in meshlet, and retrieve actual vertex index in vertex data chunks.
				currentMeshlet.m_vertexIndices[currentMeshlet.m_verticesSize] = vertexIndices[currentTriangle.x];
				currentMeshletTriangle.x = currentMeshlet.m_verticesSize;
				currentMeshlet.m_verticesSize++;
			}

			searchY = assignedVertices.find(currentTriangle.y);
			if (searchY != assignedVertices.end())
			{
				currentMeshletTriangle.y = searchY->second;
			}
			else
			{
				//Add current vertex index into the map.
				assignedVertices[currentTriangle.y] = currentMeshlet.m_verticesSize;

				//Assign new vertex in meshlet, and retrieve actual vertex index in vertex data chunks.
				currentMeshlet.m_vertexIndices[currentMeshlet.m_verticesSize] = vertexIndices[currentTriangle.y];
				currentMeshletTriangle.y = currentMeshlet.m_verticesSize;
				currentMeshlet.m_verticesSize++;
			}

			searchZ = assignedVertices.find(currentTriangle.z);
			if (searchZ != assignedVertices.end())
			{
				currentMeshletTriangle.z = searchZ->second;
			}
			else
			{
				//Add current vertex index into the map.
				assignedVertices[currentTriangle.z] = currentMeshlet.m_verticesSize;

				//Assign new vertex in meshlet, and retrieve actual vertex index in vertex data chunks.
				currentMeshlet.m_vertexIndices[currentMeshlet.m_verticesSize] = vertexIndices[currentTriangle.z];
				currentMeshletTriangle.z = currentMeshlet.m_verticesSize;
				currentMeshlet.m_verticesSize++;
			}
			currentMeshlet.m_triangleSize++;
			currentTriangleIndex++;
		}
		meshletIndices.emplace_back(currentMeshletIndex);
	}
	for (auto& i : storage.m_requireMeshDataDeviceUpdate) i = true;
}

void GeometryStorage::AllocateSkinnedMesh(const std::vector<SkinnedVertex>& skinnedVertices,
	const std::vector<glm::uvec3>& triangles, std::vector<uint32_t>& skinnedMeshletIndices)
{
	if (skinnedVertices.empty() || triangles.empty()) return;
	auto& storage = GetInstance();
	/* From mesh skinnedVertices to skinnedVertex data indices
	 */
	std::vector<uint32_t> skinnedVertexIndices;
	skinnedVertexIndices.resize(skinnedVertices.size());
	for (uint32_t i = 0; i < skinnedVertexIndices.size(); i++)
	{
		size_t currentSkinnedVertexIndex;
		if (!storage.m_skinnedVertexDataPool.empty())
		{
			currentSkinnedVertexIndex = storage.m_skinnedVertexDataPool.front();
			storage.m_skinnedVertexDataPool.pop();
		}
		else
		{
			currentSkinnedVertexIndex = storage.m_skinnedVerticesCount;
			storage.m_skinnedVerticesCount++;
		}

		skinnedVertexIndices[i] = currentSkinnedVertexIndex;
		if (currentSkinnedVertexIndex / Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE >= storage.m_skinnedVertexDataChunks.size())
		{
			storage.m_skinnedVertexDataChunks.emplace_back();
		}
		storage.m_skinnedVertexDataChunks[currentSkinnedVertexIndex / Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE].m_skinnedVertexData[currentSkinnedVertexIndex % Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE] = skinnedVertices[i];
	}

	skinnedMeshletIndices.clear();
	uint32_t currentTriangleIndex = 0;
	while (currentTriangleIndex < triangles.size())
	{
		uint32_t currentSkinnedMeshletIndex;
		if (!storage.m_skinnedMeshletPool.empty())
		{
			currentSkinnedMeshletIndex = storage.m_skinnedMeshletPool.front();
			storage.m_skinnedMeshletPool.pop();
		}
		else
		{
			currentSkinnedMeshletIndex = storage.m_skinnedMeshlets.size();
			storage.m_skinnedMeshlets.emplace_back();
		}
		auto& currentSkinnedMeshlet = storage.m_skinnedMeshlets[currentSkinnedMeshletIndex];
		currentSkinnedMeshlet.m_skinnedVerticesSize = currentSkinnedMeshlet.m_triangleSize = 0;

		std::unordered_map<uint32_t, uint32_t> assignedSkinnedVertices{};
		while (currentSkinnedMeshlet.m_triangleSize < Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE && currentTriangleIndex < triangles.size())
		{
			const auto& currentTriangle = triangles[currentTriangleIndex];
			uint32_t newSkinnedVerticesAmount = 0;
			auto searchX = assignedSkinnedVertices.find(currentTriangle.x);
			if (searchX == assignedSkinnedVertices.end()) newSkinnedVerticesAmount++;

			auto searchY = assignedSkinnedVertices.find(currentTriangle.y);
			if (searchY == assignedSkinnedVertices.end()) newSkinnedVerticesAmount++;

			auto searchZ = assignedSkinnedVertices.find(currentTriangle.z);
			if (searchZ == assignedSkinnedVertices.end()) newSkinnedVerticesAmount++;

			if (currentSkinnedMeshlet.m_skinnedVerticesSize + newSkinnedVerticesAmount > Graphics::Constants::MESHLET_MAX_VERTICES_SIZE)
			{
				break;
			}
			auto& currentSkinnedMeshletTriangle = currentSkinnedMeshlet.m_triangles[currentSkinnedMeshlet.m_triangleSize];

			if (searchX != assignedSkinnedVertices.end())
			{
				currentSkinnedMeshletTriangle.x = searchX->second;
			}
			else
			{
				//Add current skinnedVertex index into the map.
				assignedSkinnedVertices[currentTriangle.x] = currentSkinnedMeshlet.m_skinnedVerticesSize;

				//Assign new skinnedVertex in skinnedMeshlet, and retrieve actual skinnedVertex index in skinnedVertex data chunks.
				currentSkinnedMeshlet.m_skinnedVertexIndices[currentSkinnedMeshlet.m_skinnedVerticesSize] = skinnedVertexIndices[currentTriangle.x];
				currentSkinnedMeshletTriangle.x = currentSkinnedMeshlet.m_skinnedVerticesSize;
				currentSkinnedMeshlet.m_skinnedVerticesSize++;
			}

			searchY = assignedSkinnedVertices.find(currentTriangle.y);
			if (searchY != assignedSkinnedVertices.end())
			{
				currentSkinnedMeshletTriangle.y = searchY->second;
			}
			else
			{
				//Add current skinnedVertex index into the map.
				assignedSkinnedVertices[currentTriangle.y] = currentSkinnedMeshlet.m_skinnedVerticesSize;

				//Assign new skinnedVertex in skinnedMeshlet, and retrieve actual skinnedVertex index in skinnedVertex data chunks.
				currentSkinnedMeshlet.m_skinnedVertexIndices[currentSkinnedMeshlet.m_skinnedVerticesSize] = skinnedVertexIndices[currentTriangle.y];
				currentSkinnedMeshletTriangle.y = currentSkinnedMeshlet.m_skinnedVerticesSize;
				currentSkinnedMeshlet.m_skinnedVerticesSize++;
			}

			searchZ = assignedSkinnedVertices.find(currentTriangle.z);
			if (searchZ != assignedSkinnedVertices.end())
			{
				currentSkinnedMeshletTriangle.z = searchZ->second;
			}
			else
			{
				//Add current skinnedVertex index into the map.
				assignedSkinnedVertices[currentTriangle.z] = currentSkinnedMeshlet.m_skinnedVerticesSize;

				//Assign new skinnedVertex in skinnedMeshlet, and retrieve actual skinnedVertex index in skinnedVertex data chunks.
				currentSkinnedMeshlet.m_skinnedVertexIndices[currentSkinnedMeshlet.m_skinnedVerticesSize] = skinnedVertexIndices[currentTriangle.z];
				currentSkinnedMeshletTriangle.z = currentSkinnedMeshlet.m_skinnedVerticesSize;
				currentSkinnedMeshlet.m_skinnedVerticesSize++;
			}
			currentSkinnedMeshlet.m_triangleSize++;
			currentTriangleIndex++;
		}
		skinnedMeshletIndices.emplace_back(currentSkinnedMeshletIndex);
	}
	for (auto& i : storage.m_requireSkinnedMeshDataDeviceUpdate) i = true;
}

void GeometryStorage::AllocateStrands(const std::vector<StrandPoint>& strandPoints,
	const std::vector<glm::uvec4>& segments, std::vector<uint32_t>& strandMeshletIndices)
{
	if (strandPoints.empty() || segments.empty()) return;
	auto& storage = GetInstance();
	/* From mesh strandPoints to strandPoint data indices
	 */
	std::vector<uint32_t> strandPointIndices;
	strandPointIndices.resize(strandPoints.size());
	for (uint32_t i = 0; i < strandPointIndices.size(); i++)
	{
		size_t currentStrandPointIndex;
		if (!storage.m_strandPointDataPool.empty())
		{
			currentStrandPointIndex = storage.m_strandPointDataPool.front();
			storage.m_strandPointDataPool.pop();
		}
		else
		{
			currentStrandPointIndex = storage.m_strandPointsCount;
			storage.m_strandPointsCount++;
		}

		strandPointIndices[i] = currentStrandPointIndex;
		if (currentStrandPointIndex / Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE >= storage.m_strandPointDataChunks.size())
		{
			storage.m_strandPointDataChunks.emplace_back();
		}
		storage.m_strandPointDataChunks[currentStrandPointIndex / Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE].m_strandPointData[currentStrandPointIndex % Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE] = strandPoints[i];
	}

	strandMeshletIndices.clear();
	uint32_t currentSegmentIndex = 0;
	while (currentSegmentIndex < segments.size())
	{
		uint32_t currentStrandMeshletIndex;
		if (!storage.m_strandMeshletPool.empty())
		{
			currentStrandMeshletIndex = storage.m_strandMeshletPool.front();
			storage.m_strandMeshletPool.pop();
		}
		else
		{
			currentStrandMeshletIndex = storage.m_strandMeshlets.size();
			storage.m_strandMeshlets.emplace_back();
		}
		auto& currentStrandMeshlet = storage.m_strandMeshlets[currentStrandMeshletIndex];
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
				currentStrandMeshlet.m_strandPointIndices[currentStrandMeshlet.m_strandPointsSize] = strandPointIndices[currentSegment.x];
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
				currentStrandMeshlet.m_strandPointIndices[currentStrandMeshlet.m_strandPointsSize] = strandPointIndices[currentSegment.y];
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
				currentStrandMeshlet.m_strandPointIndices[currentStrandMeshlet.m_strandPointsSize] = strandPointIndices[currentSegment.z];
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
				currentStrandMeshlet.m_strandPointIndices[currentStrandMeshlet.m_strandPointsSize] = strandPointIndices[currentSegment.w];
				currentStrandMeshletSegment.w = currentStrandMeshlet.m_strandPointsSize;
				currentStrandMeshlet.m_strandPointsSize++;
			}
			currentStrandMeshlet.m_segmentSize++;
			currentSegmentIndex++;
		}
		strandMeshletIndices.emplace_back(currentStrandMeshletIndex);
	}
	for (auto& i : storage.m_requireStrandMeshDataDeviceUpdate) i = true;
}

void GeometryStorage::FreeMesh(const std::vector<uint32_t>& meshletIndices)
{
	auto& storage = GetInstance();
	std::unordered_set<uint32_t> vertexIndices;
	for (const auto& i : meshletIndices)
	{
		storage.m_meshletPool.emplace(i);
		auto& meshlet = storage.m_meshlets[i];
		for (int j = 0; j < meshlet.m_verticesSize; j++)
		{
			vertexIndices.emplace(meshlet.m_vertexIndices[j]);
		}
		meshlet.m_verticesSize = meshlet.m_triangleSize = 0;
	}
	for (const auto& i : vertexIndices)
	{
		storage.m_vertexDataVertexPool.push(i);
	}
}

void GeometryStorage::FreeSkinnedMesh(const std::vector<uint32_t>& skinnedMeshletIndices)
{
	auto& storage = GetInstance();
	std::unordered_set<uint32_t> skinnedVertexIndices;
	for (const auto& i : skinnedMeshletIndices)
	{
		storage.m_skinnedMeshletPool.emplace(i);
		auto& skinnedMeshlet = storage.m_skinnedMeshlets[i];
		for (int j = 0; j < skinnedMeshlet.m_skinnedVerticesSize; j++)
		{
			skinnedVertexIndices.emplace(skinnedMeshlet.m_skinnedVertexIndices[j]);
		}
		skinnedMeshlet.m_skinnedVerticesSize = skinnedMeshlet.m_triangleSize = 0;
	}
	for (const auto& i : skinnedVertexIndices)
	{
		storage.m_skinnedVertexDataPool.push(i);
	}
}

void GeometryStorage::FreeStrands(const std::vector<uint32_t>& strandMeshletIndices)
{
	auto& storage = GetInstance();
	std::unordered_set<uint32_t> strandPointIndices;
	for (const auto& i : strandMeshletIndices)
	{
		storage.m_strandMeshletPool.emplace(i);
		auto& strandMeshlet = storage.m_strandMeshlets[i];
		for (int j = 0; j < strandMeshlet.m_strandPointsSize; j++)
		{
			strandPointIndices.emplace(strandMeshlet.m_strandPointIndices[j]);
		}
		strandMeshlet.m_strandPointsSize = strandMeshlet.m_segmentSize = 0;
	}
	for (const auto& i : strandPointIndices)
	{
		storage.m_strandPointDataPool.push(i);
	}
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
