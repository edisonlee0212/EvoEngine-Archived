#include "MeshStorage.hpp"
using namespace EvoEngine;

void MeshStorage::UploadData()
{
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	if (m_requireDeviceUpdate[currentFrameIndex]) {
		m_vertexBuffer[currentFrameIndex]->UploadVector(m_vertexDataChunks);
		m_meshletBuffer[currentFrameIndex]->UploadVector(m_meshlets);
		m_requireDeviceUpdate[currentFrameIndex] = false;
	}
}

void MeshStorage::PreUpdate()
{
	auto& storage = GetInstance();
	storage.UploadData();
}

void MeshStorage::Initialize()
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
	storage.m_requireDeviceUpdate.resize(maxFrameInFlight);
}

const std::unique_ptr<Buffer>& MeshStorage::GetVertexBuffer()
{
	const auto& storage = GetInstance();
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	return storage.m_vertexBuffer[currentFrameIndex];
}

const std::unique_ptr<Buffer>& MeshStorage::GetMeshletBuffer()
{
	const auto& storage = GetInstance();
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	return storage.m_meshletBuffer[currentFrameIndex];
}

void MeshStorage::Bind(VkCommandBuffer commandBuffer)
{
	const auto& storage = GetInstance();
	constexpr VkDeviceSize offsets[] = { 0 };
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &storage.m_vertexBuffer[currentFrameIndex]->GetVkBuffer(), offsets);
}

const Vertex& MeshStorage::PeekVertex(const size_t vertexIndex)
{
	const auto& storage = GetInstance();
	return storage.m_vertexDataChunks[vertexIndex / Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE].m_vertexData[vertexIndex % Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE];
}

void MeshStorage::Allocate(const std::vector<Vertex>& vertices, const std::vector<glm::uvec3>& triangles,
                           std::vector<uint32_t>& meshletIndices)
{
	if(vertices.empty() || triangles.empty()) return;
	auto& storage = GetInstance();
	/* From mesh vertices to vertex data indices
	 */
	std::vector<uint32_t> vertexIndices;
	vertexIndices.resize(vertices.size());
	size_t currentChunkIndex = 0;
	size_t currentVertexIndexInChunk = Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE - 1;
	for(uint32_t i = 0; i < vertexIndices.size(); i++)
	{
		if(currentVertexIndexInChunk == Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE - 1)
		{
			currentVertexIndexInChunk = 0;
			if (!storage.m_vertexDataChunkPool.empty())
			{
				currentChunkIndex = storage.m_vertexDataChunkPool.front();
				storage.m_vertexDataChunkPool.pop();
			}
			else
			{
				currentChunkIndex = storage.m_vertexDataChunks.size();
				storage.m_vertexDataChunks.emplace_back();
			}
		}
		vertexIndices[i] = currentChunkIndex * Graphics::Constants::VERTEX_CHUNK_VERTICES_SIZE + currentVertexIndexInChunk;
		storage.m_vertexDataChunks[currentChunkIndex].m_vertexData[currentVertexIndexInChunk] = vertices[i];
		currentVertexIndexInChunk++;
	}

	meshletIndices.clear();
	uint32_t currentTriangleIndex = 0;
	while(currentTriangleIndex < triangles.size())
	{
		uint32_t currentMeshletIndex;
		if(!storage.m_meshletPool.empty())
		{
			currentMeshletIndex = storage.m_meshletPool.front();
			storage.m_meshletPool.pop();
		}else
		{
			currentMeshletIndex = storage.m_meshlets.size();
			storage.m_meshlets.emplace_back();
		}
		auto& currentMeshlet = storage.m_meshlets[currentMeshletIndex];
		currentMeshlet.m_verticesSize = currentMeshlet.m_triangleSize = 0;

		std::unordered_map<uint32_t, uint32_t> assignedVertices{};
		while(currentMeshlet.m_triangleSize < Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE && currentTriangleIndex < triangles.size())
		{
			const auto& currentTriangle = triangles[currentTriangleIndex];
			uint32_t newVerticesAmount = 0;
			auto searchX = assignedVertices.find(currentTriangle.x);
			if(searchX == assignedVertices.end()) newVerticesAmount++;

			auto searchY = assignedVertices.find(currentTriangle.y);
			if (searchY == assignedVertices.end()) newVerticesAmount++;

			auto searchZ = assignedVertices.find(currentTriangle.z);
			if (searchZ == assignedVertices.end()) newVerticesAmount++;

			if(currentMeshlet.m_verticesSize + newVerticesAmount > Graphics::Constants::MESHLET_MAX_VERTICES_SIZE)
			{
				break;
			}
			auto& currentMeshletTriangle = currentMeshlet.m_triangles[currentMeshlet.m_triangleSize];

			if (searchX != assignedVertices.end())
			{
				currentMeshletTriangle.x = searchX->second;
			}else
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
			if(currentMeshletIndex == 4)
			{
				int a = 4;
			}
			currentMeshlet.m_triangleSize++;
			currentTriangleIndex++;
		}
		meshletIndices.emplace_back(currentMeshletIndex);
	}
	for(auto& i : storage.m_requireDeviceUpdate) i = true;
}

void MeshStorage::Free(const std::vector<uint32_t>& meshletIndices)
{
	auto& storage = GetInstance();
	std::unordered_set<uint32_t> vertexIndices;
	for(const auto& i : meshletIndices)
	{
		storage.m_meshletPool.emplace(i);
		auto& meshlet = storage.m_meshlets[i];
		for(const auto& j : meshlet.m_vertexIndices)
		{
			vertexIndices.emplace(j);
		}
		meshlet.m_verticesSize = meshlet.m_triangleSize = 0;
	}
	for(const auto& i : vertexIndices)
	{
		storage.m_vertexDataChunkPool.push(i);
	}
}

const Meshlet& MeshStorage::PeekMeshlet(const uint32_t meshletIndex)
{
	const auto& storage = GetInstance();
	return storage.m_meshlets[meshletIndex];
}
