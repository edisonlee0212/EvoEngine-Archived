#include "Mesh.hpp"

#include "Console.hpp"
#include "Graphics.hpp"
#include "ClassRegistry.hpp"
using namespace EvoEngine;
AssetRegistration<Mesh> MeshRegistry("Mesh", { ".uemesh" });

void Mesh::SubmitDrawIndexed(VkCommandBuffer vkCommandBuffer)
{
	VkBuffer vertexBuffers[] = { m_verticesBuffer->GetVkBuffer() };
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(vkCommandBuffer, 0, 1, vertexBuffers, offsets);
	vkCmdBindIndexBuffer(vkCommandBuffer, m_trianglesBuffer->GetVkBuffer(), 0, VK_INDEX_TYPE_UINT32);

	vkCmdDrawIndexed(vkCommandBuffer, static_cast<uint32_t>(m_triangles.size() * 3), 1, 0, 0, 0);

}

void Mesh::UploadData()
{
	const auto verticesDataSize = sizeof(Vertex) * m_vertices.size();

	VkBufferCreateInfo stagingBufferCreateInfo{};
	stagingBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingBufferCreateInfo.size = verticesDataSize;
	stagingBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	stagingBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo stagingBufferVmaAllocationCreateInfo{};
	stagingBufferVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
	stagingBufferVmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	const Buffer vertexStagingBuffer(stagingBufferCreateInfo, stagingBufferVmaAllocationCreateInfo);
	void* data = nullptr;
	vmaMapMemory(Graphics::GetVmaAllocator(), vertexStagingBuffer.GetVmaAllocation(), &data);
	memcpy(data, m_vertices.data(), verticesDataSize);
	vmaUnmapMemory(Graphics::GetVmaAllocator(), vertexStagingBuffer.GetVmaAllocation());

	VkBufferCreateInfo verticesBufferCreateInfo{};
	verticesBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	verticesBufferCreateInfo.size = verticesDataSize;
	verticesBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	verticesBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo verticesVmaAllocationCreateInfo{};
	verticesVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	m_verticesBuffer = std::make_unique<Buffer>(verticesBufferCreateInfo, verticesVmaAllocationCreateInfo);
	m_verticesBuffer->Copy(vertexStagingBuffer, verticesDataSize);


	const auto triangleDataSize = sizeof(glm::uvec3) * m_triangles.size();
	stagingBufferCreateInfo.size = triangleDataSize;
	const Buffer triangleStagingBuffer = { stagingBufferCreateInfo, stagingBufferVmaAllocationCreateInfo };
	vmaMapMemory(Graphics::GetVmaAllocator(), triangleStagingBuffer.GetVmaAllocation(), &data);
	memcpy(data, m_triangles.data(), triangleDataSize);
	vmaUnmapMemory(Graphics::GetVmaAllocator(), triangleStagingBuffer.GetVmaAllocation());


	VkBufferCreateInfo trianglesBufferCreateInfo{};
	trianglesBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	trianglesBufferCreateInfo.size = triangleDataSize;
	trianglesBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	trianglesBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo trianglesVmaAllocationCreateInfo{};
	trianglesVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	m_trianglesBuffer = std::make_unique<Buffer>(trianglesBufferCreateInfo, trianglesVmaAllocationCreateInfo);
	m_trianglesBuffer->Copy(triangleStagingBuffer, triangleDataSize);
}

void Mesh::SetVertices(const VertexAttributes& vertexAttributes, std::vector<Vertex>& vertices,
	const std::vector<unsigned>& indices)
{
	if (indices.size() % 3 != 0)
	{
		EVOENGINE_ERROR("Triangle size wrong!");
		return;
	}
	std::vector<glm::uvec3> triangles;
	triangles.resize(indices.size() / 3);
	memcpy(triangles.data(), indices.data(), indices.size() * sizeof(unsigned));
	SetVertices(vertexAttributes, vertices, triangles);
}

void Mesh::SetVertices(const VertexAttributes& vertexAttributes, const std::vector<Vertex>& vertices,
	const std::vector<glm::uvec3>& triangles)
{
	if (vertices.empty() || triangles.empty())
	{
		EVOENGINE_LOG("Vertices or triangles empty!");
		return;
	}
	if (!vertexAttributes.m_position)
	{
		EVOENGINE_ERROR("No Position Data!");
		return;
	}
	m_vertices = vertices;
	m_triangles = triangles;
	/*
#pragma region Bound
	glm::vec3 minBound = m_vertices.at(0).m_position;
	glm::vec3 maxBound = m_vertices.at(0).m_position;
	for (const auto& vertex : m_vertices)
	{
		minBound = glm::vec3(
			(glm::min)(minBound.x, vertex.m_position.x),
			(glm::min)(minBound.y, vertex.m_position.y),
			(glm::min)(minBound.z, vertex.m_position.z));
		maxBound = glm::vec3(
			(glm::max)(maxBound.x, vertex.m_position.x),
			(glm::max)(maxBound.y, vertex.m_position.y),
			(glm::max)(maxBound.z, vertex.m_position.z));
	}
	m_bound.m_max = maxBound;
	m_bound.m_min = minBound;
#pragma endregion
*/
	if (!vertexAttributes.m_normal)
		RecalculateNormal();
	if (!vertexAttributes.m_tangent)
		RecalculateTangent();

	m_vertexAttributes = vertexAttributes;
	m_vertexAttributes.m_normal = true;
	m_vertexAttributes.m_tangent = true;

	UploadData();
}

size_t Mesh::GetVerticesAmount() const
{
	return m_vertices.size();
}

size_t Mesh::GetTriangleAmount() const
{
	return m_triangles.size();
}

void Mesh::RecalculateNormal()
{
	/*
	auto normalLists = std::vector<std::vector<glm::vec3>>();
	const auto size = m_vertices.size();
	for (auto i = 0; i < size; i++)
	{
		normalLists.emplace_back();
	}
	for (const auto& triangle : m_triangles)
	{
		const auto i1 = triangle.x;
		const auto i2 = triangle.y;
		const auto i3 = triangle.z;
		auto v1 = m_vertices[i1].m_position;
		auto v2 = m_vertices[i2].m_position;
		auto v3 = m_vertices[i3].m_position;
		auto normal = glm::normalize(glm::cross(v1 - v2, v1 - v3));
		normalLists[i1].push_back(normal);
		normalLists[i2].push_back(normal);
		normalLists[i3].push_back(normal);
	}
	for (auto i = 0; i < size; i++)
	{
		auto normal = glm::vec3(0.0f);
		for (const auto j : normalLists[i])
		{
			normal += j;
		}
		m_vertices[i].m_normal = glm::normalize(normal);
	}
	*/
}

void Mesh::RecalculateTangent()
{
	/*
	auto tangentLists = std::vector<std::vector<glm::vec3>>();
	auto size = m_vertices.size();
	for (auto i = 0; i < size; i++)
	{
		tangentLists.emplace_back();
	}
	for (auto& triangle : m_triangles)
	{
		const auto i1 = triangle.x;
		const auto i2 = triangle.y;
		const auto i3 = triangle.z;
		auto p1 = m_vertices[i1].m_position;
		auto p2 = m_vertices[i2].m_position;
		auto p3 = m_vertices[i3].m_position;
		auto uv1 = m_vertices[i1].m_texCoord;
		auto uv2 = m_vertices[i2].m_texCoord;
		auto uv3 = m_vertices[i3].m_texCoord;

		auto e21 = p2 - p1;
		auto d21 = uv2 - uv1;
		auto e31 = p3 - p1;
		auto d31 = uv3 - uv1;
		float f = 1.0f / (d21.x * d31.y - d31.x * d21.y);
		auto tangent =
			f * glm::vec3(d31.y * e21.x - d21.y * e31.x, d31.y * e21.y - d21.y * e31.y, d31.y * e21.z - d21.y * e31.z);
		tangentLists[i1].push_back(tangent);
		tangentLists[i2].push_back(tangent);
		tangentLists[i3].push_back(tangent);
	}
	for (auto i = 0; i < size; i++)
	{
		auto tangent = glm::vec3(0.0f);
		for (auto j : tangentLists[i])
		{
			tangent += j;
		}
		m_vertices[i].m_tangent = glm::normalize(tangent);
	}
	*/
}

size_t& Mesh::GetVersion()
{
	return m_version;
}

std::vector<Vertex>& Mesh::UnsafeGetVertices()
{
	return m_vertices;
}

std::vector<glm::uvec3>& Mesh::UnsafeGetTriangles()
{
	return m_triangles;
}

void Mesh::Serialize(YAML::Emitter& out)
{
	out << YAML::Key << "m_version" << YAML::Value << m_version;

	out << YAML::Key << "m_vertexAttributes" << YAML::BeginMap;
	m_vertexAttributes.Serialize(out);
	out << YAML::EndMap;

	if (!m_vertices.empty() && !m_triangles.empty())
	{
		out << YAML::Key << "m_vertices" << YAML::Value
			<< YAML::Binary((const unsigned char*)m_vertices.data(), m_vertices.size() * sizeof(Vertex));
		out << YAML::Key << "m_triangles" << YAML::Value
			<< YAML::Binary((const unsigned char*)m_triangles.data(), m_triangles.size() * sizeof(glm::uvec3));
	}
}

void Mesh::Deserialize(const YAML::Node& in)
{
	if (in["m_vertices"] && in["m_triangles"] && in["m_vertexAttributes"])
	{
		auto vertexData = in["m_vertices"].as<YAML::Binary>();
		std::vector<Vertex> vertices;
		vertices.resize(vertexData.size() / sizeof(Vertex));
		std::memcpy(vertices.data(), vertexData.data(), vertexData.size());

		auto triangleData = in["m_triangles"].as<YAML::Binary>();
		std::vector<glm::uvec3> triangles;
		triangles.resize(triangleData.size() / sizeof(glm::uvec3));
		std::memcpy(triangles.data(), triangleData.data(), triangleData.size());

		m_vertexAttributes.Deserialize(in["m_vertexAttributes"]);

		SetVertices(m_vertexAttributes, vertices, triangles);

		m_version++;
	}
}
void VertexAttributes::Serialize(YAML::Emitter& out) const
{
	out << YAML::Key << "m_position" << YAML::Value << m_position;
	out << YAML::Key << "m_normal" << YAML::Value << m_normal;
	out << YAML::Key << "m_tangent" << YAML::Value << m_tangent;
	out << YAML::Key << "m_texCoord" << YAML::Value << m_texCoord;
	out << YAML::Key << "m_color" << YAML::Value << m_color;
}

void VertexAttributes::Deserialize(const YAML::Node& in)
{
	if (in["m_position"]) m_position = in["m_position"].as<bool>();
	if (in["m_normal"]) m_normal = in["m_normal"].as<bool>();
	if (in["m_tangent"]) m_tangent = in["m_tangent"].as<bool>();
	if (in["m_texCoord"]) m_texCoord = in["m_texCoord"].as<bool>();
	if (in["m_color"]) m_color = in["m_color"].as<bool>();
}