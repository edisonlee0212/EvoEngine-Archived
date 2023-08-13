#include "Mesh.hpp"

#include "Console.hpp"
#include "Graphics.hpp"
#include "ClassRegistry.hpp"
#include "Jobs.hpp"
#include "GeometryStorage.hpp"
using namespace EvoEngine;

void Mesh::OnCreate()
{
	m_bound = Bound();
}

Mesh::~Mesh()
{
	GeometryStorage::FreeMesh(GetHandle());
	m_triangleRange.reset();
	m_meshletRange.reset();
}

void Mesh::DrawIndexed(const VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState, const int instancesCount) const
{
	if (instancesCount == 0) return;
	globalPipelineState.ApplyAllStates(vkCommandBuffer);
	vkCmdDrawIndexed(vkCommandBuffer, static_cast<uint32_t>(m_triangles.size() * 3), instancesCount, m_triangleRange->m_offset * 3, 0, 0);
}

void Mesh::SetVertices(const VertexAttributes& vertexAttributes, const std::vector<Vertex>& vertices,
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
	m_vertices = vertices;
	m_triangles = triangles;
	
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
	if (!vertexAttributes.m_normal)
		RecalculateNormal();
	if (!vertexAttributes.m_tangent)
		RecalculateTangent();

	m_vertexAttributes = vertexAttributes;
	m_vertexAttributes.m_normal = true;
	m_vertexAttributes.m_tangent = true;

	
	if(m_triangleRange || m_meshletRange) GeometryStorage::FreeMesh(GetHandle());
	m_triangleRange.reset();
	m_meshletRange.reset();
	GeometryStorage::AllocateMesh(GetHandle(), m_vertices, m_triangles, m_meshletRange, m_triangleRange);
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
}

void Mesh::RecalculateTangent()
{
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
}

std::vector<Vertex>& Mesh::UnsafeGetVertices()
{
	return m_vertices;
}

std::vector<glm::uvec3>& Mesh::UnsafeGetTriangles()
{
	return m_triangles;
}

Bound Mesh::GetBound() const
{
	return m_bound;
}


void Mesh::Serialize(YAML::Emitter& out)
{
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
	if(in["m_vertexAttributes"])
	{
		m_vertexAttributes.Deserialize(in["m_vertexAttributes"]);
	}else
	{
		m_vertexAttributes = {};
		m_vertexAttributes.m_normal = true;
		m_vertexAttributes.m_tangent = true;
		m_vertexAttributes.m_texCoord = true;
		m_vertexAttributes.m_color = true;
	}

	if (in["m_vertices"] && in["m_triangles"])
	{
		auto vertexData = in["m_vertices"].as<YAML::Binary>();
		std::vector<Vertex> vertices;
		vertices.resize(vertexData.size() / sizeof(Vertex));
		std::memcpy(vertices.data(), vertexData.data(), vertexData.size());

		auto triangleData = in["m_triangles"].as<YAML::Binary>();
		std::vector<glm::uvec3> triangles;
		triangles.resize(triangleData.size() / sizeof(glm::uvec3));
		std::memcpy(triangles.data(), triangleData.data(), triangleData.size());

		SetVertices(m_vertexAttributes, vertices, triangles);
		m_version++;
	}
}
void VertexAttributes::Serialize(YAML::Emitter& out) const
{
	out << YAML::Key << "m_normal" << YAML::Value << m_normal;
	out << YAML::Key << "m_tangent" << YAML::Value << m_tangent;
	out << YAML::Key << "m_texCoord" << YAML::Value << m_texCoord;
	out << YAML::Key << "m_color" << YAML::Value << m_color;
}

void VertexAttributes::Deserialize(const YAML::Node& in)
{
	if (in["m_normal"]) m_normal = in["m_normal"].as<bool>();
	if (in["m_tangent"]) m_tangent = in["m_tangent"].as<bool>();
	if (in["m_texCoord"]) m_texCoord = in["m_texCoord"].as<bool>();
	if (in["m_color"]) m_color = in["m_color"].as<bool>();
}

void ParticleInfoList::Serialize(YAML::Emitter& out)
{
	if (!m_particleInfos.empty())
	{
		out << YAML::Key << "m_particleInfos" << YAML::Value
			<< YAML::Binary(reinterpret_cast<const unsigned char*>(m_particleInfos.data()), m_particleInfos.size() * sizeof(ParticleInfo));
	}
}

void ParticleInfoList::Deserialize(const YAML::Node& in)
{
	if (in["m_particleInfos"])
	{
		const auto& vertexData = in["m_particleInfos"].as<YAML::Binary>();
		m_particleInfos.resize(vertexData.size() / sizeof(ParticleInfo));
		std::memcpy(m_particleInfos.data(), vertexData.data(), vertexData.size());
		SetPendingUpdate();
	}
}

void ParticleInfoList::UploadData(const bool force)
{
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	if(force || m_pendingUpdate[currentFrameIndex])
	{
		m_buffer[currentFrameIndex]->UploadVector(m_particleInfos);
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.offset = 0;
		bufferInfo.range = VK_WHOLE_SIZE;
		bufferInfo.buffer = m_buffer[currentFrameIndex]->GetVkBuffer();
		m_descriptorSet[currentFrameIndex]->UpdateBufferDescriptorBinding(18, bufferInfo, 0);
		m_pendingUpdate[currentFrameIndex] = false;
		m_version++;
	}
}

void ParticleInfoList::SetPendingUpdate()
{
	for (auto& i : m_pendingUpdate) i = true;
}

ParticleInfoList::ParticleInfoList()
{
	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.size = sizeof(ParticleInfo);
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	VmaAllocationCreateInfo bufferVmaAllocationCreateInfo{};
	bufferVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	const auto maxFramesInFlight = Graphics::GetMaxFramesInFlight();
	m_pendingUpdate.resize(maxFramesInFlight);
	for (int i = 0; i < maxFramesInFlight; i++) {
		m_pendingUpdate[i] = false;
		m_buffer.emplace_back(std::make_shared<Buffer>(bufferCreateInfo, bufferVmaAllocationCreateInfo));
		m_descriptorSet.emplace_back(std::make_shared<DescriptorSet>(Graphics::GetDescriptorSetLayout("INSTANCED_DATA_LAYOUT")));
	}
}

void ParticleInfoList::ApplyRays(const std::vector<Ray>& rays, const glm::vec4& color, float rayWidth)
{
	m_particleInfos.resize(rays.size());
	Jobs::ParallelFor(
		rays.size(),
		[&](unsigned i) {
			auto& ray = rays[i];
			glm::quat rotation = glm::quatLookAt(ray.m_direction,
				{ ray.m_direction.y, ray.m_direction.z, ray.m_direction.x });
			rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
			const glm::mat4 rotationMat = glm::mat4_cast(rotation);
			const auto model = glm::translate((ray.m_start + ray.m_direction * ray.m_length / 2.0f)) * rotationMat *
				glm::scale(glm::vec3(rayWidth, ray.m_length, rayWidth));
			m_particleInfos[i].m_instanceMatrix.m_value = model;
			m_particleInfos[i].m_instanceColor = color;
		});
	SetPendingUpdate();
}

void ParticleInfoList::ApplyRays(const std::vector<Ray>& rays, const std::vector<glm::vec4>& colors, float rayWidth)
{
	m_particleInfos.resize(rays.size());
	Jobs::ParallelFor(
		rays.size(),
		[&](unsigned i) {
			auto& ray = rays[i];
			glm::quat rotation = glm::quatLookAt(ray.m_direction,
				{ ray.m_direction.y, ray.m_direction.z, ray.m_direction.x });
			rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
			const glm::mat4 rotationMat = glm::mat4_cast(rotation);
			const auto model = glm::translate((ray.m_start + ray.m_direction * ray.m_length / 2.0f)) * rotationMat *
				glm::scale(glm::vec3(rayWidth, ray.m_length, rayWidth));
			m_particleInfos[i].m_instanceMatrix.m_value = model;
			m_particleInfos[i].m_instanceColor = colors[i];
		}
	);
	SetPendingUpdate();
}

void ParticleInfoList::ApplyConnections(const std::vector<glm::vec3>& starts, const std::vector<glm::vec3>& ends,
	const glm::vec4& color, float rayWidth)
{
	m_particleInfos.resize(starts.size());
	Jobs::ParallelFor(
		starts.size(),
		[&](unsigned i) {
			auto& start = starts[i];
			auto& end = ends[i];
			const auto direction = glm::normalize(end - start);
			glm::quat rotation = glm::quatLookAt(direction, glm::vec3(direction.y, direction.z, direction.x));
			rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
			const glm::mat4 rotationMat = glm::mat4_cast(rotation);
			const auto model = glm::translate((start + end) / 2.0f) * rotationMat *
				glm::scale(glm::vec3(rayWidth, glm::distance(end, start), rayWidth));
			m_particleInfos[i].m_instanceMatrix.m_value = model;
			m_particleInfos[i].m_instanceColor = color;
		}
	);
	SetPendingUpdate();
}

void ParticleInfoList::ApplyConnections(const std::vector<glm::vec3>& starts, const std::vector<glm::vec3>& ends,
                                        const std::vector<glm::vec4>& colors, float rayWidth)
{
	m_particleInfos.resize(starts.size());
	Jobs::ParallelFor(
		starts.size(),
		[&](unsigned i) {
			auto& start = starts[i];
			auto& end = ends[i];
			const auto direction = glm::normalize(end - start);
			glm::quat rotation = glm::quatLookAt(direction, glm::vec3(direction.y, direction.z, direction.x));
			rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
			const glm::mat4 rotationMat = glm::mat4_cast(rotation);
			const auto model = glm::translate((start + end) / 2.0f) * rotationMat *
				glm::scale(glm::vec3(rayWidth, glm::distance(end, start), rayWidth));
			m_particleInfos[i].m_instanceMatrix.m_value = model;
			m_particleInfos[i].m_instanceColor = colors[i];
		}
	);
	SetPendingUpdate();
}

const std::shared_ptr<DescriptorSet>& ParticleInfoList::GetDescriptorSet() const
{
	const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
	return m_descriptorSet[currentFrameIndex];
}
