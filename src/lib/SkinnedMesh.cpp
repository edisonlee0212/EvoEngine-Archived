#include "Mesh.hpp"
#include "SkinnedMesh.hpp"
using namespace EvoEngine;

const std::shared_ptr<DescriptorSet>& BoneMatrices::GetDescriptorSet() const
{
	return m_descriptorSet;
}

BoneMatrices::BoneMatrices()
{
	VkBufferCreateInfo boneMatricesCrateInfo{};
	boneMatricesCrateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	boneMatricesCrateInfo.size = 256 * sizeof(glm::mat4);
	boneMatricesCrateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	boneMatricesCrateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo allocationCreateInfo{};
	allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	m_boneMatricesBuffer = std::make_unique<Buffer>(boneMatricesCrateInfo, allocationCreateInfo);

	m_descriptorSet = std::make_shared<DescriptorSet>(Graphics::GetDescriptorSetLayout("BONE_MATRICES_LAYOUT"));
}

size_t& BoneMatrices::GetVersion()
{
	return m_version;
}


void BoneMatrices::UploadData()
{
	m_version++;
	if (!m_value.empty())m_boneMatricesBuffer->UploadVector(m_value);
	VkDescriptorBufferInfo bufferInfo;
	bufferInfo.offset = 0;
	bufferInfo.buffer = m_boneMatricesBuffer->GetVkBuffer();
	bufferInfo.range = VK_WHOLE_SIZE;
	m_descriptorSet->UpdateBufferDescriptorBinding(5, bufferInfo);
}


void SkinnedMesh::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	ImGui::Text(("Vertices size: " + std::to_string(m_skinnedVertices.size())).c_str());
	ImGui::Text(("Triangle amount: " + std::to_string(m_triangles.size())).c_str());

	if (!m_skinnedVertices.empty()) {
		FileUtils::SaveFile(
			"Export as OBJ",
			"Mesh",
			{ ".obj" },
			[&](const std::filesystem::path& path) { Export(path); },
			false);
	}
}
bool SkinnedMesh::SaveInternal(const std::filesystem::path& path)
{
	if (path.extension() == ".evesmesh") {
		return IAsset::SaveInternal(path);
	}
	else if (path.extension() == ".obj") {
		std::ofstream of;
		of.open(path.string(), std::ofstream::out | std::ofstream::trunc);
		if (of.is_open()) {
			std::string start = "#Mesh exporter, by Bosheng Li";
			start += "\n";
			of.write(start.c_str(), start.size());
			of.flush();
			if (!m_triangles.empty()) {
				unsigned startIndex = 1;
				std::string header =
					"#Vertices: " + std::to_string(m_skinnedVertices.size()) +
					", tris: " + std::to_string(m_triangles.size());
				header += "\n";
				of.write(header.c_str(), header.size());
				of.flush();
				std::string data;
#pragma region Data collection
				for (const auto& skinnedVertex : m_skinnedVertices)
				{
					auto& vertexPosition = skinnedVertex.m_position;
					auto& color = skinnedVertex.m_color;
					data += "v " + std::to_string(vertexPosition.x) + " " +
						std::to_string(vertexPosition.y) + " " +
						std::to_string(vertexPosition.z) + " " +
						std::to_string(color.x) + " " + std::to_string(color.y) + " " +
						std::to_string(color.z) + "\n";
				}
				for (const auto& vertex : m_skinnedVertices) {
					data += "vn " + std::to_string(vertex.m_normal.x) + " " +
						std::to_string(vertex.m_normal.y) + " " +
						std::to_string(vertex.m_normal.z) + "\n";
				}

				for (const auto& vertex : m_skinnedVertices) {
					data += "vt " + std::to_string(vertex.m_texCoord.x) + " " +
						std::to_string(vertex.m_texCoord.y) + "\n";
				}
				// data += "s off\n";
				data += "# List of indices for faces vertices, with (x, y, z).\n";
				auto& triangles = m_triangles;
				for (auto i = 0; i < m_triangles.size(); i++) {
					const auto triangle = triangles[i];
					const auto f1 = triangle.x + startIndex;
					const auto f2 = triangle.y + startIndex;
					const auto f3 = triangle.z + startIndex;
					data += "f " + std::to_string(f1) + "/" + std::to_string(f1) + "/" +
						std::to_string(f1) + " " + std::to_string(f2) + "/" +
						std::to_string(f2) + "/" + std::to_string(f2) + " " +
						std::to_string(f3) + "/" + std::to_string(f3) + "/" +
						std::to_string(f3) + "\n";
				}
				startIndex += m_skinnedVertices.size();
#pragma endregion
				of.write(data.c_str(), data.size());
				of.flush();
			}
			of.close();
			return true;
		}
		else {
			EVOENGINE_ERROR("Can't open file!");
			return false;
		}
	}
	return false;
}

void SkinnedMesh::Bind(VkCommandBuffer vkCommandBuffer) const
{
	constexpr VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(vkCommandBuffer, 0, 1, &m_verticesBuffer->GetVkBuffer(), offsets);
	vkCmdBindIndexBuffer(vkCommandBuffer, m_trianglesBuffer->GetVkBuffer(), 0, VK_INDEX_TYPE_UINT32);
}

void SkinnedMesh::DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState, int instanceCount,
                              bool enableMetrics) const
{
	auto& graphics = Graphics::GetInstance();
	if (enableMetrics) {
		graphics.m_drawCall++;
		graphics.m_triangles += m_triangles.size();
	}
	globalPipelineState.ApplyAllStates(vkCommandBuffer);
	vkCmdDrawIndexed(vkCommandBuffer, static_cast<uint32_t>(m_triangles.size() * 3), instanceCount, 0, 0, 0);
}

glm::vec3 SkinnedMesh::GetCenter() const
{
	return m_bound.Center();
}
Bound SkinnedMesh::GetBound() const
{
	return m_bound;
}

void SkinnedMesh::FetchIndices()
{
	m_boneAnimatorIndices.resize(m_bones.size());
	for (int i = 0; i < m_bones.size(); i++)
	{
		m_boneAnimatorIndices[i] = m_bones[i]->m_index;
	}
}

void SkinnedMesh::OnCreate()
{


	VkBufferCreateInfo verticesBufferCreateInfo{};
	verticesBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	verticesBufferCreateInfo.size = 1;
	verticesBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	verticesBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo verticesVmaAllocationCreateInfo{};
	verticesVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	m_verticesBuffer = std::make_unique<Buffer>(verticesBufferCreateInfo, verticesVmaAllocationCreateInfo);


	VkBufferCreateInfo trianglesBufferCreateInfo{};
	trianglesBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	trianglesBufferCreateInfo.size = 1;
	trianglesBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	trianglesBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo trianglesVmaAllocationCreateInfo{};
	trianglesVmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	m_trianglesBuffer = std::make_unique<Buffer>(trianglesBufferCreateInfo, trianglesVmaAllocationCreateInfo);

	m_bound = Bound();
}
void SkinnedMesh::UploadData()
{
	if (m_skinnedVertices.empty())
	{
		EVOENGINE_ERROR("Vertices empty!")
			return;
	}
	if (m_triangles.empty())
	{
		EVOENGINE_ERROR("Triangles empty!")
			return;
	}
	m_version++;

	m_verticesBuffer->UploadVector(m_skinnedVertices);
	m_trianglesBuffer->UploadVector(m_triangles);
}

void SkinnedMesh::SetVertices(
	const unsigned& mask, const std::vector<SkinnedVertex>& skinnedVertices, const std::vector<unsigned>& indices)
{
	if (indices.size() % 3 != 0)
	{
		EVOENGINE_ERROR("Triangle size wrong!");
		return;
	}
	std::vector<glm::uvec3> triangles;
	triangles.resize(indices.size() / 3);
	memcpy(triangles.data(), indices.data(), indices.size() * sizeof(unsigned));
	SetVertices(mask, skinnedVertices, triangles);
}

void SkinnedMesh::SetVertices(
	const unsigned& mask, const std::vector<SkinnedVertex>& skinnedVertices, const std::vector<glm::uvec3>& triangles)
{
	if (skinnedVertices.empty() || triangles.empty())
	{
		EVOENGINE_LOG("Vertices or triangles empty!");
		return;
	}
	if (!(mask & (unsigned)VertexAttribute::Position))
	{
		EVOENGINE_ERROR("No Position Data!");
		return;
	}
	m_mask = mask;
	m_skinnedVertices = skinnedVertices;
	m_triangles = triangles;
#pragma region Bound
	glm::vec3 minBound = m_skinnedVertices.at(0).m_position;
	glm::vec3 maxBound = m_skinnedVertices.at(0).m_position;
	for (size_t i = 0; i < m_skinnedVertices.size(); i++)
	{
		minBound = glm::vec3(
			(glm::min)(minBound.x, m_skinnedVertices[i].m_position.x),
			(glm::min)(minBound.y, m_skinnedVertices[i].m_position.y),
			(glm::min)(minBound.z, m_skinnedVertices[i].m_position.z));
		maxBound = glm::vec3(
			(glm::max)(maxBound.x, m_skinnedVertices[i].m_position.x),
			(glm::max)(maxBound.y, m_skinnedVertices[i].m_position.y),
			(glm::max)(maxBound.z, m_skinnedVertices[i].m_position.z));
	}
	m_bound.m_max = maxBound;
	m_bound.m_min = minBound;
#pragma endregion
	if (!(m_mask & (unsigned)VertexAttribute::Normal))
		RecalculateNormal();
	if (!(m_mask & (unsigned)VertexAttribute::Tangent))
		RecalculateTangent();
	UploadData();
}

size_t SkinnedMesh::GetSkinnedVerticesAmount() const
{
	return m_skinnedVertices.size();
}

size_t SkinnedMesh::GetTriangleAmount() const
{
	return m_triangles.size();
}

void SkinnedMesh::RecalculateNormal()
{
	auto normalLists = std::vector<std::vector<glm::vec3>>();
	const auto size = m_skinnedVertices.size();
	for (auto i = 0; i < size; i++)
	{
		normalLists.emplace_back();
	}
	for (const auto& triangle : m_triangles)
	{
		const auto i1 = triangle.x;
		const auto i2 = triangle.y;
		const auto i3 = triangle.z;
		auto v1 = m_skinnedVertices[i1].m_position;
		auto v2 = m_skinnedVertices[i2].m_position;
		auto v3 = m_skinnedVertices[i3].m_position;
		auto normal = glm::normalize(glm::cross(v1 - v2, v1 - v3));
		normalLists[i1].push_back(normal);
		normalLists[i2].push_back(normal);
		normalLists[i3].push_back(normal);
	}
	for (auto i = 0; i < size; i++)
	{
		auto normal = glm::vec3(0.0f);
		for (auto j : normalLists[i])
		{
			normal += j;
		}
		m_skinnedVertices[i].m_normal = glm::normalize(normal);
	}
}

void SkinnedMesh::RecalculateTangent()
{
	auto tangentLists = std::vector<std::vector<glm::vec3>>();
	auto size = m_skinnedVertices.size();
	for (auto i = 0; i < size; i++)
	{
		tangentLists.emplace_back();
	}
	for (auto& triangle : m_triangles)
	{
		const auto i1 = triangle.x;
		const auto i2 = triangle.y;
		const auto i3 = triangle.z;
		auto p1 = m_skinnedVertices[i1].m_position;
		auto p2 = m_skinnedVertices[i2].m_position;
		auto p3 = m_skinnedVertices[i3].m_position;
		auto uv1 = m_skinnedVertices[i1].m_texCoord;
		auto uv2 = m_skinnedVertices[i2].m_texCoord;
		auto uv3 = m_skinnedVertices[i3].m_texCoord;

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
		m_skinnedVertices[i].m_tangent = glm::normalize(tangent);
	}
}

std::vector<glm::uvec3>& SkinnedMesh::UnsafeGetTriangles()
{
	return m_triangles;
}
std::vector<SkinnedVertex>& SkinnedMesh::UnsafeGetSkinnedVertices()
{
	return m_skinnedVertices;
}

void SkinnedMesh::Serialize(YAML::Emitter& out)
{
	if (!m_boneAnimatorIndices.empty())
	{
		out << YAML::Key << "m_boneAnimatorIndices" << YAML::Value
			<< YAML::Binary(
				reinterpret_cast<const unsigned char*>(m_boneAnimatorIndices.data()),
				m_boneAnimatorIndices.size() * sizeof(unsigned));
	}
	out << YAML::Key << "m_mask" << YAML::Value << m_mask;
	out << YAML::Key << "m_offset" << YAML::Value << m_offset;
	out << YAML::Key << "m_version" << YAML::Value << m_version;

	if (!m_skinnedVertices.empty() && !m_triangles.empty())
	{
		out << YAML::Key << "m_skinnedVertices" << YAML::Value
			<< YAML::Binary(
				reinterpret_cast<const unsigned char*>(m_skinnedVertices.data()), m_skinnedVertices.size() * sizeof(SkinnedVertex));
		out << YAML::Key << "m_triangles" << YAML::Value
			<< YAML::Binary(reinterpret_cast<const unsigned char*>(m_triangles.data()), m_triangles.size() * sizeof(glm::uvec3));
	}
}
void SkinnedMesh::Deserialize(const YAML::Node& in)
{
	if (in["m_boneAnimatorIndices"])
	{
		const YAML::Binary& boneIndices = in["m_boneAnimatorIndices"].as<YAML::Binary>();
		m_boneAnimatorIndices.resize(boneIndices.size() / sizeof(unsigned));
		std::memcpy(m_boneAnimatorIndices.data(), boneIndices.data(), boneIndices.size());
	}
	m_mask = in["m_mask"].as<unsigned>();
	m_offset = in["m_offset"].as<size_t>();
	m_version = in["m_version"].as<size_t>();

	if (in["m_skinnedVertices"] && in["m_triangles"])
	{
		const YAML::Binary& skinnedVertexData = in["m_skinnedVertices"].as<YAML::Binary>();
		std::vector<SkinnedVertex> skinnedVertices;
		skinnedVertices.resize(skinnedVertexData.size() / sizeof(SkinnedVertex));
		std::memcpy(skinnedVertices.data(), skinnedVertexData.data(), skinnedVertexData.size());

		const YAML::Binary& triangleData = in["m_triangles"].as<YAML::Binary>();
		std::vector<glm::uvec3> triangles;
		triangles.resize(triangleData.size() / sizeof(glm::uvec3));
		std::memcpy(triangles.data(), triangleData.data(), triangleData.size());

		SetVertices(m_mask, skinnedVertices, triangles);
	}
}