#include "IGeometry.hpp"

using namespace evo_engine;
const std::vector<VkVertexInputBindingDescription>& IGeometry::GetVertexBindingDescriptions(GeometryType geometryType)
{
	static std::vector<VkVertexInputBindingDescription> mesh{};
	static std::vector<VkVertexInputBindingDescription> skinnedMesh{};
	static std::vector<VkVertexInputBindingDescription> strands{};
	if (mesh.empty()) {
		mesh.resize(1);
		mesh[0].binding = 0;
		mesh[0].stride = sizeof(Vertex);
		mesh[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	}
	if (skinnedMesh.empty()) {
		skinnedMesh.resize(1);
		skinnedMesh[0].binding = 0;
		skinnedMesh[0].stride = sizeof(SkinnedVertex);
		skinnedMesh[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	}
	if (strands.empty()) {
		strands.resize(1);
		strands[0].binding = 0;
		strands[0].stride = sizeof(StrandPoint);
		strands[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	}
	switch (geometryType)
	{
	case GeometryType::Mesh:
		return mesh;
	case GeometryType::SkinnedMesh:
		return skinnedMesh;
	case GeometryType::Strands:
		return strands;
	}
	return {};
}

const std::vector<VkVertexInputAttributeDescription>& IGeometry::GetVertexAttributeDescriptions(GeometryType geometryType)
{
	static std::vector<VkVertexInputAttributeDescription> mesh{};
	static std::vector<VkVertexInputAttributeDescription> skinnedMesh{};
	static std::vector<VkVertexInputAttributeDescription> strands{};
	if (mesh.empty())
	{
		mesh.resize(5);
		mesh[0].binding = 0;
		mesh[0].location = 0;
		mesh[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		mesh[0].offset = offsetof(Vertex, m_position);

		mesh[1].binding = 0;
		mesh[1].location = 1;
		mesh[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		mesh[1].offset = offsetof(Vertex, m_normal);

		mesh[2].binding = 0;
		mesh[2].location = 2;
		mesh[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		mesh[2].offset = offsetof(Vertex, m_tangent);

		mesh[3].binding = 0;
		mesh[3].location = 3;
		mesh[3].format = VK_FORMAT_R32G32_SFLOAT;
		mesh[3].offset = offsetof(Vertex, m_texCoord);

		mesh[4].binding = 0;
		mesh[4].location = 4;
		mesh[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		mesh[4].offset = offsetof(Vertex, m_color);
	}

	if (skinnedMesh.empty())
	{
		skinnedMesh.resize(9);
		skinnedMesh[0].binding = 0;
		skinnedMesh[0].location = 0;
		skinnedMesh[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		skinnedMesh[0].offset = offsetof(SkinnedVertex, m_position);

		skinnedMesh[1].binding = 0;
		skinnedMesh[1].location = 1;
		skinnedMesh[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		skinnedMesh[1].offset = offsetof(SkinnedVertex, m_normal);

		skinnedMesh[2].binding = 0;
		skinnedMesh[2].location = 2;
		skinnedMesh[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		skinnedMesh[2].offset = offsetof(SkinnedVertex, m_tangent);

		skinnedMesh[3].binding = 0;
		skinnedMesh[3].location = 3;
		skinnedMesh[3].format = VK_FORMAT_R32G32_SFLOAT;
		skinnedMesh[3].offset = offsetof(SkinnedVertex, m_texCoord);

		skinnedMesh[4].binding = 0;
		skinnedMesh[4].location = 4;
		skinnedMesh[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		skinnedMesh[4].offset = offsetof(SkinnedVertex, m_color);

		skinnedMesh[5].binding = 0;
		skinnedMesh[5].location = 5;
		skinnedMesh[5].format = VK_FORMAT_R32G32B32A32_SINT;
		skinnedMesh[5].offset = offsetof(SkinnedVertex, m_bondId);

		skinnedMesh[6].binding = 0;
		skinnedMesh[6].location = 6;
		skinnedMesh[6].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		skinnedMesh[6].offset = offsetof(SkinnedVertex, m_weight);

		skinnedMesh[7].binding = 0;
		skinnedMesh[7].location = 7;
		skinnedMesh[7].format = VK_FORMAT_R32G32B32A32_SINT;
		skinnedMesh[7].offset = offsetof(SkinnedVertex, m_bondId2);

		skinnedMesh[8].binding = 0;
		skinnedMesh[8].location = 8;
		skinnedMesh[8].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		skinnedMesh[8].offset = offsetof(SkinnedVertex, m_weight2);

	}

	if (strands.empty())
	{
		strands.resize(5);
		strands[0].binding = 0;
		strands[0].location = 0;
		strands[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		strands[0].offset = offsetof(StrandPoint, m_position);

		strands[1].binding = 0;
		strands[1].location = 1;
		strands[1].format = VK_FORMAT_R32_SFLOAT;
		strands[1].offset = offsetof(StrandPoint, m_thickness);

		strands[2].binding = 0;
		strands[2].location = 2;
		strands[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		strands[2].offset = offsetof(StrandPoint, m_normal);

		strands[3].binding = 0;
		strands[3].location = 3;
		strands[3].format = VK_FORMAT_R32_SFLOAT;
		strands[3].offset = offsetof(StrandPoint, m_texCoord);

		strands[4].binding = 0;
		strands[4].location = 4;
		strands[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		strands[4].offset = offsetof(StrandPoint, m_color);
	}
	switch (geometryType)
	{
	case GeometryType::Mesh:
		return mesh;
	case GeometryType::SkinnedMesh:
		return skinnedMesh;
	case GeometryType::Strands:
		return strands;
	}
	return {};
}