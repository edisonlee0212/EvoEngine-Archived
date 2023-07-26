#include "IGeometry.hpp"

using namespace EvoEngine;


const std::vector<VkVertexInputBindingDescription>& IGeometry::GetVertexBindingDescriptions(GeometryType geometryType)
{
	static std::vector<VkVertexInputBindingDescription> mesh{};
	if (mesh.empty()) {
		mesh.resize(1);
		mesh[0].binding = 0;
		mesh[0].stride = sizeof(Vertex);
		mesh[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	}
	switch (geometryType)
	{
	case GeometryType::Mesh:
		return mesh;
	}
	return {};
}

const std::vector<VkVertexInputAttributeDescription>& IGeometry::GetVertexAttributeDescriptions(GeometryType geometryType)
{
	static std::vector<VkVertexInputAttributeDescription> mesh{};
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
	switch (geometryType)
	{
	case GeometryType::Mesh:
		return mesh;
	}
	return {};
}