#include "Mesh.hpp"

#include "Console.hpp"
#include "Graphics.hpp"
#include "ClassRegistry.hpp"
#include "Jobs.hpp"
#include "GeometryStorage.hpp"
using namespace EvoEngine;

bool Mesh::SaveInternal(const std::filesystem::path& path)
{
	if (path.extension() == ".evemesh") {
		return IAsset::SaveInternal(path);
	}
	if (path.extension() == ".obj") {
		std::ofstream of;
		of.open(path.string(), std::ofstream::out | std::ofstream::trunc);
		if (of.is_open()) {
			std::string start = "#Mesh exporter, by Bosheng Li";
			start += "\n";
			of.write(start.c_str(), start.size());
			of.flush();
			unsigned startIndex = 1;
			if (!m_triangles.empty()) {
				std::string header =
					"#Vertices: " + std::to_string(m_vertices.size()) +
					", tris: " + std::to_string(m_triangles.size());
				header += "\n";
				of.write(header.c_str(), header.size());
				of.flush();
				std::stringstream data;
#pragma region Data collection
				for (auto i = 0; i < m_vertices.size(); i++) {
					auto& vertexPosition = m_vertices.at(i).m_position;
					auto& color = m_vertices.at(i).m_color;
					data << "v " + std::to_string(vertexPosition.x) + " " +
						std::to_string(vertexPosition.y) + " " +
						std::to_string(vertexPosition.z) + " " +
						std::to_string(color.x) + " " + std::to_string(color.y) + " " +
						std::to_string(color.z) + "\n";
				}
				for (const auto& vertex : m_vertices) {
					data << "vn " + std::to_string(vertex.m_normal.x) + " " +
						std::to_string(vertex.m_normal.y) + " " +
						std::to_string(vertex.m_normal.z) + "\n";
				}

				for (const auto& vertex : m_vertices) {
					data << "vt " + std::to_string(vertex.m_texCoord.x) + " " +
						std::to_string(vertex.m_texCoord.y) + "\n";
				}
				// data += "s off\n";
				data << "# List of indices for faces vertices, with (x, y, z).\n";
				auto& triangles = m_triangles;
				for (auto i = 0; i < m_triangles.size(); i++) {
					const auto triangle = triangles[i];
					const auto f1 = triangle.x + startIndex;
					const auto f2 = triangle.y + startIndex;
					const auto f3 = triangle.z + startIndex;
					data << "f " + std::to_string(f1) + "/" + std::to_string(f1) + "/" +
						std::to_string(f1) + " " + std::to_string(f2) + "/" +
						std::to_string(f2) + "/" + std::to_string(f2) + " " +
						std::to_string(f3) + "/" + std::to_string(f3) + "/" +
						std::to_string(f3) + "\n";
				}
				startIndex += m_vertices.size();
#pragma endregion
				const auto result = data.str();
				of.write(result.c_str(), result.size());
				of.flush();
			}
			of.close();
			return true;
		}
		EVOENGINE_ERROR("Can't open file!");
		return false;
	}
	return false;
}

void Mesh::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	ImGui::Text(("Vertices size: " + std::to_string(m_vertices.size())).c_str());
	ImGui::Text(("Triangle amount: " + std::to_string(m_triangles.size())).c_str());
	if (!m_vertices.empty()) {
		FileUtils::SaveFile(
			"Export as OBJ",
			"Mesh",
			{ ".obj" },
			[&](const std::filesystem::path& path)
			{
				Export(path);
			},
			false);
	}
	/*
	static bool visualize = true;
	static std::shared_ptr<Camera> visualizationCamera;
	static ImVec2 visualizationCameraResolution = { 200, 200 };
	if (visualize) {
		if (!visualizationCamera) {
			visualizationCamera = Serialization::ProduceSerializable<Camera>();
			visualizationCamera->m_clearColor = glm::vec3(0.0f);
			visualizationCamera->m_useClearColor = true;
			visualizationCamera->OnCreate();
		}
		else
		{
			// Show texture first;
			// Render for next frame;
			visualizationCamera->ResizeResolution(visualizationCameraResolution.x, visualizationCameraResolution.y);
			visualizationCamera->Clear();
			auto renderLayer = Application::GetLayer<RenderLayer>();
			static GlobalTransform visCameraGT;
			renderLayer->RenderToCamera(visualizationCamera, visCameraGT);
			ImGui::Image(
				reinterpret_cast<ImTextureID>(visualizationCamera->GetTexture()->UnsafeGetGLTexture()->Id()),
				visualizationCameraResolution,
				ImVec2(0, 1),
				ImVec2(1, 0));
		}
	}
	*/
}

void Mesh::OnCreate()
{
	m_version = 0;
	m_bound = Bound();
	m_triangleRange = std::make_shared<RangeDescriptor>();
	m_meshletRange = std::make_shared<RangeDescriptor>();
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
	vkCmdDrawIndexed(vkCommandBuffer, m_triangleRange->m_prevFrameIndexCount * 3, instancesCount, m_triangleRange->m_prevFrameOffset * 3, 0, 0);
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
#ifndef NDEBUG
		EVOENGINE_LOG("Vertices or triangles empty!");
#endif
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

	//MergeVertices();

	if (m_version != 0) GeometryStorage::FreeMesh(GetHandle());
	GeometryStorage::AllocateMesh(GetHandle(), m_vertices, m_triangles, m_meshletRange, m_triangleRange);
	
	m_version++;
	m_saved = false;
}

void Mesh::MergeVertices()
{
	for (uint32_t i = 0; i < m_vertices.size() - 1; i++)
	{
		for (uint32_t j = i + 1; j < m_vertices.size(); j++)
		{
			auto& vi = m_vertices.at(i);
			const auto& vj = m_vertices.at(j);
			if (glm::distance(vi.m_position, vj.m_position) > glm::epsilon<float>())
			{
				continue;
			}
			vi.m_texCoord = (vi.m_texCoord + vj.m_texCoord) * 0.5f;
			vi.m_color = (vi.m_color + vj.m_color) * 0.5f;
			m_vertices.at(j) = m_vertices.back();
			for (auto& triangle : m_triangles)
			{
				if (triangle.x == j) triangle.x = i;
				else if (triangle.x == m_vertices.size() - 1) triangle.x = j;
				if (triangle.y == j) triangle.y = i;
				else if (triangle.y == m_vertices.size() - 1) triangle.y = j;
				if (triangle.z == j) triangle.z = i;
				else if (triangle.z == m_vertices.size() - 1) triangle.z = j;
			}
			m_vertices.pop_back();
			j--;
		}
	}
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
	if (in["m_vertexAttributes"])
	{
		m_vertexAttributes.Deserialize(in["m_vertexAttributes"]);
	}
	else
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

void ParticleInfoList::OnCreate()
{
	m_rangeDescriptor = std::make_shared<RangeDescriptor>();
	GeometryStorage::AllocateParticleInfo(GetHandle(), m_rangeDescriptor);
}

ParticleInfoList::~ParticleInfoList()
{
	GeometryStorage::FreeParticleInfo(m_rangeDescriptor);
}


void ParticleInfoList::Serialize(YAML::Emitter& out)
{
	const auto particleInfos = GeometryStorage::PeekParticleInfoList(m_rangeDescriptor);
	if (!particleInfos.empty())
	{
		out << YAML::Key << "m_particleInfos" << YAML::Value
			<< YAML::Binary(reinterpret_cast<const unsigned char*>(particleInfos.data()), particleInfos.size() * sizeof(ParticleInfo));
	}
}

void ParticleInfoList::Deserialize(const YAML::Node& in)
{
	if (in["m_particleInfos"])
	{
		const auto& vertexData = in["m_particleInfos"].as<YAML::Binary>();
		std::vector<ParticleInfo> particleInfos;
		particleInfos.resize(vertexData.size() / sizeof(ParticleInfo));
		std::memcpy(particleInfos.data(), vertexData.data(), vertexData.size());
		GeometryStorage::UpdateParticleInfo(m_rangeDescriptor, particleInfos);
	}
}

void ParticleInfoList::ApplyRays(const std::vector<Ray>& rays, const glm::vec4& color, float rayWidth)
{
	std::vector<ParticleInfo> particleInfos;
	particleInfos.resize(rays.size());
	Jobs::RunParallelFor(
		rays.size(),
		[&](unsigned i) {
			auto& ray = rays[i];
			glm::quat rotation = glm::quatLookAt(ray.m_direction,
				{ ray.m_direction.y, ray.m_direction.z, ray.m_direction.x });
			rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
			const glm::mat4 rotationMat = glm::mat4_cast(rotation);
			const auto model = glm::translate((ray.m_start + ray.m_direction * ray.m_length / 2.0f)) * rotationMat *
				glm::scale(glm::vec3(rayWidth, ray.m_length, rayWidth));
			particleInfos[i].m_instanceMatrix.m_value = model;
			particleInfos[i].m_instanceColor = color;
		});
	GeometryStorage::UpdateParticleInfo(m_rangeDescriptor, particleInfos);
}

void ParticleInfoList::ApplyRays(const std::vector<Ray>& rays, const std::vector<glm::vec4>& colors, float rayWidth)
{
	std::vector<ParticleInfo> particleInfos;
	particleInfos.resize(rays.size());
	Jobs::RunParallelFor(
		rays.size(),
		[&](unsigned i) {
			auto& ray = rays[i];
			glm::quat rotation = glm::quatLookAt(ray.m_direction,
				{ ray.m_direction.y, ray.m_direction.z, ray.m_direction.x });
			rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
			const glm::mat4 rotationMat = glm::mat4_cast(rotation);
			const auto model = glm::translate((ray.m_start + ray.m_direction * ray.m_length / 2.0f)) * rotationMat *
				glm::scale(glm::vec3(rayWidth, ray.m_length, rayWidth));
			particleInfos[i].m_instanceMatrix.m_value = model;
			particleInfos[i].m_instanceColor = colors[i];
		}
	);
	GeometryStorage::UpdateParticleInfo(m_rangeDescriptor, particleInfos);
}

void ParticleInfoList::ApplyConnections(const std::vector<glm::vec3>& starts, const std::vector<glm::vec3>& ends,
	const glm::vec4& color, const float rayWidth) const
{
	std::vector<ParticleInfo> particleInfos;
	particleInfos.resize(starts.size());
	Jobs::RunParallelFor(
		starts.size(),
		[&](unsigned i) {
			const auto& start = starts[i];
			const auto& end = ends[i];
			const auto direction = glm::normalize(end - start);
			glm::quat rotation = glm::quatLookAt(direction, glm::vec3(direction.y, direction.z, direction.x));
			rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
			const glm::mat4 rotationMat = glm::mat4_cast(rotation);
			const auto model = glm::translate((start + end) / 2.0f) * rotationMat *
				glm::scale(glm::vec3(rayWidth, glm::distance(end, start), rayWidth));
			particleInfos[i].m_instanceMatrix.m_value = model;
			particleInfos[i].m_instanceColor = color;
		}
	);
	GeometryStorage::UpdateParticleInfo(m_rangeDescriptor, particleInfos);
}

void ParticleInfoList::ApplyConnections(const std::vector<glm::vec3>& starts, const std::vector<glm::vec3>& ends,
	const std::vector<glm::vec4>& colors, const float rayWidth) const
{
	std::vector<ParticleInfo> particleInfos;
	particleInfos.resize(starts.size());
	Jobs::RunParallelFor(
		starts.size(),
		[&](unsigned i) {
			const auto& start = starts[i];
			const auto& end = ends[i];
			const auto direction = glm::normalize(end - start);
			glm::quat rotation = glm::quatLookAt(direction, glm::vec3(direction.y, direction.z, direction.x));
			rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
			const glm::mat4 rotationMat = glm::mat4_cast(rotation);
			const auto model = glm::translate((start + end) / 2.0f) * rotationMat *
				glm::scale(glm::vec3(rayWidth, glm::distance(end, start), rayWidth));
			particleInfos[i].m_instanceMatrix.m_value = model;
			particleInfos[i].m_instanceColor = colors[i];
		}
	);
	GeometryStorage::UpdateParticleInfo(m_rangeDescriptor, particleInfos);
}

void ParticleInfoList::ApplyConnections(const std::vector<glm::vec3>& starts, const std::vector<glm::vec3>& ends,
	const std::vector<glm::vec4>& colors, const std::vector<float>& rayWidths) const
{
	std::vector<ParticleInfo> particleInfos;
	particleInfos.resize(starts.size());
	Jobs::RunParallelFor(
		starts.size(),
		[&](unsigned i) {
			const auto& start = starts[i];
			const auto& end = ends[i];
			const auto& width = rayWidths[i];
			const auto direction = glm::normalize(end - start);
			glm::quat rotation = glm::quatLookAt(direction, glm::vec3(direction.y, direction.z, direction.x));
			rotation *= glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f));
			const glm::mat4 rotationMat = glm::mat4_cast(rotation);
			const auto model = glm::translate((start + end) / 2.0f) * rotationMat *
				glm::scale(glm::vec3(width, glm::distance(end, start), width));
			particleInfos[i].m_instanceMatrix.m_value = model;
			particleInfos[i].m_instanceColor = colors[i];
		}
	);
	GeometryStorage::UpdateParticleInfo(m_rangeDescriptor, particleInfos);
}

void ParticleInfoList::SetParticleInfos(const std::vector<ParticleInfo>& particleInfos)
{
	GeometryStorage::UpdateParticleInfo(m_rangeDescriptor, particleInfos);
}

const std::vector<ParticleInfo>& ParticleInfoList::PeekParticleInfoList() const
{
	return GeometryStorage::PeekParticleInfoList(m_rangeDescriptor);
}

const std::shared_ptr<DescriptorSet>& ParticleInfoList::GetDescriptorSet() const
{
	return GeometryStorage::PeekDescriptorSet(m_rangeDescriptor);
}
