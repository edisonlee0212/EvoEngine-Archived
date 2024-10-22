//
// Created by lllll on 9/26/2022.
//

#include "Strands.hpp"
#include "Console.hpp"
#include "ClassRegistry.hpp"
#include "GeometryStorage.hpp"
#include "Jobs.hpp"
#include "RenderLayer.hpp"
using namespace EvoEngine;

void StrandPointAttributes::Serialize(YAML::Emitter& out) const
{
	out << YAML::Key << "m_normal" << YAML::Value << m_normal;
	out << YAML::Key << "m_texCoord" << YAML::Value << m_texCoord;
	out << YAML::Key << "m_color" << YAML::Value << m_color;
}

void StrandPointAttributes::Deserialize(const YAML::Node& in)
{
	if (in["m_normal"]) m_normal = in["m_normal"].as<bool>();
	if (in["m_texCoord"]) m_texCoord = in["m_texCoord"].as<bool>();
	if (in["m_color"]) m_color = in["m_color"].as<bool>();
}

std::vector<StrandPoint>& Strands::UnsafeGetStrandPoints() {
	return m_strandPoints;
}

std::vector<glm::uint>& Strands::UnsafeGetSegments() {
	return m_segmentRawIndices;
}

void Strands::PrepareStrands(const StrandPointAttributes& strandPointAttributes) {
	m_segments.resize(m_segmentRawIndices.size());
	Jobs::RunParallelFor(m_segmentRawIndices.size(), [&](unsigned i)
		{
			m_segments[i].x = m_segmentRawIndices[i];
			m_segments[i].y = m_segmentRawIndices[i] + 1;
			m_segments[i].z = m_segmentRawIndices[i] + 2;
			m_segments[i].w = m_segmentRawIndices[i] + 3;
		}
	);

#pragma region Bound
	glm::vec3 minBound = m_strandPoints.at(0).m_position;
	glm::vec3 maxBound = m_strandPoints.at(0).m_position;
	for (auto& vertex : m_strandPoints)
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
	m_strandPointAttributes = strandPointAttributes;
	if (!m_strandPointAttributes.m_normal)
		RecalculateNormal();
	m_strandPointAttributes.m_normal = true;
	if(m_version != 0) GeometryStorage::FreeStrands(GetHandle());
	GeometryStorage::AllocateStrands(GetHandle(), m_strandPoints, m_segments, m_strandMeshletRange, m_segmentRange);
	m_version++;
	m_saved = false;
}

// .hair format spec here: http://www.cemyuksel.com/research/hairmodels/
struct HairHeader {
	// Bytes 0 - 3  Must be "HAIR" in ascii code(48 41 49 52)
	char m_magic[4];

	// Bytes 4 - 7  Number of hair strands as unsigned int
	uint32_t m_numStrands;

	// Bytes 8 - 11  Total number of points of all strands as unsigned int
	uint32_t m_numPoints;

	// Bytes 12 - 15  Bit array of data in the file
	// Bit - 5 to Bit - 31 are reserved for future extension(must be 0).
	uint32_t m_flags;

	// Bytes 16 - 19  Default number of segments of hair strands as unsigned int
	// If the file does not have a segments array, this default value is used.
	uint32_t m_defaultNumSegments;

	// Bytes 20 - 23  Default thickness hair strands as float
	// If the file does not have a thickness array, this default value is used.
	float m_defaultThickness;

	// Bytes 24 - 27  Default transparency hair strands as float
	// If the file does not have a transparency array, this default value is used.
	float m_defaultAlpha;

	// Bytes 28 - 39  Default color hair strands as float array of size 3
	// If the file does not have a color array, this default value is used.
	glm::vec3 m_defaultColor;

	// Bytes 40 - 127  File information as char array of size 88 in ascii
	char m_fileInfo[88];

	[[nodiscard]] bool HasSegments() const {
		return (m_flags & (0x1 << 0)) > 0;
	}

	[[nodiscard]] bool HasPoints() const {
		return (m_flags & (0x1 << 1)) > 0;
	}

	[[nodiscard]] bool HasThickness() const {
		return (m_flags & (0x1 << 2)) > 0;
	}

	[[nodiscard]] bool HasAlpha() const {
		return (m_flags & (0x1 << 3)) > 0;
	}

	[[nodiscard]] bool HasColor() const {
		return (m_flags & (0x1 << 4)) > 0;
	}
};

bool Strands::LoadInternal(const std::filesystem::path& path) {
	if (path.extension() == ".evestrands") {
		return IAsset::LoadInternal(path);
	}
	if (path.extension() == ".hair") {
		try {
			std::string fileName = path.string();
			std::ifstream input(fileName.c_str(), std::ios::binary);
			HairHeader header;
			input.read(reinterpret_cast<char*>(&header), sizeof(HairHeader));
			assert(input);
			assert(strncmp(header.m_magic, "HAIR", 4) == 0);
			header.m_fileInfo[87] = 0;

			// Segments array(unsigned short)
			// The segements array contains the number of linear segments per strand;
			// thus there are segments + 1 control-points/vertices per strand.
			auto strandSegments = std::vector<unsigned short>(header.m_numStrands);
			if (header.HasSegments()) {
				input.read(reinterpret_cast<char*>(strandSegments.data()),
					header.m_numStrands * sizeof(unsigned short));
				assert(input);
			}
			else {
				std::fill(strandSegments.begin(), strandSegments.end(), header.m_defaultNumSegments);
			}

			// Compute strands vector<unsigned int>. Each element is the index to the
			// first point of the first segment of the strand. The last entry is the
			// index "one beyond the last vertex".
			auto strands = std::vector<glm::uint>(strandSegments.size() + 1);
			auto strand = strands.begin();
			*strand++ = 0;
			for (auto segments : strandSegments) {
				*strand = *(strand - 1) + 1 + segments;
				strand++;
			}

			// Points array(float)
			assert(header.HasPoints());
			auto points = std::vector<glm::vec3>(header.m_numPoints);
			input.read(reinterpret_cast<char*>(points.data()), header.m_numPoints * sizeof(glm::vec3));
			assert(input);

			// Thickness array(float)
			auto thickness = std::vector<float>(header.m_numPoints);
			if (header.HasThickness()) {
				input.read(reinterpret_cast<char*>(thickness.data()), header.m_numPoints * sizeof(float));
				assert(input);
			}
			else {
				std::fill(thickness.begin(), thickness.end(), header.m_defaultThickness);
			}

			// Color array(float)
			auto color = std::vector<glm::vec3>(header.m_numPoints);
			if (header.HasColor()) {
				input.read(reinterpret_cast<char*>(color.data()), header.m_numPoints * sizeof(glm::vec3));
				assert(input);
			}
			else {
				std::fill(color.begin(), color.end(), header.m_defaultColor);
			}

			// Alpha array(float)
			auto alpha = std::vector<float>(header.m_numPoints);
			if (header.HasAlpha()) {
				input.read(reinterpret_cast<char*>(alpha.data()), header.m_numPoints * sizeof(float));
				assert(input);
			}
			else {
				std::fill(alpha.begin(), alpha.end(), header.m_defaultAlpha);
			}
			std::vector<StrandPoint> strandPoints;
			strandPoints.resize(header.m_numPoints);
			for (int i = 0; i < header.m_numPoints; i++) {
				strandPoints[i].m_position = points[i];
				strandPoints[i].m_thickness = thickness[i];
				strandPoints[i].m_color = glm::vec4(color[i], alpha[i]);
				strandPoints[i].m_texCoord = 0.0f;
			}
			StrandPointAttributes strandPointAttributes{};
			strandPointAttributes.m_texCoord = true;
			strandPointAttributes.m_color = true;
			SetStrands(strandPointAttributes, strands, strandPoints);
			return true;
		}
		catch (std::exception& e) {
			return false;
		}

	}
	return false;
}

bool Strands::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) {
	bool changed = false;
	ImGui::Text(("Point size: " + std::to_string(m_strandPoints.size())).c_str());
	return changed;
}

void Strands::Serialize(YAML::Emitter& out) const {

	if (!m_segmentRawIndices.empty() && !m_strandPoints.empty()) {
		out << YAML::Key << "m_segmentRawIndices" << YAML::Value
			<< YAML::Binary((const unsigned char*)m_segmentRawIndices.data(), m_segmentRawIndices.size() * sizeof(glm::uint));

		out << YAML::Key << "m_scatteredPoints" << YAML::Value
			<< YAML::Binary((const unsigned char*)m_strandPoints.data(), m_strandPoints.size() * sizeof(StrandPoint));
	}
}

void Strands::Deserialize(const YAML::Node& in) {
	if (in["m_segmentRawIndices"] && in["m_scatteredPoints"]) {
		const auto& segmentData = in["m_segmentRawIndices"].as<YAML::Binary>();
		m_segmentRawIndices.resize(segmentData.size() / sizeof(glm::uint));
		std::memcpy(m_segmentRawIndices.data(), segmentData.data(), segmentData.size());

		const auto& pointData = in["m_scatteredPoints"].as<YAML::Binary>();
		m_strandPoints.resize(pointData.size() / sizeof(StrandPoint));
		std::memcpy(m_strandPoints.data(), pointData.data(), pointData.size());

		StrandPointAttributes strandPointAttributes{};
		strandPointAttributes.m_texCoord = true;
		strandPointAttributes.m_color = true;
		strandPointAttributes.m_normal = true;
		PrepareStrands(strandPointAttributes);
	}

}

void Strands::OnCreate()
{
	m_version = 0;
	m_bound = Bound();
	m_segmentRange = std::make_shared<RangeDescriptor>();
	m_strandMeshletRange = std::make_shared<RangeDescriptor>();
}
Bound Strands::GetBound() const
{
	return m_bound;
}

size_t Strands::GetSegmentAmount() const
{
	return m_segments.size();
}

Strands::~Strands()
{
	GeometryStorage::FreeStrands(GetHandle());
	m_segmentRange.reset();
	m_strandMeshletRange.reset();
}

size_t Strands::GetStrandPointAmount() const
{
	return m_strandPoints.size();
}


void Strands::SetSegments(const StrandPointAttributes& strandPointAttributes, const std::vector<glm::uint>& segments, const std::vector<StrandPoint>& points) {
	if (points.empty() || segments.empty()) {
		return;
	}

	m_segmentRawIndices = segments;
	m_strandPoints = points;

	PrepareStrands(strandPointAttributes);
}

void Strands::SetStrands(const StrandPointAttributes& strandPointAttributes, const std::vector<glm::uint>& strands, const std::vector<StrandPoint>& points)
{
	if (points.empty() || strands.empty()) {
		return;
	}

	m_segmentRawIndices.clear();
	// loop to one before end, as last strand value is the "past last valid vertex"
	// index
	for (auto strand = strands.begin(); strand != strands.end() - 1; ++strand) {
		const int start = *(strand);                      // first vertex in first segment
		const int end = *(strand + 1) - 3;//CurveDegree();  // second vertex of last segment
		for (int i = start; i < end; i++) {
			m_segmentRawIndices.emplace_back(i);
		}
	}
	m_strandPoints = points;
	PrepareStrands(strandPointAttributes);
}

void Strands::RecalculateNormal()
{
	glm::vec3 tangent, temp;
	for (const auto& indices : m_segments)
	{
		CubicInterpolation(m_strandPoints[indices[0]].m_position, m_strandPoints[indices[1]].m_position, m_strandPoints[indices[2]].m_position, m_strandPoints[indices[3]].m_position, temp, tangent, 0.0f);
		m_strandPoints[indices[0]].m_normal = glm::vec3(tangent.y, tangent.z, tangent.x);

		CubicInterpolation(m_strandPoints[indices[0]].m_position, m_strandPoints[indices[1]].m_position, m_strandPoints[indices[2]].m_position, m_strandPoints[indices[3]].m_position, temp, tangent, 0.25f);
		m_strandPoints[indices[1]].m_normal = glm::cross(glm::cross(tangent, m_strandPoints[indices[0]].m_normal), tangent);

		CubicInterpolation(m_strandPoints[indices[0]].m_position, m_strandPoints[indices[1]].m_position, m_strandPoints[indices[2]].m_position, m_strandPoints[indices[3]].m_position, temp, tangent, 0.75f);
		m_strandPoints[indices[2]].m_normal = glm::cross(glm::cross(tangent, m_strandPoints[indices[1]].m_normal), tangent);

		CubicInterpolation(m_strandPoints[indices[0]].m_position, m_strandPoints[indices[1]].m_position, m_strandPoints[indices[2]].m_position, m_strandPoints[indices[3]].m_position, temp, tangent, 1.0f);
		m_strandPoints[indices[3]].m_normal = glm::cross(glm::cross(tangent, m_strandPoints[indices[2]].m_normal), tangent);
	}
}

void Strands::DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState,
	int instancesCount) const
{
	if (instancesCount == 0) return;
	globalPipelineState.ApplyAllStates(vkCommandBuffer);
	vkCmdDrawIndexed(vkCommandBuffer, m_segmentRange->m_prevFrameIndexCount * 4, instancesCount, m_segmentRange->m_prevFrameOffset * 4, 0, 0);
}


