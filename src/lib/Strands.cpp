//
// Created by lllll on 9/26/2022.
//

#include "Strands.hpp"
#include "Console.hpp"
#include "ClassRegistry.hpp"
#include "Jobs.hpp"
#include "RenderLayer.hpp"
using namespace EvoEngine;

void StrandPointAttributes::Serialize(YAML::Emitter& out) const
{
	out << YAML::Key << "m_thickness" << YAML::Value << m_thickness;
	out << YAML::Key << "m_normal" << YAML::Value << m_normal;
	out << YAML::Key << "m_texCoord" << YAML::Value << m_texCoord;
	out << YAML::Key << "m_color" << YAML::Value << m_color;
}

void StrandPointAttributes::Deserialize(const YAML::Node& in)
{
	if (in["m_thickness"]) m_normal = in["m_thickness"].as<bool>();
	if (in["m_normal"]) m_normal = in["m_normal"].as<bool>();
	if (in["m_texCoord"]) m_texCoord = in["m_texCoord"].as<bool>();
	if (in["m_color"]) m_color = in["m_color"].as<bool>();
}

unsigned int Strands::CurveDegree() const {
	switch (m_splineMode) {
	case SplineMode::Linear:
		return 1;
	case SplineMode::Quadratic:
		return 2;
	case SplineMode::Cubic:
		return 3;
	default: EVOENGINE_ERROR("Invalid spline mode.");
	}
}

Strands::SplineMode Strands::GetSplineMode() const { return m_splineMode; }


std::vector<StrandPoint>& Strands::UnsafeGetPoints() {
	return m_points;
}

std::vector<glm::uint>& Strands::UnsafeGetSegments() {
	return m_segments;
}

void Strands::PrepareStrands(const StrandPointAttributes& strandPointAttributes) {
	m_strandPointAttributes = strandPointAttributes;

	m_splineMode = SplineMode::Cubic;
	m_segmentIndices.resize(m_segments.size());
	Jobs::ParallelFor(m_segments.size(), [&](unsigned i)
		{
			m_segmentIndices[i].x = m_segments[i];
			m_segmentIndices[i].y = m_segments[i] + 1;
			m_segmentIndices[i].z = m_segments[i] + 2;
			m_segmentIndices[i].w = m_segments[i] + 3;
		}
	);

#pragma region Bound
	glm::vec3 minBound = m_points.at(0).m_position;
	glm::vec3 maxBound = m_points.at(0).m_position;
	for (auto& vertex : m_points)
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

	if (!m_strandPointAttributes.m_normal) RecalculateNormal();

	Upload();
	
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
	if (path.extension() == ".uestrands") {
		return IAsset::LoadInternal(path);
	}
	else if (path.extension() == ".hair") {
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
			strandPointAttributes.m_thickness = true;
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

static const char* SplineModes[]{ "Linear", "Quadratic", "Cubic" };

void Strands::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) {
	bool changed = false;
	ImGui::Text(("Point size: " + std::to_string(m_points.size())).c_str());
	if (changed) m_saved = false;
}

void Strands::Serialize(YAML::Emitter& out) {
	out << YAML::Key << "m_offset" << YAML::Value << m_offset;
	out << YAML::Key << "m_version" << YAML::Value << m_version;

	if (!m_segments.empty() && !m_points.empty()) {
		out << YAML::Key << "m_segments" << YAML::Value
			<< YAML::Binary((const unsigned char*)m_segments.data(), m_segments.size() * sizeof(glm::uint));

		out << YAML::Key << "m_scatteredPoints" << YAML::Value
			<< YAML::Binary((const unsigned char*)m_points.data(), m_points.size() * sizeof(StrandPoint));
	}
}

void Strands::Deserialize(const YAML::Node& in) {
	if (in["m_offset"]) m_offset = in["m_offset"].as<size_t>();
	if (in["m_version"]) m_version = in["m_version"].as<size_t>();

	if (in["m_segments"] && in["m_scatteredPoints"]) {
		auto segmentData = in["m_segments"].as<YAML::Binary>();
		m_segments.resize(segmentData.size() / sizeof(glm::uint));
		std::memcpy(m_segments.data(), segmentData.data(), segmentData.size());

		auto pointData = in["m_scatteredPoints"].as<YAML::Binary>();
		m_points.resize(pointData.size() / sizeof(StrandPoint));
		std::memcpy(m_points.data(), pointData.data(), pointData.size());

		StrandPointAttributes strandPointAttributes{};
		strandPointAttributes.m_thickness = true;
		strandPointAttributes.m_texCoord = true;
		strandPointAttributes.m_color = true;
		strandPointAttributes.m_normal = true;
		PrepareStrands(strandPointAttributes);
	}

}

void Strands::Upload()
{
	if (m_segmentIndices.empty())
	{
		EVOENGINE_ERROR("Indices empty!")
			return;
	}
	if (m_points.empty())
	{
		EVOENGINE_ERROR("Points empty!")
			return;
	}


	m_pointSize = m_points.size();
	m_segmentIndicesSize = m_segmentIndices.size();
	m_version++;
}

void Strands::Bind(VkCommandBuffer vkCommandBuffer) const
{

}

void Strands::OnCreate()
{
	m_bound = Bound();
}
Bound Strands::GetBound()
{
	return m_bound;
}

size_t Strands::GetSegmentAmount() const
{
	return m_segmentIndicesSize;
}

size_t Strands::GetPointAmount() const
{
	return m_pointSize;
}

void Strands::SetSegments(const StrandPointAttributes& strandPointAttributes, const std::vector<glm::uint>& segments, const std::vector<StrandPoint>& points) {
	if (points.empty() || segments.empty()) {
		return;
	}

	m_segments = segments;
	m_points = points;

	PrepareStrands(strandPointAttributes);
}

void Strands::SetStrands(const StrandPointAttributes& strandPointAttributes, const std::vector<glm::uint>& strands, const std::vector<StrandPoint>& points)
{
	if (points.empty() || strands.empty()) {
		return;
	}

	m_segments.clear();
	// loop to one before end, as last strand value is the "past last valid vertex"
	// index
	for (auto strand = strands.begin(); strand != strands.end() - 1; ++strand) {
		const int start = *(strand);                      // first vertex in first segment
		const int end = *(strand + 1) - CurveDegree();  // second vertex of last segment
		for (int i = start; i < end; i++) {
			m_segments.emplace_back(i);
		}
	}
	m_points = points;
	PrepareStrands(strandPointAttributes);
}


void CubicInterpolation(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, glm::vec3& result, glm::vec3& tangent, const float t)
{
	float t2 = t * t;

	glm::vec3 a0 = -0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3;
	glm::vec3 a1 = v0 - 2.5f * v1 + 2.0f * v2 - 0.5f * v3;
	glm::vec3 a2 = -0.5f * v0 + 0.5f * v2;
	glm::vec3 a3 = v1;

	result = glm::vec3(a0 * t * t2 + a1 * t2 + a2 * t + a3);
	tangent = normalize(glm::vec3(3.0f * a0 * t2 + 2.0f * a1 * t + a2));
}

void Strands::RecalculateNormal()
{
	glm::vec3 tangent, temp;
	for (const auto& indices : m_segmentIndices)
	{
		CubicInterpolation(m_points[indices[0]].m_position, m_points[indices[1]].m_position, m_points[indices[2]].m_position, m_points[indices[3]].m_position, temp, tangent, 0.0f);
		m_points[indices[0]].m_normal = glm::vec3(tangent.y, tangent.z, tangent.x);

		CubicInterpolation(m_points[indices[0]].m_position, m_points[indices[1]].m_position, m_points[indices[2]].m_position, m_points[indices[3]].m_position, temp, tangent, 0.25f);
		m_points[indices[1]].m_normal = glm::cross(glm::cross(tangent, m_points[indices[0]].m_normal), tangent);

		CubicInterpolation(m_points[indices[0]].m_position, m_points[indices[1]].m_position, m_points[indices[2]].m_position, m_points[indices[3]].m_position, temp, tangent, 0.75f);
		m_points[indices[2]].m_normal = glm::cross(glm::cross(tangent, m_points[indices[1]].m_normal), tangent);

		CubicInterpolation(m_points[indices[0]].m_position, m_points[indices[1]].m_position, m_points[indices[2]].m_position, m_points[indices[3]].m_position, temp, tangent, 1.0f);
		m_points[indices[3]].m_normal = glm::cross(glm::cross(tangent, m_points[indices[2]].m_normal), tangent);
	}
}

void Strands::DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState,
	int instanceCount, bool enableMetrics) const
{
}


