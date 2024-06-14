#pragma once
#include "Application.hpp"
#include "Camera.hpp"
#include "Mesh.hpp"
namespace evo_engine
{

class PointCloud : public IAsset
{
    glm::dvec3 m_min = glm::dvec3(FLT_MAX);
    glm::dvec3 m_max = glm::dvec3(-FLT_MAX);
protected:
    bool LoadInternal(const std::filesystem::path &path) override;
    bool SaveInternal(const std::filesystem::path &path) const override;
  public:
    struct PointCloudSaveSettings
    {
	    bool m_binary = true;
        bool m_doublePrecision = false;
    };
    struct PointCloudLoadSettings
    {
	    bool m_binary = true;
    };
    glm::dvec3 m_offset;
    bool m_hasPositions = false;
    bool m_hasNormals = false;
    bool m_hasColors = false;
    std::vector<glm::dvec3> m_points;
    std::vector<glm::dvec3> m_normals;
    std::vector<glm::vec4> m_colors;
    float m_pointSize = 0.01f;
    float m_compressFactor = 0.01f;
    void OnCreate() override;
    


    bool Load(const PointCloudLoadSettings& settings, const std::filesystem::path &path);
    bool Save(const PointCloudSaveSettings& settings, const std::filesystem::path& path) const;
    bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
    void Compress(std::vector<glm::dvec3>& points);
    void ApplyCompressed();
    void ApplyOriginal();
    void RecalculateBoundingBox();
    void Crop(std::vector<glm::dvec3>& points, const glm::dvec3& min, const glm::dvec3& max);
    void Serialize(YAML::Emitter &out) const override;
    void Deserialize(const YAML::Node &in) override;
    
};
} // namespace evo_engine
