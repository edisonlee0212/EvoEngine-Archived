#pragma once
#include <Utilities.hpp>
#include "IAsset.hpp"
#include "Particles.hpp"
#include "Vertex.hpp"
#include "GeometryStorage.hpp"
namespace EvoEngine
{
    struct StrandPointAttributes
    {
        bool m_normal = false;
        bool m_texCoord = false;
        bool m_color = false;

        void Serialize(YAML::Emitter& out) const;
        void Deserialize(const YAML::Node& in);
    };
    class Strands : public IAsset, public IGeometry {
    public:
        [[nodiscard]] std::vector<glm::uvec4>& UnsafeGetSegments();
        [[nodiscard]] std::vector<StrandPoint>& UnsafeGetStrandPoints();
        void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;

        void Serialize(YAML::Emitter& out) override;

        void Deserialize(const YAML::Node& in) override;

        void SetStrands(const StrandPointAttributes& strandPointAttributes, const std::vector<glm::uvec4>& segments,
            const std::vector<StrandPoint>& points);
        void RecalculateNormal();
        void DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState, int instancesCount) const override;
        void OnCreate() override;

        [[nodiscard]] Bound GetBound() const;

        [[nodiscard]] size_t GetSegmentAmount() const;
        ~Strands() override;
        [[nodiscard]] size_t GetStrandPointAmount() const;

        static void CubicInterpolation(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, glm::vec3& result, glm::vec3& tangent, float u);
    protected:
        bool LoadInternal(const std::filesystem::path& path) override;

    private:
        std::shared_ptr<RangeDescriptor> m_segmentRange;
        std::shared_ptr<RangeDescriptor> m_strandMeshletRange;

        StrandPointAttributes m_strandPointAttributes = {};

        friend class StrandsRenderer;
        friend class RenderLayer;
        Bound m_bound;

        void PrepareStrands(const StrandPointAttributes& strandPointAttributes);
        
        std::vector<glm::uvec4> m_segments;
        std::vector<StrandPoint> m_strandPoints;
    };
}