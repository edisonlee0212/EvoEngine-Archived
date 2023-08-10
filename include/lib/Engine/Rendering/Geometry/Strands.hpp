#pragma once
#include <Utilities.hpp>
#include "IAsset.hpp"
#include "Particles.hpp"
#include "Vertex.hpp"
namespace EvoEngine
{
    struct StrandPointAttributes
    {
        bool m_thickness = false;
        bool m_normal = false;
        bool m_texCoord = false;
        bool m_color = false;

        void Serialize(YAML::Emitter& out) const;
        void Deserialize(const YAML::Node& in);
    };
    class Strands : public IAsset, public IGeometry {
    public:
        enum class SplineMode {
            Linear = 2,
            Quadratic = 3,
            Cubic = 4
        };
        [[nodiscard]] SplineMode GetSplineMode() const;
        [[nodiscard]] std::vector<glm::uint>& UnsafeGetSegments();
        [[nodiscard]] std::vector<StrandPoint>& UnsafeGetPoints();
        void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;

        void Serialize(YAML::Emitter& out) override;

        void Deserialize(const YAML::Node& in) override;

        void SetSegments(const StrandPointAttributes& strandPointAttributes, const std::vector<glm::uint>& segments,
            const std::vector<StrandPoint>& points);

        void SetStrands(const StrandPointAttributes& strandPointAttributes, const std::vector<glm::uint>& strands,
            const std::vector<StrandPoint>& points);
        void RecalculateNormal();
        void DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState, int instanceCount, bool enableMetrics) const override;
        void Upload();
        void Bind(VkCommandBuffer vkCommandBuffer) const override;
        void OnCreate() override;

        [[nodiscard]] Bound GetBound();

        [[nodiscard]] size_t GetSegmentAmount() const;
        [[nodiscard]] size_t GetPointAmount() const;
    protected:
        bool LoadInternal(const std::filesystem::path& path) override;

    private:
        StrandPointAttributes m_strandPointAttributes = {};

        size_t m_offset = 0;
        unsigned m_segmentIndicesSize = 0;
        unsigned m_pointSize = 0;

        friend class StrandsRenderer;
        friend class RenderLayer;
        Bound m_bound;

        [[nodiscard]] unsigned int CurveDegree() const;

        void PrepareStrands(const StrandPointAttributes& strandPointAttributes);
        //The starting index of the point where this segment starts;
        std::vector<glm::uint> m_segments;

        std::vector<glm::uvec4> m_segmentIndices;
        std::vector<StrandPoint> m_points;

        SplineMode m_splineMode = SplineMode::Cubic;
    };
}