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
        [[nodiscard]] std::vector<glm::uint>& UnsafeGetSegments();
        [[nodiscard]] std::vector<StrandPoint>& UnsafeGetStrandPoints();
        void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;

        void Serialize(YAML::Emitter& out) override;

        void Deserialize(const YAML::Node& in) override;

        void SetSegments(const StrandPointAttributes& strandPointAttributes, const std::vector<glm::uint>& segments,
            const std::vector<StrandPoint>& points);

        void SetStrands(const StrandPointAttributes& strandPointAttributes, const std::vector<glm::uint>& strands,
            const std::vector<StrandPoint>& points);
        void RecalculateNormal();
        void DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState, int instanceCount, bool enableMetrics) const override;
        void UploadData();
        void Bind(VkCommandBuffer vkCommandBuffer) const override;
        void OnCreate() override;

        [[nodiscard]] Bound GetBound() const;

        [[nodiscard]] size_t GetSegmentAmount() const;
        ~Strands() override;
        [[nodiscard]] size_t GetStrandPointAmount() const;
        [[nodiscard]] const std::vector<uint32_t>& PeekStrandMeshletIndices() const;
    protected:
        bool LoadInternal(const std::filesystem::path& path) override;

    private:
        std::vector<glm::uvec4> m_geometryStorageSegments;
        std::unique_ptr<Buffer> m_segmentsBuffer = {};

        std::vector<uint32_t> m_strandMeshletIndices;

        StrandPointAttributes m_strandPointAttributes = {};

        friend class StrandsRenderer;
        friend class RenderLayer;
        Bound m_bound;

        void PrepareStrands(const StrandPointAttributes& strandPointAttributes);
        //The starting index of the point where this segment starts;
        std::vector<glm::uint> m_segmentRawIndices;

        std::vector<glm::uvec4> m_segments;
        std::vector<StrandPoint> m_strandPoints;
    };
}