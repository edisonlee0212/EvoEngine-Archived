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
        [[nodiscard]] std::vector<glm::uint>& UnsafeGetSegments();
        [[nodiscard]] std::vector<StrandPoint>& UnsafeGetStrandPoints();
        bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;

        void Serialize(YAML::Emitter& out) const override;

        void Deserialize(const YAML::Node& in) override;

        void SetSegments(const StrandPointAttributes& strandPointAttributes, const std::vector<glm::uint>& segments,
            const std::vector<StrandPoint>& points);

        void SetStrands(const StrandPointAttributes& strandPointAttributes, const std::vector<glm::uint>& strands,
            const std::vector<StrandPoint>& points);
        void RecalculateNormal();
        void DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState, int instancesCount) const override;
        void OnCreate() override;

        [[nodiscard]] Bound GetBound() const;

        [[nodiscard]] size_t GetSegmentAmount() const;
        ~Strands() override;
        [[nodiscard]] size_t GetStrandPointAmount() const;

        template<class T>
        static void CubicInterpolation(const T& v0, const T& v1, const T& v2, const T& v3, T& result, T& tangent, float u);
        template<class T>
        static T CubicInterpolation(const T& v0, const T& v1, const T& v2, const T& v3, float u);
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
        //The starting index of the point where this segment starts;
        std::vector<glm::uint> m_segmentRawIndices;

        std::vector<glm::uvec4> m_segments;
        std::vector<StrandPoint> m_strandPoints;
    };

    template <class T>
    void Strands::CubicInterpolation(const T& v0, const T& v1, const T& v2, const T& v3, T& result, T& tangent, float u)
    {
        const T p0 = (v2 + v0) / 6.0f + v1 * (4.0f / 6.0f);
        const T p1 = v2 - v0;
        const T p2 = v2 - v1;
        const T p3 = v3 - v1;
        const float uu = u * u;
        const float u3 = 1.0f / 6.0f * uu * u;
        const glm::vec3 q = glm::vec3(u3 + 0.5 * (u - uu), uu - 4.0 * u3, u3);
        result = p0 + q.x * p1 + q.y * p2 + q.z * p3;
        if (u == 0.0)
            u = 0.000001f;
        if (u == 1.0)
            u = 0.999999f;
        const float v = 1.0f - u;
        tangent = 0.5f * v * v * p1 + 2.0f * v * u * p2 + 0.5f * u * u * p3;
    }

    template <class T>
    T Strands::CubicInterpolation(const T& v0, const T& v1, const T& v2, const T& v3, float u)
    {
        const T p0 = (v2 + v0) / 6.0f + v1 * (4.0f / 6.0f);
        const T p1 = v2 - v0;
        const T p2 = v2 - v1;
        const T p3 = v3 - v1;
        const float uu = u * u;
        const float u3 = 1.0f / 6.0f * uu * u;
        const glm::vec3 q = glm::vec3(u3 + 0.5 * (u - uu), uu - 4.0 * u3, u3);
        return p0 + q.x * p1 + q.y * p2 + q.z * p3;
    }
}
