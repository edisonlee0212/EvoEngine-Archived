#pragma once
#include "Animator.hpp"
#include "Scene.hpp"
#include "Vertex.hpp"
#include "GraphicsResources.hpp"
#include "GeometryStorage.hpp"
namespace EvoEngine
{
	struct SkinnedVertexAttributes
	{
		bool m_normal = false;
		bool m_tangent = false;
		bool m_texCoord = false;
		bool m_color = false;

		void Serialize(YAML::Emitter& out) const;
		void Deserialize(const YAML::Node& in);
	};

	class BoneMatrices
	{
		size_t m_version = 0;
		std::vector<std::unique_ptr<Buffer>> m_boneMatricesBuffer = {};
		std::vector<std::shared_ptr<DescriptorSet>> m_descriptorSet;
		friend class RenderLayer;
		void UploadData();
	public:
		[[nodiscard]] const std::shared_ptr<DescriptorSet>& GetDescriptorSet() const;
		BoneMatrices();
		[[nodiscard]] size_t& GetVersion();
		
		std::vector<glm::mat4> m_value;
	};
	class SkinnedMesh : public IAsset, public IGeometry
	{
		Bound m_bound;
		friend class SkinnedMeshRenderer;
		friend class Particles;
		friend class Graphics;
		friend class RenderLayer;
		SkinnedVertexAttributes m_skinnedVertexAttributes = {};
		std::vector<SkinnedVertex> m_skinnedVertices;
		std::vector<glm::uvec3> m_skinnedTriangles;
		std::shared_ptr<RangeDescriptor> m_skinnedTriangleRange;
		std::shared_ptr<RangeDescriptor> m_skinnedMeshletRange;


		friend struct SkinnedMeshBonesBlock;
		//Don't serialize.
		std::vector<std::shared_ptr<Bone>> m_bones;
		friend class Prefab;
	protected:
		bool SaveInternal(const std::filesystem::path& path) override;
	public:
		~SkinnedMesh() override;
		void DrawIndexed(VkCommandBuffer vkCommandBuffer, GraphicsPipelineStates& globalPipelineState, int instancesCount) const override;
		void OnCreate() override;
		void FetchIndices();
		//Need serialize
		std::vector<unsigned> m_boneAnimatorIndices;
		void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		[[nodiscard]] glm::vec3 GetCenter() const;
		[[nodiscard]] Bound GetBound() const;
		void SetVertices(const SkinnedVertexAttributes& skinnedVertexAttributes, const std::vector<SkinnedVertex>& skinnedVertices, const std::vector<unsigned>& indices);
		void SetVertices(const SkinnedVertexAttributes& skinnedVertexAttributes, const std::vector<SkinnedVertex>& skinnedVertices, const std::vector<glm::uvec3>& triangles);
		[[nodiscard]] size_t GetSkinnedVerticesAmount() const;
		[[nodiscard]] size_t GetTriangleAmount() const;
		void RecalculateNormal();
		void RecalculateTangent();
		[[nodiscard]] std::vector<SkinnedVertex>& UnsafeGetSkinnedVertices();
		[[nodiscard]] std::vector<glm::uvec3>& UnsafeGetTriangles();

		void Serialize(YAML::Emitter& out) override;
		void Deserialize(const YAML::Node& in) override;
	};
} // namespace EvoEngine