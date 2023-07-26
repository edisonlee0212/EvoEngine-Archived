#pragma once
#include "Camera.hpp"
#include "ILayer.hpp"
#include "Material.hpp"
#include "Mesh.hpp"
#include "IGeometry.hpp"
#include "SkinnedMeshRenderer.hpp"

namespace EvoEngine
{
	enum class RenderCommandType {
		None,
		FromRenderer,
		FromAPI,
		FromAPIInstanced
	};

	enum class RenderGeometryType {
		None,
		Mesh,
		SkinnedMesh,
		Strands
	};
	struct RenderCommand {
		RenderCommandType m_commandType = RenderCommandType::None;
		RenderGeometryType m_geometryType = RenderGeometryType::None;
		Entity m_owner = Entity();
		std::shared_ptr<IGeometry> m_renderGeometry;
		bool m_castShadow = true;
		bool m_receiveShadow = true;
		//std::shared_ptr<ParticleMatrices> m_matrices;
		std::shared_ptr<BoneMatrices> m_boneMatrices; // We require the skinned mesh renderer to provide bones.
		GlobalTransform m_globalTransform;
	};

	struct RenderCommandGroup {
		std::shared_ptr<Material> m_material;
		std::unordered_map<Handle, std::vector<RenderCommand>> m_renderCommands;
	};

	struct RenderTask {
		std::shared_ptr<Camera> m_camera;
		std::unordered_map<Handle, RenderCommandGroup> m_renderCommandsGroups;
		void Dispatch(const std::function<void(const std::shared_ptr<Material>&)>& beginCommandGroupAction,
			const std::function<void(const RenderCommand&)>& commandAction) const;
	};

	class RenderLayer : public ILayer {
		friend class Resources;
		friend class GraphicsPipeline;
		friend class EditorLayer;
		size_t m_triangles = 0;
		size_t m_strandsSegments = 0;
		size_t m_drawCall = 0;

		void OnCreate() override;
		void OnDestroy() override;
		void PreUpdate() override;
		void LateUpdate() override;
		void CreateRenderPasses();
		void CreateGraphicsPipelines();
		std::unordered_map<std::string, std::shared_ptr<RenderPass>> m_renderPasses;
		std::unordered_map<std::string, std::shared_ptr<GraphicsPipeline>> m_graphicsPipelines;

		std::unordered_map<Handle, RenderTask> m_deferredRenderInstances;
		std::unordered_map<Handle, RenderTask> m_deferredInstancedRenderInstances;
		std::unordered_map<Handle, RenderTask> m_forwardRenderInstances;
		std::unordered_map<Handle, RenderTask> m_forwardInstancedRenderInstances;
		std::unordered_map<Handle, RenderTask> m_transparentRenderInstances;
		std::unordered_map<Handle, RenderTask> m_instancedTransparentRenderInstances;


		void CollectRenderTasks(Bound& worldBound, std::vector<std::shared_ptr<Camera>>& cameras);
	public:
		[[nodiscard]] const std::shared_ptr<RenderPass>& GetRenderPass(const std::string& name);

		int m_mainCameraResolutionX = 1;
		int m_mainCameraResolutionY = 1;
		bool m_allowAutoResize = true;
		float m_mainCameraResolutionMultiplier = 1.0f;

		RenderInfoBlock m_renderInfoBlock = {};
		EnvironmentInfoBlock m_environmentInfoBlock = {};

		void RenderToCamera(const std::shared_ptr<Camera>& camera, const GlobalTransform& cameraModel);


	};
}
