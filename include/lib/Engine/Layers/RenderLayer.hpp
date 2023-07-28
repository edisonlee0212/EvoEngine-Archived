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

	struct RenderInstancePushConstant
	{
		uint32_t m_instanceIndex = 0;
		uint32_t m_materialIndex = 0;
		uint32_t m_cameraIndex = 0;
	};

	struct RenderInstance {
		RenderInstancePushConstant m_pushConstant;
		RenderCommandType m_commandType = RenderCommandType::None;
		RenderGeometryType m_geometryType = RenderGeometryType::None;
		Entity m_owner = Entity();
		std::shared_ptr<IGeometry> m_renderGeometry;
		bool m_castShadow = true;
		bool m_receiveShadow = true;
		//std::shared_ptr<ParticleMatrices> m_matrices;
		std::shared_ptr<BoneMatrices> m_boneMatrices; // We require the skinned mesh renderer to provide bones.
	};

	struct RenderCommandGroup {
		std::shared_ptr<Material> m_material;
		std::unordered_map<Handle, std::vector<RenderInstance>> m_renderCommands;
	};

	struct RenderTask {
		std::shared_ptr<Camera> m_camera;
		std::unordered_map<Handle, RenderCommandGroup> m_renderCommandsGroups;
		void Dispatch(const std::function<void(const std::shared_ptr<Material>&)>& beginCommandGroupAction,
			const std::function<void(const RenderInstance&)>& commandAction) const;
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
		void CreateGraphicsPipelines();
		std::unordered_map<std::string, std::shared_ptr<GraphicsPipeline>> m_graphicsPipelines;

		std::unordered_map<Handle, RenderTask> m_deferredRenderInstances;
		std::unordered_map<Handle, RenderTask> m_deferredInstancedRenderInstances;
		std::unordered_map<Handle, RenderTask> m_forwardRenderInstances;
		std::unordered_map<Handle, RenderTask> m_forwardInstancedRenderInstances;
		std::unordered_map<Handle, RenderTask> m_transparentRenderInstances;
		std::unordered_map<Handle, RenderTask> m_instancedTransparentRenderInstances;


		std::unique_ptr<DescriptorSetLayout> m_perFrameLayout = {};
		std::vector<VkDescriptorSet> m_perFrameDescriptorSets = {};

		void CollectRenderTasks(Bound& worldBound, std::vector<std::shared_ptr<Camera>>& cameras);

		std::unordered_map<uint32_t, DescriptorSetLayoutBinding> m_descriptorSetLayoutBindings;
		bool m_layoutReady = false;
		bool m_descriptorSetsReady = false;
		void UpdateStandardBindings();
		void ClearDescriptorSets();
		void CheckDescriptorSetsReady();
	public:
		void PushDescriptorBinding(uint32_t binding, VkDescriptorType type, VkShaderStageFlags stageFlags);
		/**
		 * \brief 
		 * \param binding Target binding
		 * \param imageInfos The image info for update. Make sure the size is max frame size.
		 */
		void UpdateImageDescriptorBinding(uint32_t binding, const std::vector<VkDescriptorImageInfo>& imageInfos);
		void UpdateBufferDescriptorBinding(uint32_t binding, const std::vector<VkDescriptorBufferInfo>& bufferInfos);
		void PreparePerFrameLayouts();
		int m_mainCameraResolutionX = 1;
		int m_mainCameraResolutionY = 1;
		bool m_allowAutoResize = true;
		float m_mainCameraResolutionMultiplier = 1.0f;

		RenderInfoBlock m_renderInfoBlock = {};
		EnvironmentInfoBlock m_environmentInfoBlock = {};

		void RenderToCamera(const std::shared_ptr<Camera>& camera);


	};
}
