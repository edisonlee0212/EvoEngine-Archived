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


	struct RenderInfoBlock {
		glm::vec4 m_splitDistances = {};
		alignas(4) int m_pcfSampleAmount = 64;
		alignas(4) int m_blockerSearchAmount = 1;
		alignas(4) float m_seamFixRatio = 0.05f;
		alignas(4) float m_gamma = 2.2f;

		alignas(4) float m_strandsSubdivisionXFactor = 50.0f;
		alignas(4) float m_strandsSubdivisionYFactor = 50.0f;
		alignas(4) int m_strandsSubdivisionMaxX = 15;
		alignas(4) int m_strandsSubdivisionMaxY = 8;
	};

	struct EnvironmentInfoBlock {
		glm::vec4 m_backgroundColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
		alignas(4) float m_environmentalMapGamma = 1.0f;
		alignas(4) float m_environmentalLightingIntensity = 1.0f;
		alignas(4) float m_backgroundIntensity = 1.0f;
		alignas(4) float m_environmentalPadding2 = 0.0f;
	};

	struct InstanceInfoBlock
	{
		GlobalTransform m_model;
	};

	class RenderLayer : public ILayer {
		friend class Resources;
		friend class GraphicsPipeline;
		friend class EditorLayer;
		friend class Material;
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


		std::shared_ptr<DescriptorSetLayout> m_perFrameLayout = {};
		std::vector<VkDescriptorSet> m_perFrameDescriptorSets = {};

		std::shared_ptr<DescriptorSetLayout> m_materialLayout = {};

		void CollectRenderTasks(Bound& worldBound, std::vector<std::shared_ptr<Camera>>& cameras);

		std::unordered_map<uint32_t, DescriptorSetLayoutBinding> m_descriptorSetLayoutBindings;
		bool m_layoutReady = false;
		bool m_descriptorSetsReady = false;

		std::vector<RenderInfoBlock*> m_renderInfoBlockMemory;
		std::vector<EnvironmentInfoBlock*> m_environmentalInfoBlockMemory;
		std::vector<CameraInfoBlock*> m_cameraInfoBlockMemory;
		std::vector<MaterialInfoBlock*> m_materialInfoBlockMemory;
		std::vector<InstanceInfoBlock*> m_instanceInfoBlockMemory;
		std::vector<std::unique_ptr<Buffer>> m_renderInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_environmentInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_cameraInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_materialInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_objectInfoDescriptorBuffers = {};

		void CreateStandardDescriptorBuffers();
		void UpdateStandardBindings();
		void ClearDescriptorSets();
		void CheckDescriptorSetsReady();

		std::unordered_map<Handle, uint32_t> m_cameraIndices;
		std::unordered_map<Handle, uint32_t> m_materialIndices;
		std::unordered_map<Handle, uint32_t> m_instanceIndices;

		std::vector<CameraInfoBlock> m_cameraInfoBlocks{};
		std::vector<MaterialInfoBlock> m_materialInfoBlocks{};
		std::vector<InstanceInfoBlock> m_instanceInfoBlocks{};

		void UploadCameraInfoBlocks(const std::vector<CameraInfoBlock>& cameraInfoBlocks);
		void UploadMaterialInfoBlocks(const std::vector<MaterialInfoBlock>& materialInfoBlocks);
		void UploadInstanceInfoBlocks(const std::vector<InstanceInfoBlock>& objectInfoBlocks);

		void PrepareMaterialLayout();
	public:
		uint32_t GetCameraIndex(const Handle& handle);
		uint32_t GetMaterialIndex(const Handle& handle);
		uint32_t GetInstanceIndex(const Handle& handle);
		void UploadCameraInfoBlock(const Handle& handle, const CameraInfoBlock& cameraInfoBlock);
		void UploadMaterialInfoBlock(const Handle& handle, const MaterialInfoBlock& materialInfoBlock);
		void UploadInstanceInfoBlock(const Handle& handle, const InstanceInfoBlock& instanceInfoBlock);

		uint32_t RegisterCameraIndex(const Handle& handle, const CameraInfoBlock& cameraInfoBlock, bool upload = false);
		uint32_t RegisterMaterialIndex(const Handle& handle, const MaterialInfoBlock& materialInfoBlock, bool upload = false);
		uint32_t RegisterInstanceIndex(const Handle& handle, const InstanceInfoBlock& instanceInfoBlock, bool upload = false);

		void UploadEnvironmentInfoBlock(const EnvironmentInfoBlock& environmentInfoBlock);
		void UploadRenderInfoBlock(const RenderInfoBlock& renderInfoBlock);

		void PushDescriptorBinding(uint32_t binding, VkDescriptorType type, VkShaderStageFlags stageFlags);
		/**
		 * \brief UpdateImageDescriptorBinding
		 * \param binding Target binding
		 * \param imageInfos The image info for update. Make sure the size is max frame size.
		 */
		void UpdateImageDescriptorBinding(uint32_t binding, const std::vector<VkDescriptorImageInfo>& imageInfos);
		void UpdateBufferDescriptorBinding(uint32_t binding, const std::vector<VkDescriptorBufferInfo>& bufferInfos);
		void PreparePerFrameLayout();

		int m_mainCameraResolutionX = 1;
		int m_mainCameraResolutionY = 1;
		bool m_allowAutoResize = true;
		float m_mainCameraResolutionMultiplier = 1.0f;

		RenderInfoBlock m_renderInfoBlock = {};
		EnvironmentInfoBlock m_environmentInfoBlock = {};

		void RenderToCamera(const std::shared_ptr<Camera>& camera);


	};
}
