#pragma once
#include "Camera.hpp"
#include "ILayer.hpp"
#include "Material.hpp"
#include "Mesh.hpp"
#include "IGeometry.hpp"
#include "Lights.hpp"
#include "SkinnedMeshRenderer.hpp"
namespace EvoEngine
{
#pragma region Enums Structs
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
		int m_instanceIndex = 0;
		int m_materialIndex = 0;
		int m_cameraIndex = 0;
		int m_infoIndex = 0;
	};

	struct RenderInstance {
		uint32_t m_instanceIndex = 0;
		uint32_t m_materialIndex = 0;
		uint32_t m_selected = 0;
		RenderCommandType m_commandType = RenderCommandType::None;
		RenderGeometryType m_geometryType = RenderGeometryType::None;
		Entity m_owner = Entity();
		std::shared_ptr<IGeometry> m_renderGeometry;
		bool m_castShadow = true;
		bool m_receiveShadow = true;
		//std::shared_ptr<ParticleMatrices> m_matrices;
		std::shared_ptr<BoneMatrices> m_boneMatrices; // We require the skinned mesh renderer to provide bones.
	};

	struct RenderInstanceGroup {
		std::shared_ptr<Material> m_material;
		std::unordered_map<Handle, std::vector<RenderInstance>> m_renderCommands;
	};

	struct RenderInstanceCollection
	{
		std::unordered_map<Handle, RenderInstanceGroup> m_renderInstanceGroups;

		void Add(const std::shared_ptr<Material>& material, const RenderInstance& renderInstance);

		void Dispatch(const std::function<void(const std::shared_ptr<Material>&)>& beginCommandGroupAction,
			const std::function<void(const RenderInstance&)>& commandAction) const;
	};

	struct RenderInfoBlock {
		glm::vec4 m_splitDistances = {};
		alignas(4) int m_pcfSampleAmount = 32;
		alignas(4) int m_blockerSearchAmount = 2;
		alignas(4) float m_seamFixRatio = 0.3f;
		alignas(4) float m_gamma = 2.2f;

		alignas(4) float m_strandsSubdivisionXFactor = 50.0f;
		alignas(4) float m_strandsSubdivisionYFactor = 50.0f;
		alignas(4) int m_strandsSubdivisionMaxX = 15;
		alignas(4) int m_strandsSubdivisionMaxY = 8;

		alignas(4) int m_directionalLightSize = 0;
		alignas(4) int m_pointLightSize = 0;
		alignas(4) int m_spotLightSize = 0;
	};

	struct EnvironmentInfoBlock {
		glm::vec4 m_backgroundColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
		alignas(4) float m_environmentalMapGamma = 1.0f;
		alignas(4) float m_environmentalLightingIntensity = 0.8f;
		alignas(4) float m_backgroundIntensity = 1.0f;
		alignas(4) float m_environmentalPadding2 = 0.0f;
	};

	struct InstanceInfoBlock
	{
		GlobalTransform m_model;
	};
#pragma endregion

	class RenderLayer final : public ILayer {
		friend class Resources;
		friend class Camera;
		friend class GraphicsPipeline;
		friend class EditorLayer;
		friend class Material;
		friend class Lighting;
		
		void OnCreate() override;
		void OnDestroy() override;
		void PreUpdate() override;

		void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;

		void PreparePointAndSpotLightShadowMap();
		
		void PrepareEnvironmentalBrdfLut();

	public:
		bool m_enableRenderMenu = false;
		bool m_stableFit = true;
		float m_maxShadowDistance = 200;
		float m_shadowCascadeSplit[4] = { 0.075f, 0.15f, 0.3f, 1.0f };

		[[nodiscard]] uint32_t GetCameraIndex(const Handle& handle);
		[[nodiscard]] uint32_t GetMaterialIndex(const Handle& handle);
		[[nodiscard]] uint32_t GetInstanceIndex(const Handle& handle);
		[[nodiscard]] Handle GetInstanceHandle(uint32_t index);
		void UploadCameraInfoBlock(const Handle& handle, const CameraInfoBlock& cameraInfoBlock);
		void UploadMaterialInfoBlock(const Handle& handle, const MaterialInfoBlock& materialInfoBlock);
		void UploadInstanceInfoBlock(const Handle& handle, const InstanceInfoBlock& instanceInfoBlock);

		[[nodiscard]] uint32_t RegisterCameraIndex(const Handle& handle, const CameraInfoBlock& cameraInfoBlock, bool upload = false);
		[[nodiscard]] uint32_t RegisterMaterialIndex(const Handle& handle, const MaterialInfoBlock& materialInfoBlock, bool upload = false);
		[[nodiscard]] uint32_t RegisterInstanceIndex(const Handle& handle, const InstanceInfoBlock& instanceInfoBlock, bool upload = false);

		void UploadEnvironmentalInfoBlock(const EnvironmentInfoBlock& environmentInfoBlock) const;
		void UploadRenderInfoBlock(const RenderInfoBlock& renderInfoBlock) const;

		RenderInfoBlock m_renderInfoBlock = {};
		EnvironmentInfoBlock m_environmentInfoBlock = {};

		void RenderToCamera(const GlobalTransform& cameraGlobalTransform, const std::shared_ptr<Camera>& camera);

	private:
		bool m_needFade = false;
#pragma region Render procedure
		RenderInstanceCollection m_deferredRenderInstances;
		RenderInstanceCollection m_deferredInstancedRenderInstances;
		RenderInstanceCollection m_forwardRenderInstances;
		RenderInstanceCollection m_forwardInstancedRenderInstances;
		RenderInstanceCollection m_transparentRenderInstances;
		RenderInstanceCollection m_instancedTransparentRenderInstances;
		void CollectCameras(std::vector<std::pair<GlobalTransform, std::shared_ptr<Camera>>>& cameras);
		void CollectRenderInstances(Bound& worldBound);
		void CollectDirectionalLights(const std::vector<std::pair<GlobalTransform, std::shared_ptr<Camera>>>& cameras);
		void CollectPointLights();
		void CollectSpotLights();

		std::unique_ptr<Lighting> m_lighting;
		std::shared_ptr<Image> m_environmentalBRDFLut = {};
		std::shared_ptr<ImageView> m_environmentalBRDFView = {};
		std::shared_ptr<Sampler> m_environmentalBRDFSampler = {};
#pragma endregion
#pragma region Per Frame Descriptor Sets
		std::vector<std::shared_ptr<DescriptorSet>> m_perFrameDescriptorSets = {};
		std::vector<RenderInfoBlock*> m_renderInfoBlockMemory;
		std::vector<EnvironmentInfoBlock*> m_environmentalInfoBlockMemory;
		std::vector<CameraInfoBlock*> m_cameraInfoBlockMemory;
		std::vector<MaterialInfoBlock*> m_materialInfoBlockMemory;
		std::vector<InstanceInfoBlock*> m_instanceInfoBlockMemory;
		std::vector<glm::vec4*> m_kernelBlockMemory;
		std::vector<DirectionalLightInfo*> m_directionalLightInfoBlockMemory;
		std::vector<PointLightInfo*> m_pointLightInfoBlockMemory;
		std::vector<SpotLightInfo*> m_spotLightInfoBlockMemory;

		std::vector<std::unique_ptr<Buffer>> m_renderInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_environmentInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_cameraInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_materialInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_objectInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_kernelDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_directionalLightInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_pointLightInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_spotLightInfoDescriptorBuffers = {};
		void CreateStandardDescriptorBuffers();
		void UpdateStandardBindings();

		std::unordered_map<Handle, uint32_t> m_cameraIndices;
		std::unordered_map<Handle, uint32_t> m_materialIndices;
		std::unordered_map<Handle, uint32_t> m_instanceIndices;
		std::unordered_map<uint32_t, Handle> m_instanceHandles;

		std::vector<CameraInfoBlock> m_cameraInfoBlocks{};
		std::vector<MaterialInfoBlock> m_materialInfoBlocks{};
		std::vector<InstanceInfoBlock> m_instanceInfoBlocks{};

		std::vector<DirectionalLightInfo> m_directionalLightInfoBlocks;
		std::vector<PointLightInfo> m_pointLightInfoBlocks;
		std::vector<SpotLightInfo> m_spotLightInfoBlocks;
		void UploadCameraInfoBlocks(const std::vector<CameraInfoBlock>& cameraInfoBlocks) const;
		void UploadMaterialInfoBlocks(const std::vector<MaterialInfoBlock>& materialInfoBlocks) const;
		void UploadInstanceInfoBlocks(const std::vector<InstanceInfoBlock>& objectInfoBlocks) const;

		void UploadDirectionalLightInfoBlocks(const std::vector<DirectionalLightInfo>& directionalLightInfoBlocks) const;
		void UploadPointLightInfoBlocks(const std::vector<PointLightInfo>& pointLightInfoBlocks) const;
		void UploadSpotLightInfoBlocks(const std::vector<SpotLightInfo>& spotLightInfoBlocks) const;

#pragma endregion
	};
}
