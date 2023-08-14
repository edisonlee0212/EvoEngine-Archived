#pragma once
#include "Camera.hpp"
#include "ILayer.hpp"
#include "Material.hpp"
#include "Mesh.hpp"
#include "IGeometry.hpp"
#include "Lights.hpp"
#include "SkinnedMeshRenderer.hpp"
#include "Strands.hpp"

namespace EvoEngine
{
#pragma region Enums Structs
	enum class RenderCommandType {
		None,
		FromRenderer,
		FromAPI,
	};

	struct RenderInstancePushConstant
	{
		int m_instanceIndex = 0;
		int m_cameraIndex = 0;
		int m_lightSplitIndex = 0;
	};

	struct RenderInstance {
		uint32_t m_instanceIndex = 0;
		RenderCommandType m_commandType = RenderCommandType::None;
		Entity m_owner = Entity();
		std::shared_ptr<Mesh> m_mesh;
		bool m_castShadow = true;

		uint32_t m_meshletSize = 0;
	};

	struct SkinnedRenderInstance {
		uint32_t m_instanceIndex = 0;
		RenderCommandType m_commandType = RenderCommandType::None;
		Entity m_owner = Entity();
		std::shared_ptr<SkinnedMesh> m_skinnedMesh;
		bool m_castShadow = true;
		std::shared_ptr<BoneMatrices> m_boneMatrices; // We require the skinned mesh renderer to provide bones.

		uint32_t m_skinnedMeshletSize = 0;
	};

	struct InstancedRenderInstance
	{
		uint32_t m_instanceIndex = 0;
		RenderCommandType m_commandType = RenderCommandType::None;
		Entity m_owner = Entity();
		std::shared_ptr<Mesh> m_mesh;
		bool m_castShadow = true;
		std::shared_ptr<ParticleInfoList> m_particleInfos;

		uint32_t m_meshletSize = 0;
	};

	struct StrandsRenderInstance
	{
		uint32_t m_instanceIndex = 0;
		RenderCommandType m_commandType = RenderCommandType::None;
		Entity m_owner = Entity();
		std::shared_ptr<Strands> m_strands;
		bool m_castShadow = true;

		uint32_t m_strandMeshletSize = 0;
	};

	struct RenderInstanceCollection
	{
		std::vector<RenderInstance> m_renderCommands;
		void Dispatch(const std::function<void(const RenderInstance&)>& commandAction) const;
	};
	struct SkinnedRenderInstanceCollection
	{
		std::vector<SkinnedRenderInstance> m_renderCommands;
		void Dispatch(const std::function<void(const SkinnedRenderInstance&)>& commandAction) const;
	};
	struct StrandsRenderInstanceCollection
	{
		std::vector<StrandsRenderInstance> m_renderCommands;
		void Dispatch(const std::function<void(const StrandsRenderInstance&)>& commandAction) const;
	};
	struct InstancedRenderInstanceCollection
	{
		std::vector<InstancedRenderInstance> m_renderCommands;
		void Dispatch(const std::function<void(const InstancedRenderInstance&)>& commandAction) const;
	};
	struct RenderInfoBlock {
		glm::vec4 m_splitDistances = {};
		alignas(4) int m_pcfSampleAmount = 16;
		alignas(4) int m_blockerSearchAmount = 1;
		alignas(4) float m_seamFixRatio = 0.1f;
		alignas(4) float m_gamma = 2.2f;

		alignas(4) float m_strandsSubdivisionXFactor = 50.0f;
		alignas(4) float m_strandsSubdivisionYFactor = 50.0f;
		alignas(4) int m_strandsSubdivisionMaxX = 15;
		alignas(4) int m_strandsSubdivisionMaxY = 8;

		alignas(4) int m_directionalLightSize = 0;
		alignas(4) int m_pointLightSize = 0;
		alignas(4) int m_spotLightSize = 0;
		alignas(4) int m_brdflutTextureIndex = 0;
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
		GlobalTransform m_model = {};
		uint32_t m_materialIndex = 0;
		uint32_t m_entitySelected = 0;
		uint32_t m_meshletIndexOffset = 0;
		uint32_t m_renderBufferIndex = 0;
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
		void LateUpdate() override;
		void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;

		void PreparePointAndSpotLightShadowMap() const;
		
		void PrepareEnvironmentalBrdfLut();
		void RenderToCamera(const GlobalTransform& cameraGlobalTransform, const std::shared_ptr<Camera>& camera);

	public:
		bool m_enableIndirectRendering = true;
		bool m_enableMeshletDebug = false;
		bool m_enableRenderMenu = false;
		bool m_stableFit = true;
		float m_maxShadowDistance = 100;
		float m_shadowCascadeSplit[4] = { 0.075f, 0.15f, 0.3f, 1.0f };

		[[nodiscard]] uint32_t GetCameraIndex(const Handle& handle);
		[[nodiscard]] uint32_t GetMaterialIndex(const Handle& handle);
		[[nodiscard]] uint32_t GetInstanceIndex(const Handle& handle);
		[[nodiscard]] Handle GetInstanceHandle(uint32_t index);
		
		[[nodiscard]] uint32_t RegisterCameraIndex(const Handle& handle, const CameraInfoBlock& cameraInfoBlock);
		[[nodiscard]] uint32_t RegisterMaterialIndex(const Handle& handle, const MaterialInfoBlock& materialInfoBlock);
		[[nodiscard]] uint32_t RegisterInstanceIndex(const Handle& handle, const InstanceInfoBlock& instanceInfoBlock);

		RenderInfoBlock m_renderInfoBlock = {};
		EnvironmentInfoBlock m_environmentInfoBlock = {};
		void DrawMesh(const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<Material>& material, glm::mat4 model, bool castShadow);
		
		[[nodiscard]] const std::shared_ptr<DescriptorSet>& GetPerFrameDescriptorSet() const;
	private:
		std::vector<std::pair<GlobalTransform, std::shared_ptr<Camera>>> m_cameras;
		bool m_needFade = false;
#pragma region Render procedure
		RenderInstanceCollection m_deferredRenderInstances;
		SkinnedRenderInstanceCollection m_deferredSkinnedRenderInstances;
		InstancedRenderInstanceCollection m_deferredInstancedRenderInstances;
		StrandsRenderInstanceCollection m_deferredStrandsRenderInstances;

		RenderInstanceCollection m_transparentRenderInstances;
		SkinnedRenderInstanceCollection m_transparentSkinnedRenderInstances;
		InstancedRenderInstanceCollection m_transparentInstancedRenderInstances;
		StrandsRenderInstanceCollection m_transparentStrandsRenderInstances;
		void CollectCameras(std::vector<std::pair<GlobalTransform, std::shared_ptr<Camera>>>& cameras);
		void CollectRenderInstances(Bound& worldBound);
		void CollectDirectionalLights(const std::vector<std::pair<GlobalTransform, std::shared_ptr<Camera>>>& cameras);
		void CollectPointLights();
		void CollectSpotLights();

		std::unique_ptr<Lighting> m_lighting;
		std::shared_ptr<Texture2D> m_environmentalBRDFLut = {};

		void ApplyAnimator() const;

#pragma endregion
#pragma region Per Frame Descriptor Sets
		friend class TextureStorage;
		std::vector<std::shared_ptr<DescriptorSet>> m_perFrameDescriptorSets = {};

		std::vector<std::unique_ptr<Buffer>> m_renderInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_environmentInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_cameraInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_materialInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_instanceInfoDescriptorBuffers = {};
		
		std::vector<std::unique_ptr<Buffer>> m_kernelDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_directionalLightInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_pointLightInfoDescriptorBuffers = {};
		std::vector<std::unique_ptr<Buffer>> m_spotLightInfoDescriptorBuffers = {};

		void CreateStandardDescriptorBuffers();
		void CreatePerFrameDescriptorSets();

		std::unordered_map<Handle, uint32_t> m_cameraIndices;
		std::unordered_map<Handle, uint32_t> m_materialIndices;
		std::unordered_map<Handle, uint32_t> m_instanceIndices;
		std::unordered_map<uint32_t, Handle> m_instanceHandles;

		std::vector<CameraInfoBlock> m_cameraInfoBlocks{};
		std::vector<MaterialInfoBlock> m_materialInfoBlocks{};
		std::vector<InstanceInfoBlock> m_instanceInfoBlocks{};

		std::vector<std::unique_ptr<Buffer>> m_meshDrawIndexedIndirectCommandsBuffers = {};
		std::vector<VkDrawIndexedIndirectCommand> m_meshDrawIndexedIndirectCommands{};
		uint32_t m_totalMeshTriangles = 0;

		std::vector<std::unique_ptr<Buffer>> m_meshDrawMeshTasksIndirectCommandsBuffers = {};
		std::vector<VkDrawMeshTasksIndirectCommandEXT> m_meshDrawMeshTasksIndirectCommands{};

		std::vector<DirectionalLightInfo> m_directionalLightInfoBlocks;
		std::vector<PointLightInfo> m_pointLightInfoBlocks;
		std::vector<SpotLightInfo> m_spotLightInfoBlocks;
		
#pragma endregion
	};
}
