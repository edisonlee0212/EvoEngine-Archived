#pragma once
#include "ISingleton.hpp"
#include "GraphicsResources.hpp"

namespace EvoEngine
{
	struct QueueFamilyIndices {
		std::optional<uint32_t> m_graphicsFamily;
		std::optional<uint32_t> m_presentFamily;
		[[nodiscard]] bool IsComplete() const {
			return m_graphicsFamily.has_value() && m_presentFamily.has_value();
		}
	};

	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR m_capabilities;
		std::vector<VkSurfaceFormatKHR> m_formats;
		std::vector<VkPresentModeKHR> m_presentModes;
	};

	class GlobalPipelineState
	{
		friend class Graphics;
		VkViewport m_viewPortApplied = {};
		VkRect2D m_scissorApplied = {};
		uint32_t m_patchControlPointsApplied = 1;
		bool m_depthClampApplied = false;
		bool m_rasterizerDiscardApplied = false;
		VkPolygonMode m_polygonModeApplied = VK_POLYGON_MODE_FILL;
		VkCullModeFlags m_cullModeApplied = VK_CULL_MODE_BACK_BIT;
		VkFrontFace m_frontFaceApplied = VK_FRONT_FACE_CLOCKWISE;
		bool m_depthBiasApplied = false;
		glm::vec3 m_depthBiasConstantClampSlopeApplied = glm::vec3(0.0f);
		float m_lineWidthApplied = 1.0f;
		bool m_depthTestApplied = true;
		bool m_depthWriteApplied = true;
		VkCompareOp m_depthCompareApplied = VK_COMPARE_OP_LESS;
		bool m_depthBoundTestApplied = false;
		glm::vec2 m_minMaxDepthBoundApplied = glm::vec2(-1.0f, 1.0f);
		bool m_stencilTestApplied = false;
		VkStencilFaceFlags m_stencilFaceMaskApplied = VK_STENCIL_FACE_FRONT_BIT;
		VkStencilOp m_stencilFailOpApplied = VK_STENCIL_OP_ZERO;
		VkStencilOp m_stencilPassOpApplied = VK_STENCIL_OP_ZERO;
		VkStencilOp m_stencilDepthFailOpApplied = VK_STENCIL_OP_ZERO;
		VkCompareOp m_stencilCompareOpApplied = VK_COMPARE_OP_LESS;

		void ResetAllStates(VkCommandBuffer commandBuffer);
	public:
		VkViewport m_viewPort = {};
		VkRect2D m_scissor = {};
		uint32_t m_patchControlPoints = 1;
		bool m_depthClamp = false;
		bool m_rasterizerDiscard = false;
		VkPolygonMode m_polygonMode = VK_POLYGON_MODE_FILL;
		VkCullModeFlags m_cullMode = VK_CULL_MODE_BACK_BIT;
		VkFrontFace m_frontFace = VK_FRONT_FACE_CLOCKWISE;
		bool m_depthBias = false;
		glm::vec3 m_depthBiasConstantClampSlope = glm::vec3(0.0f);
		float m_lineWidth = 1.0f;
		bool m_depthTest = true;
		bool m_depthWrite = true;
		VkCompareOp m_depthCompare = VK_COMPARE_OP_LESS;
		bool m_depthBoundTest = false;
		glm::vec2 m_minMaxDepthBound = glm::vec2(0.0f, 1.0f);
		bool m_stencilTest = false;
		VkStencilFaceFlags m_stencilFaceMask = VK_STENCIL_FACE_FRONT_BIT;
		VkStencilOp m_stencilFailOp = VK_STENCIL_OP_ZERO;
		VkStencilOp m_stencilPassOp = VK_STENCIL_OP_ZERO;
		VkStencilOp m_stencilDepthFailOp = VK_STENCIL_OP_ZERO;
		VkCompareOp m_stencilCompareOp = VK_COMPARE_OP_LESS;

		void ApplyAllStates(VkCommandBuffer commandBuffer, bool forceSet = false);
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

	struct CameraInfoBlock
	{
		glm::mat4 m_projection = {};
		glm::mat4 m_view = {};
		glm::mat4 m_projectionView = {};
		glm::mat4 m_inverseProjection = {};
		glm::mat4 m_inverseView = {};
		glm::mat4 m_inverseProjectionView = {};
		glm::vec4 m_clearColor = {};
		glm::vec4 m_reservedParameters1 = {};
		glm::vec4 m_reservedParameters2 = {};

		[[nodiscard]] glm::vec3 Project(const glm::vec3& position) const;
		[[nodiscard]] glm::vec3 UnProject(const glm::vec3& position) const;
	};

	struct MaterialInfoBlock {
		alignas(4) bool m_albedoEnabled = false;
		alignas(4) bool m_normalEnabled = false;
		alignas(4) bool m_metallicEnabled = false;
		alignas(4) bool m_roughnessEnabled = false;

		alignas(4) bool m_aoEnabled = false;
		alignas(4) bool m_castShadow = true;
		alignas(4) bool m_receiveShadow = true;
		alignas(4) bool m_enableShadow = true;

		glm::vec4 m_albedoColorVal = glm::vec4(1.0f);
		glm::vec4 m_subsurfaceColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
		glm::vec4 m_subsurfaceRadius = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);

		alignas(4) float m_metallicVal = 0.5f;
		alignas(4) float m_roughnessVal = 0.5f;
		alignas(4) float m_aoVal = 1.0f;
		alignas(4) float m_emissionVal = 0.0f;
	};

	struct ObjectInfoBlock
	{
		glm::mat4 m_model;
	};

	class Graphics final : public ISingleton<Graphics>
	{
		friend class Application;
		friend class Resources;
#pragma region Vulkan
		VkInstance m_vkInstance = VK_NULL_HANDLE;
		std::vector<std::string> m_requiredDeviceExtensions = {};
		std::vector<std::string> m_requiredLayers = {};
		std::vector<VkExtensionProperties> m_vkExtensions;
		std::vector<VkLayerProperties> m_vkLayers;
		VkDebugUtilsMessengerEXT m_vkDebugMessenger = {};
		VkPhysicalDeviceFeatures m_vkPhysicalDeviceFeatures = {};
		VkPhysicalDeviceProperties m_vkPhysicalDeviceProperties = {};

		VkSurfaceKHR m_vkSurface = VK_NULL_HANDLE;
		VkPhysicalDevice m_vkPhysicalDevice = VK_NULL_HANDLE;
		VkDevice m_vkDevice = VK_NULL_HANDLE;

		VmaAllocator m_vmaAllocator = VK_NULL_HANDLE;

		QueueFamilyIndices m_queueFamilyIndices = {};

		VkQueue m_vkGraphicsQueue = VK_NULL_HANDLE;
		VkQueue m_vkPresentQueue = VK_NULL_HANDLE;

		std::unique_ptr<Swapchain> m_swapchain = {};

		VkSurfaceFormatKHR m_vkSurfaceFormat = {};


#pragma endregion
#pragma region Internals
		std::unique_ptr<CommandPool> m_commandPool = {};
		std::unique_ptr<DescriptorPool> m_descriptorPool = {};

		int m_maxFrameInFlight = 2;

		std::vector<VkCommandBuffer> m_vkCommandBuffers = {};

		std::vector<std::unique_ptr<Semaphore>> m_imageAvailableSemaphores = {};
		std::vector<std::unique_ptr<Semaphore>> m_renderFinishedSemaphores = {};
		std::vector<std::unique_ptr<Fence>> m_inFlightFences = {};
		uint32_t m_currentFrameIndex = 0;

		uint32_t m_nextImageIndex = 0;

		GlobalPipelineState m_globalPipelineState = {};

		std::unique_ptr<RenderPass> m_swapChainRenderPass = {};
		std::vector<std::unique_ptr<Framebuffer>> m_swapChainFramebuffers = {};
#pragma endregion
#pragma region Shader related
		std::unique_ptr<std::string> m_standardShaderIncludes;
		size_t m_maxBoneAmount = 65536;
		size_t m_maxMaterialAmount = 1;
		size_t m_maxKernelAmount = 64;
		size_t m_maxDirectionalLightAmount = 8;
		size_t m_maxPointLightAmount = 8;
		size_t m_maxSpotLightAmount = 8;
		size_t m_shadowCascadeAmount = 4;

		friend class ShaderProgram;
		std::vector<void*> m_renderInfoBlockMemory;
		std::vector<void*> m_environmentalInfoBlockMemory;
		std::vector<void*> m_cameraInfoBlockMemory;
		std::vector<void*> m_materialInfoBlockMemory;
		std::vector<void*> m_objectInfoBlockMemory;
		std::vector<std::unique_ptr<Buffer>> m_standardDescriptorBuffers = {};
#pragma endregion
		QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice physicalDevice);
		SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice physicalDevice);
		bool IsDeviceSuitable(VkPhysicalDevice physicalDevice, const std::vector<std::string>& requiredDeviceExtensions);

		void CreateInstance();
		void CreateSurface();
		void CreateDebugMessenger();
		void CreatePhysicalDevice();
		void CreateLogicalDevice();
		void SetupVmaAllocator();

		void CreateSwapChain();
		void CreateRenderPass();
		bool UpdateFrameBuffers();

		void CreateSwapChainSyncObjects();
		void CreateStandardDescriptorLayout();

		void RecreateSwapChain();

		void OnDestroy();
		void SwapChainSwapImage();
		void SubmitPresent();
		static void Initialize();
		static void Destroy();
		static void PreUpdate();
		static void LateUpdate();

		bool m_recreateSwapChain = false;
		unsigned m_swapchainVersion = 0;
		static uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	public:
		static const std::string& GetStandardShaderIncludes();
		static size_t GetMaxBoneAmount();
		static size_t GetMaxMaterialAmount();
		static size_t GetMaxKernelAmount();
		static size_t GetMaxDirectionalLightAmount();
		static size_t GetMaxPointLightAmount();
		static size_t GetMaxSpotLightAmount();
		static size_t GetMaxShadowCascadeAmount();

		static void UploadEnvironmentInfo(const EnvironmentInfoBlock& environmentInfoBlock);
		static void UploadRenderInfo(const RenderInfoBlock& renderInfoBlock);
		static void UploadCameraInfo(const CameraInfoBlock& cameraInfoBlock);
		static void UploadMaterialInfo(const MaterialInfoBlock& materialInfoBlock);
		static void UploadObjectInfo(const ObjectInfoBlock& objectInfoBlock);

		static const std::unique_ptr<RenderPass>& GetSwapchainRenderPass();
		static const std::unique_ptr<Framebuffer>& GetSwapchainFramebuffer();
		static GlobalPipelineState& GlobalState();
		static void CreateCommandBuffers(const std::unique_ptr <CommandPool>& commandPool, std::vector<VkCommandBuffer>& commandBuffers);
		static void AppendCommands(const std::function<void(VkCommandBuffer commandBuffer, GlobalPipelineState& globalPipelineState)>& action);
		static void ImmediateSubmit(const std::function<void(VkCommandBuffer commandBuffer)>& action);
		static QueueFamilyIndices GetQueueFamilyIndices();
		static int GetMaxFramesInFlight();
		static void NotifyRecreateSwapChain();
		static VkInstance GetVkInstance();
		static VkPhysicalDevice GetVkPhysicalDevice();
		static VkDevice GetVkDevice();
		static uint32_t GetCurrentFrameIndex();
		static uint32_t GetNextImageIndex();
		static VkCommandPool GetVkCommandPool();
		static VkQueue GetGraphicsVkQueue();
		static VkQueue GetPresentVkQueue();
		static VmaAllocator GetVmaAllocator();
		static VkCommandBuffer GetCurrentVkCommandBuffer();
		static const std::unique_ptr<Swapchain>& GetSwapchain();
		static const std::unique_ptr<DescriptorPool>& GetDescriptorPool();
		static unsigned GetSwapchainVersion();
		static VkSurfaceFormatKHR GetVkSurfaceFormat();
		static const VkPhysicalDeviceProperties& GetVkPhysicalDeviceProperties();
		[[nodiscard]] static bool CheckExtensionSupport(const std::string& extensionName);
		[[nodiscard]] static bool CheckLayerSupport(const std::string& layerName);
	};
}