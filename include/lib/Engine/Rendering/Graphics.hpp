#pragma once
#include "ISingleton.hpp"
#include "GraphicsResources.hpp"
#include "GraphicsGlobalStates.hpp"
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

		std::vector<std::unordered_map<std::string, CommandBuffer>> m_vkCommandBuffers = {};

		std::vector<std::unique_ptr<Semaphore>> m_imageAvailableSemaphores = {};
		std::vector<std::unique_ptr<Semaphore>> m_renderFinishedSemaphores = {};
		std::vector<std::unique_ptr<Fence>> m_inFlightFences = {};
		uint32_t m_currentFrameIndex = 0;

		uint32_t m_nextImageIndex = 0;

		GraphicsGlobalStates m_globalPipelineState = {};

		std::shared_ptr<RenderPass> m_swapChainRenderPass = {};
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

		friend class GraphicsPipeline;
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
		static std::string StringifyResultVk(const VkResult& result);
		static void CheckVk(const VkResult& result);
#pragma region Formats
		class ImageFormats
		{
		public:
			inline static VkFormat m_texture2D = VK_FORMAT_R32G32B32A32_SFLOAT;
			inline static VkFormat m_renderTextureDepthStencil = VK_FORMAT_D24_UNORM_S8_UINT;
			inline static VkFormat m_renderTextureColor = VK_FORMAT_R16G16B16A16_SFLOAT;
			inline static VkFormat m_gBufferDepth = VK_FORMAT_D24_UNORM_S8_UINT;
			inline static VkFormat m_gBufferColor = VK_FORMAT_R16G16B16A16_SFLOAT;
		};

#pragma endregion
		static void RegisterCommandBuffer(const std::string& name);
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

		static const std::shared_ptr<RenderPass>& GetSwapchainRenderPass();
		static const std::unique_ptr<Framebuffer>& GetSwapchainFramebuffer();
		static GraphicsGlobalStates& GlobalState();
		static void AppendCommands(const std::string& name, const std::function<void(VkCommandBuffer commandBuffer, GraphicsGlobalStates& globalPipelineState)>& action);
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
		static VkCommandBuffer GetCurrentVkCommandBuffer(const std::string& name);
		static const std::unique_ptr<Swapchain>& GetSwapchain();
		static const std::unique_ptr<DescriptorPool>& GetDescriptorPool();
		static unsigned GetSwapchainVersion();
		static VkSurfaceFormatKHR GetVkSurfaceFormat();
		static const VkPhysicalDeviceProperties& GetVkPhysicalDeviceProperties();
		[[nodiscard]] static bool CheckExtensionSupport(const std::string& extensionName);
		[[nodiscard]] static bool CheckLayerSupport(const std::string& layerName);
	};
}
