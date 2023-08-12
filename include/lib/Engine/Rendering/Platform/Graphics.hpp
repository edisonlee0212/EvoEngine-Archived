#pragma once
#include "GraphicsPipeline.hpp"
#include "ISingleton.hpp"
#include "GraphicsResources.hpp"
#include "GraphicsPipelineStates.hpp"
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

	class Graphics final : public ISingleton<Graphics>
	{
		friend class Application;
		friend class Resources;
		friend class Lighting;
		friend class PointLightShadowMap;
		friend class SpotLightShadowMap;
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

		int m_usedCommandBufferSize = 0;
		std::vector<std::vector<CommandBuffer>> m_commandBufferPool = {};

		std::vector<std::unique_ptr<Semaphore>> m_imageAvailableSemaphores = {};
		std::vector<std::unique_ptr<Semaphore>> m_renderFinishedSemaphores = {};
		std::vector<std::unique_ptr<Fence>> m_inFlightFences = {};
		uint32_t m_currentFrameIndex = 0;

		uint32_t m_nextImageIndex = 0;
		
#pragma endregion
#pragma region Shader related
		std::string m_shaderBasic;
		std::string m_shaderBasicConstants;
		std::string m_shaderGizmosConstants;
		std::string m_shaderLighting;
		std::string m_shaderSkybox;
		size_t m_maxBoneAmount = 65536;
		size_t m_maxShadowCascadeAmount = 4;
		friend class RenderLayer;
		
#pragma endregion



		QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice physicalDevice) const;
		SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice physicalDevice) const;
		bool IsDeviceSuitable(VkPhysicalDevice physicalDevice, const std::vector<std::string>& requiredDeviceExtensions) const;

		void CreateInstance();
		void CreateSurface();
		void CreateDebugMessenger();
		void CreatePhysicalDevice();
		void CreateLogicalDevice();
		void SetupVmaAllocator();

		void CreateSwapChain();

		void CreateSwapChainSyncObjects();

		void RecreateSwapChain();

		void OnDestroy();
		void SwapChainSwapImage();
		void SubmitPresent();
		void WaitForCommandsComplete();
		//void Submit();

		void ResetCommandBuffers();
		static void Initialize();
		static void PostResourceLoadingInitialization();
		static void Destroy();
		static void PreUpdate();
		static void LateUpdate();

		bool m_recreateSwapChain = false;
		unsigned m_swapchainVersion = 0;
		static uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

		
		std::unordered_map<std::string, std::shared_ptr<GraphicsPipeline>> m_graphicsPipelines;
		std::unordered_map<std::string, std::shared_ptr<DescriptorSetLayout>> m_descriptorSetLayouts;
		void CreateGraphicsPipelines() const;
		void PrepareDescriptorSetLayouts() const;

	public:
		static void RegisterGraphicsPipeline(const std::string& name, const std::shared_ptr<GraphicsPipeline>& graphicsPipeline);
		[[nodiscard]] static const std::shared_ptr<GraphicsPipeline>& GetGraphicsPipeline(const std::string& name);
		static void RegisterDescriptorSetLayout(const std::string& name, const std::shared_ptr<DescriptorSetLayout>& descriptorSetLayout);
		[[nodiscard]] static const std::shared_ptr<DescriptorSetLayout>& GetDescriptorSetLayout(const std::string& name);

		size_t m_triangles = 0;
		size_t m_strandsSegments = 0;
		size_t m_drawCall = 0;


		class Constants
		{
		public:
			friend class RenderLayer;
			friend class Resources;
			friend class Lighting;
			friend class PointLightShadowMap;
			friend class SpotLightShadowMap;
			inline constexpr static uint32_t INITIAL_CAMERA_SIZE = 8;
			inline constexpr static uint32_t INITIAL_MATERIAL_SIZE = 1024;
			inline constexpr static uint32_t INITIAL_INSTANCE_SIZE = 8192;
			inline constexpr static uint32_t INITIAL_RENDER_TASK_SIZE = 8192;
			inline constexpr static uint32_t VERTEX_DATA_CHUNK_VERTICES_SIZE = 64;
			inline constexpr static uint32_t MAX_KERNEL_AMOUNT = 64;
			inline constexpr static uint32_t MAX_DIRECTIONAL_LIGHT_SIZE = 16;
			inline constexpr static uint32_t MAX_POINT_LIGHT_SIZE = 16;
			inline constexpr static uint32_t MAX_SPOT_LIGHT_SIZE = 16;

			inline constexpr static uint32_t MAX_TEXTURE_2D_RESOURCE_SIZE = 8192;
			inline constexpr static uint32_t MAX_CUBEMAP_RESOURCE_SIZE = 8192;
			inline constexpr static uint32_t MESHLET_MAX_VERTICES_SIZE = 64;
			inline constexpr static uint32_t MESHLET_MAX_TRIANGLES_SIZE = 84;

			inline constexpr static uint32_t DIRECTIONAL_LIGHT_SHADOW_MAP_RESOLUTION = 2048;
			inline constexpr static uint32_t POINT_LIGHT_SHADOW_MAP_RESOLUTION = 2048;
			inline constexpr static uint32_t SPOT_LIGHT_SHADOW_MAP_RESOLUTION = 2048;
			inline constexpr static VkFormat TEXTURE_2D = VK_FORMAT_R32G32B32A32_SFLOAT;
			inline constexpr static VkFormat RENDER_TEXTURE_DEPTH = VK_FORMAT_D32_SFLOAT;
			inline constexpr static VkFormat RENDER_TEXTURE_COLOR = VK_FORMAT_R32G32B32A32_SFLOAT;
			inline constexpr static VkFormat G_BUFFER_DEPTH = VK_FORMAT_D32_SFLOAT;
			inline constexpr static VkFormat G_BUFFER_COLOR = VK_FORMAT_R16G16B16A16_SFLOAT;
			inline constexpr static VkFormat SHADOW_MAP = VK_FORMAT_D32_SFLOAT;
		};

		static void EverythingBarrier(VkCommandBuffer commandBuffer);

		static void TransitImageLayout(VkCommandBuffer commandBuffer, VkImage targetImage, VkFormat imageFormat, uint32_t layerCount, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels = 1);

		static std::string StringifyResultVk(const VkResult& result);
		static void CheckVk(const VkResult& result);

		static size_t GetMaxBoneAmount();		
		static size_t GetMaxShadowCascadeAmount();
		static void AppendCommands(const std::function<void(VkCommandBuffer commandBuffer)>& action);
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
		static const std::unique_ptr<Swapchain>& GetSwapchain();
		static const std::unique_ptr<DescriptorPool>& GetDescriptorPool();
		static unsigned GetSwapchainVersion();
		static VkSurfaceFormatKHR GetVkSurfaceFormat();
		static const VkPhysicalDeviceProperties& GetVkPhysicalDeviceProperties();
		[[nodiscard]] static bool CheckExtensionSupport(const std::string& extensionName);
		[[nodiscard]] static bool CheckLayerSupport(const std::string& layerName);
	};
}
