#pragma once
#include "ISingleton.hpp"
#include "GraphicsResources.hpp"
#include "Mesh.hpp"

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

	class Graphics final : ISingleton<Graphics>
	{
		friend class Application;
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

		QueueFamilyIndices m_queueFamilyIndices;

		VkQueue m_vkGraphicsQueue = VK_NULL_HANDLE;
		VkQueue m_vkPresentQueue = VK_NULL_HANDLE;

		std::shared_ptr<Swapchain> m_swapChain;
		std::vector<std::shared_ptr<Framebuffer>> m_framebuffers;
		
#pragma endregion
		std::shared_ptr<PipelineLayout> m_pipelineLayout;
		std::shared_ptr<RenderPass> m_renderPass;
		std::shared_ptr<GraphicsPipeline> m_graphicsPipeline;

		std::shared_ptr<Mesh> m_mesh;

		CommandPool m_commandPool;

		int m_maxFrameInFlight = 2;

		std::vector<VkCommandBuffer> m_vkCommandBuffers = {};

		std::vector<std::shared_ptr<Semaphore>> m_imageAvailableSemaphores;
		std::vector<std::shared_ptr<Semaphore>> m_renderFinishedSemaphores;
		std::vector<std::shared_ptr<Fence>> m_inFlightFences;
		uint32_t m_currentFrameIndex = 0;

		uint32_t m_nextImageIndex = 0;

		QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice physicalDevice);
		SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice physicalDevice);
		bool IsDeviceSuitable(VkPhysicalDevice physicalDevice, const std::vector<std::string>& requiredDeviceExtensions);

		void CreateInstance();
		void CreateSurface();
		void CreateDebugMessenger();
		void CreatePhysicalDevice();
		void CreateLogicalDevice();
		void SetupVmaAllocator();
		void CreateCommandPool();
		void CreateCommandBuffers();
		void CreateSwapChainSyncObjects();
		void CreateSwapChain();

		void CleanupSwapChain();
		void CreateFramebuffers();

		void CreateRenderPass();
		void CreateGraphicsPipeline();
		void RecreateSwapChain();

		void OnDestroy();

		void SwapChainSwapImage();
		void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

		void SubmitPresent();
		static void Initialize();
		static void Destroy();
		static void PreUpdate();
		static void Update();
		static void LateUpdate();

		bool m_recreateSwapChain = false;

		static uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	public:

		static void NotifyRecreateSwapChain();
		static VkPhysicalDevice GetVkPhysicalDevice();
		static VkDevice GetVkDevice();

		static VkCommandPool GetVkCommandPool();
		static VkQueue GetGraphicsVkQueue();
		static VkQueue GetPresentVkQueue();
		static VmaAllocator GetVmaAllocator();
		[[nodiscard]] static bool CheckExtensionSupport(const std::string& extensionName);
		[[nodiscard]] static bool CheckLayerSupport(const std::string& layerName);
	};
}
