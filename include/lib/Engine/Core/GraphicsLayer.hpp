#pragma once
#include "ISingleton.hpp"
#include "Application.hpp"
#include "GraphicsPipeline.hpp"
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

	class GraphicsLayer final : public ILayer
	{
#pragma region Vulkan
		static VkInstance m_vkInstance;

		static std::vector<VkExtensionProperties> m_vkExtensions;
		static std::vector<VkLayerProperties> m_vkLayers;
		static VkDebugUtilsMessengerEXT m_vkDebugMessenger;
		static VkPhysicalDeviceFeatures m_vkPhysicalDeviceFeatures;
		static VkPhysicalDeviceProperties m_vkPhysicalDeviceProperties;

		static VkSurfaceKHR m_vkSurface;
		static VkPhysicalDevice m_vkPhysicalDevice;
		static VkDevice m_vkDevice;

		QueueFamilyIndices m_queueFamilyIndices;

		VkQueue m_vkGraphicsQueue = VK_NULL_HANDLE;
		VkQueue m_vkPresentQueue = VK_NULL_HANDLE;

		Swapchain m_swapChain;
		std::vector<Framebuffer> m_framebuffers;
		
#pragma endregion
		PipelineLayout m_pipelineLayout;
		RenderPass m_renderPass;
		GraphicsPipeline m_graphicsPipeline;

		

		CommandPool m_commandPool;

		static int m_maxFrameInFlight;

		std::vector<VkCommandBuffer> m_vkCommandBuffers = {};

		std::vector<Semaphore> m_imageAvailableSemaphores;
		std::vector<Semaphore> m_renderFinishedSemaphores;
		std::vector<Fence> m_inFlightFences;
		uint32_t m_currentFrameIndex = 0;

		uint32_t m_nextImageIndex = 0;

		void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
		QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice physicalDevice);
		SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice physicalDevice);
		bool IsDeviceSuitable(VkPhysicalDevice physicalDevice, const std::vector<std::string>& requiredDeviceExtensions);

		void CreateSwapChain();

		void CleanupSwapChain();
		void CreateFramebuffers();

		void CreateRenderPass();
		void CreateGraphicsPipeline();
		void RecreateSwapChain();

		void OnCreate() override;
		void OnDestroy() override;
		void PreUpdate() override;
		void Update() override;
		void LateUpdate() override;

		bool m_recreateSwapChain = false;
	public:

		void NotifyRecreateSwapChain();
		static VkPhysicalDevice GetVkPhysicalDevice();
		static VkDevice GetVkDevice();
		[[nodiscard]] bool CheckExtensionSupport(const std::string& extensionName) const;
		[[nodiscard]] bool CheckLayerSupport(const std::string& layerName) const;
	};
}