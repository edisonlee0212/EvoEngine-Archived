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

		VkQueue m_vkGraphicsQueue = VK_NULL_HANDLE;
		VkQueue m_vkPresentQueue = VK_NULL_HANDLE;

		Swapchain m_swapChain;
		
		std::vector<ImageView> m_swapChainImageViews;
#pragma endregion
		PipelineLayout m_pipelineLayout;
		RenderPass m_renderPass;
		GraphicsPipeline m_graphicsPipeline;

		std::vector<Framebuffer> m_framebuffers;

		CommandPool m_commandPool;
		VkCommandBuffer m_vkCommandBuffer = VK_NULL_HANDLE;

		Semaphore m_imageAvailableSemaphore;
		Semaphore m_renderFinishedSemaphore;
		Fence m_inFlightFence;

		uint32_t m_nextImageIndex = 0;

		void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
		QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice physicalDevice);
		SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice physicalDevice);
		bool IsDeviceSuitable(VkPhysicalDevice physicalDevice, const std::vector<std::string>& requiredDeviceExtensions);

		void OnCreate() override;
		void OnDestroy() override;
		void PreUpdate() override;
		void Update() override;
		void LateUpdate() override;
	public:
		static VkPhysicalDevice GetVkPhysicalDevice();
		static VkDevice GetVkDevice();

		
		
		
		bool CheckExtensionSupport(const std::string& extensionName) const;
		bool CheckLayerSupport(const std::string& layerName) const;
	};
}