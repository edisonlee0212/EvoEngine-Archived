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

	class Graphics final : ISingleton<Graphics>
	{
#pragma region Presenters
		std::vector<GLFWmonitor*> m_monitors;
		GLFWmonitor* m_primaryMonitor = nullptr;
		GLFWwindow* m_window = nullptr;
		glm::ivec2 m_windowSize = { 1 , 1 };
#pragma endregion

#pragma region Vulkan
		VkInstance m_vkInstance = VK_NULL_HANDLE;

		std::vector<VkExtensionProperties> m_vkExtensions;
		std::vector<VkLayerProperties> m_vkLayers;
		VkDebugUtilsMessengerEXT m_vkDebugMessenger = {};
		VkPhysicalDeviceFeatures m_vkPhysicalDeviceFeatures = {};
		VkPhysicalDeviceProperties m_vkPhysicalDeviceProperties = {};

		VkSurfaceKHR m_vkSurface = VK_NULL_HANDLE;
		VkPhysicalDevice m_vkPhysicalDevice = VK_NULL_HANDLE;
		VkDevice m_vkDevice = VK_NULL_HANDLE;

		VkQueue m_vkGraphicsQueue = VK_NULL_HANDLE;
		VkQueue m_vkPresentQueue = VK_NULL_HANDLE;

		VkSwapchainKHR m_vkSwapChain = VK_NULL_HANDLE;
		std::vector<VkImage> m_vkSwapChainVkImages;
		VkFormat m_vkSwapChainVkFormat = {};
		VkExtent2D m_vkSwapChainVkExtent2D = {};

		std::vector<VkImageView> m_vkSwapChainVkImageViews;
#pragma endregion
		PipelineLayout m_pipelineLayout;
		RenderPass m_renderPass;
		GraphicsPipeline m_graphicsPipeline;

		static QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice physicalDevice);
		static SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice physicalDevice);
		static bool IsDeviceSuitable(VkPhysicalDevice physicalDevice, const std::vector<std::string>& requiredDeviceExtensions);
		static void WindowResizeCallback(GLFWwindow*, int, int);
		static void SetMonitorCallback(GLFWmonitor* monitor, int event);
		static void WindowFocusCallback(GLFWwindow* window, int focused);
	public:
		static VkDevice GetVkDevice();

		static GLFWwindow* GetGlfwWindow();
		static void Initialize(const ApplicationCreateInfo& applicationCreateInfo, const VkApplicationInfo& vkApplicationInfo);
		static void Terminate();

		static bool CheckExtensionSupport(const std::string& extensionName);
		static bool CheckLayerSupport(const std::string& layerName);
	};
}