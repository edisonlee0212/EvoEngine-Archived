#include "Graphics.hpp"
#include "Console.hpp"
#include "Application.hpp"
#include "Utilities.hpp"
#include "WindowLayer.hpp"
#define VOLK_IMPLEMENTATION
#define VMA_IMPLEMENTATION
#include "Mesh.hpp"
#include "ProjectManager.hpp"
#include "Vertex.hpp"
#include "vk_mem_alloc.h"
using namespace EvoEngine;


#pragma region Helpers
uint32_t Graphics::FindMemoryType(const uint32_t typeFilter, const VkMemoryPropertyFlags properties)
{
	const auto& graphics = GetInstance();
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(graphics.m_vkPhysicalDevice, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

void Graphics::SingleTimeCommands(const std::function<void(VkCommandBuffer commandBuffer)>& action)
{
	const auto& graphics = GetInstance();
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = graphics.m_commandPool.GetVkCommandPool();
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(graphics.m_vkDevice, &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);
	action(commandBuffer);
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(graphics.m_vkGraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(graphics.m_vkGraphicsQueue);

	vkFreeCommandBuffers(graphics.m_vkDevice, graphics.m_commandPool.GetVkCommandPool(), 1, &commandBuffer);
}

int Graphics::GetMaxFramesInFlight()
{
	const auto& graphics = GetInstance();
	return graphics.m_maxFrameInFlight;
}

void Graphics::NotifyRecreateSwapChain()
{
	auto& graphics = GetInstance();
	graphics.m_recreateSwapChain = true;
}

VkPhysicalDevice Graphics::GetVkPhysicalDevice()
{
	const auto& graphics = GetInstance();
	return graphics.m_vkPhysicalDevice;
}

VkDevice Graphics::GetVkDevice()
{
	const auto& graphics = GetInstance();
	return graphics.m_vkDevice;
}

uint32_t Graphics::GetCurrentFrameIndex()
{
	const auto& graphics = GetInstance();
	return graphics.m_currentFrameIndex;
}

uint32_t Graphics::GetNextImageIndex()
{
	const auto& graphics = GetInstance();
	return graphics.m_nextImageIndex;
}

VkCommandPool Graphics::GetVkCommandPool()
{
	const auto& graphics = GetInstance();
	return graphics.m_commandPool.GetVkCommandPool();
}

VkQueue Graphics::GetGraphicsVkQueue()
{
	const auto& graphics = GetInstance();
	return graphics.m_vkGraphicsQueue;
}

VkQueue Graphics::GetPresentVkQueue()
{
	const auto& graphics = GetInstance();
	return graphics.m_vkPresentQueue;
}

VmaAllocator Graphics::GetVmaAllocator()
{
	const auto& graphics = GetInstance();
	return graphics.m_vmaAllocator;
}

VkCommandBuffer Graphics::GetCurrentVkCommandBuffer()
{
	const auto& graphics = GetInstance();
	return graphics.m_vkCommandBuffers[graphics.m_currentFrameIndex];
}

Swapchain Graphics::GetSwapchain()
{
	const auto& graphics = GetInstance();
	return graphics.m_swapchain;
}

unsigned Graphics::GetSwapchainVersion()
{
	const auto& graphics = GetInstance();
	return graphics.m_swapchainVersion;
}

VkSurfaceFormatKHR Graphics::GetVkSurfaceFormat()
{
	const auto& graphics = GetInstance();
	return graphics.m_vkSurfaceFormat;
}

VkBool32 DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                       VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                       void* pUserData)
{
	//if (messageSeverity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) return VK_FALSE;
	std::string msg = "Vulkan";
	switch (messageType)
	{
	case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
	{
		msg += " [General]";
	}
	break;
	case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
	{
		msg += " [Validation]";
	}
	break; case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
	{
		msg += " [Performance]";
	}
	break;
	}
	switch (messageSeverity) {
	case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
	{
		msg += "-[Diagnostic]: ";
	}
	break;
	case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
	{
		msg += "-[Info]: ";
	}
	break;
	case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
	{
		msg += "-[Warning]: ";
	}
	break;
	case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
	{
		msg += "-[Error]: ";
	}
	break;
	}
	msg += std::string(pCallbackData->pMessage);
	EVOENGINE_LOG(msg);
	return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}


void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& debugUtilsMessengerCreateInfo) {
	debugUtilsMessengerCreateInfo = {};
	debugUtilsMessengerCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	debugUtilsMessengerCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	debugUtilsMessengerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	debugUtilsMessengerCreateInfo.pfnUserCallback = DebugCallback;
	debugUtilsMessengerCreateInfo.pUserData = nullptr;
}


bool CheckDeviceExtensionSupport(VkPhysicalDevice physicalDevice, const std::vector<std::string>& requiredPhysicalDeviceExtensions) {
	uint32_t extensionCount;
	vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

	std::set<std::string> requiredExtensions(requiredPhysicalDeviceExtensions.begin(), requiredPhysicalDeviceExtensions.end());

	for (const auto& extension : availableExtensions) {
		requiredExtensions.erase(extension.extensionName);
	}

	return requiredExtensions.empty();
}




QueueFamilyIndices Graphics::FindQueueFamilies(VkPhysicalDevice physicalDevice) {

	auto windowLayer = Application::GetLayer<WindowLayer>();
	QueueFamilyIndices indices;

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			indices.m_graphicsFamily = i;
		}
		VkBool32 presentSupport = false;
		if (windowLayer) {
			vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, m_vkSurface, &presentSupport);
			if (presentSupport) {
				indices.m_presentFamily = i;
			}
		}
		if (indices.IsComplete()) {
			break;
		}
		i++;
	}

	return indices;
}

SwapChainSupportDetails Graphics::QuerySwapChainSupport(VkPhysicalDevice physicalDevice) {
	SwapChainSupportDetails details;

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, m_vkSurface, &details.m_capabilities);

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, m_vkSurface, &formatCount, nullptr);

	if (formatCount != 0) {
		details.m_formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, m_vkSurface, &formatCount, details.m_formats.data());
	}

	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, m_vkSurface, &presentModeCount, nullptr);

	if (presentModeCount != 0) {
		details.m_presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, m_vkSurface, &presentModeCount, details.m_presentModes.data());
	}

	return details;
}

bool Graphics::IsDeviceSuitable(VkPhysicalDevice physicalDevice, const std::vector<std::string>& requiredDeviceExtensions)
{
	if (!CheckDeviceExtensionSupport(physicalDevice, requiredDeviceExtensions)) return false;
	const auto windowLayer = Application::GetLayer<WindowLayer>();
	if (windowLayer) {
		const auto swapChainSupportDetails = QuerySwapChainSupport(physicalDevice);
		if (swapChainSupportDetails.m_formats.empty() || swapChainSupportDetails.m_presentModes.empty()) return false;
	}
	return true;
}

void Graphics::CreateInstance()
{
	auto applicationInfo = Application::GetApplicationInfo();
	const auto windowLayer = Application::GetLayer<WindowLayer>();
	if (windowLayer) {
#pragma region Windows
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		int size;
		const auto monitors = glfwGetMonitors(&size);
		for (auto i = 0; i < size; i++)
		{
			windowLayer->m_monitors.push_back(monitors[i]);
		}
		windowLayer->m_primaryMonitor = glfwGetPrimaryMonitor();
		glfwSetMonitorCallback(windowLayer->SetMonitorCallback);
		const auto& applicationInfo = Application::GetApplicationInfo();
		windowLayer->m_windowSize = applicationInfo.m_defaultWindowSize;
		windowLayer->m_window = glfwCreateWindow(windowLayer->m_windowSize.x, windowLayer->m_windowSize.y, applicationInfo.m_applicationName.c_str(), nullptr, nullptr);

		if (applicationInfo.m_fullScreen)
			glfwMaximizeWindow(windowLayer->m_window);
		glfwSetFramebufferSizeCallback(windowLayer->m_window, windowLayer->FramebufferResizeCallback);
		glfwSetWindowFocusCallback(windowLayer->m_window, windowLayer->WindowFocusCallback);
		if (windowLayer->m_window == nullptr)
		{
			EVOENGINE_ERROR("Failed to create a window");
		}
#pragma endregion
		m_requiredDeviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	}
	m_requiredLayers = { "VK_LAYER_KHRONOS_validation" };
	std::vector<const char*> cRequiredDeviceExtensions;
	for (const auto& i : m_requiredDeviceExtensions) cRequiredDeviceExtensions.emplace_back(i.c_str());
	std::vector<const char*> cRequiredLayers;
	for (const auto& i : m_requiredLayers) cRequiredLayers.emplace_back(i.c_str());

	VkApplicationInfo vkApplicationInfo{};
	vkApplicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	vkApplicationInfo.pApplicationName = Application::GetApplicationInfo().m_applicationName.c_str();
	vkApplicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	vkApplicationInfo.pEngineName = "EvoEngine";
	vkApplicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	vkApplicationInfo.apiVersion = volkGetInstanceVersion();

#pragma region Instance
	VkInstanceCreateInfo instanceCreateInfo{};
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pApplicationInfo = &vkApplicationInfo;
	instanceCreateInfo.enabledLayerCount = 0;
	instanceCreateInfo.pNext = nullptr;
#pragma region Extensions
	uint32_t extensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
	m_vkExtensions.resize(extensionCount);
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, m_vkExtensions.data());
	std::vector<const char*> requiredExtensions;
	if (windowLayer) {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		for (uint32_t i = 0; i < glfwExtensionCount; i++) {

			requiredExtensions.emplace_back(glfwExtensions[i]);
		}
	}
	//For MacOS
	instanceCreateInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
	requiredExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#ifndef NDEBUG
	requiredExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
	instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
	instanceCreateInfo.ppEnabledExtensionNames = requiredExtensions.data();


#pragma endregion
#pragma region Layer
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
	m_vkLayers.resize(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, m_vkLayers.data());
#ifndef NDEBUG
	if (!CheckLayerSupport("VK_LAYER_KHRONOS_validation"))
	{
		throw std::runtime_error("Validation layers requested, but not available!");
	}



	instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(m_requiredLayers.size());
	instanceCreateInfo.ppEnabledLayerNames = cRequiredLayers.data();

	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
	PopulateDebugMessengerCreateInfo(debugCreateInfo);
	instanceCreateInfo.pNext = &debugCreateInfo;
#endif

#pragma endregion
	if (vkCreateInstance(&instanceCreateInfo, nullptr, &m_vkInstance) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create instance!");
	}

	//Let volk connect with vulkan
	volkLoadInstance(m_vkInstance);
#pragma endregion
}

void Graphics::CreateSurface()
{
	auto windowLayer = Application::GetLayer<WindowLayer>();
#pragma region Surface
	if (windowLayer) {


		if (glfwCreateWindowSurface(m_vkInstance, windowLayer->m_window, nullptr, &m_vkSurface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}
#pragma endregion
}

void Graphics::CreateDebugMessenger()
{
#pragma region Debug Messenger
#ifndef NDEBUG
	VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo{};
	PopulateDebugMessengerCreateInfo(debugUtilsMessengerCreateInfo);
	if (CreateDebugUtilsMessengerEXT(m_vkInstance, &debugUtilsMessengerCreateInfo, nullptr, &m_vkDebugMessenger) != VK_SUCCESS) {
		throw std::runtime_error("Failed to set up debug messenger!");
	}
#endif

#pragma endregion
}
int RateDeviceSuitability(VkPhysicalDevice physicalDevice) {
	int score = 0;
	VkPhysicalDeviceProperties deviceProperties;
	VkPhysicalDeviceFeatures deviceFeatures;
	vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
	vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
	// Discrete GPUs have a significant performance advantage
	if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
		score += 1000;
	}

	// Maximum possible size of textures affects graphics quality
	score += deviceProperties.limits.maxImageDimension2D;

	// Application can't function without geometry shaders
	if (!deviceFeatures.geometryShader) {
		return 0;
	}


	return score;
}
void Graphics::CreatePhysicalDevice()
{
#pragma region Physical Device
	m_vkPhysicalDevice = VK_NULL_HANDLE;
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(m_vkInstance, &deviceCount, nullptr);
	if (deviceCount == 0) {
		throw std::runtime_error("Failed to find GPUs with Vulkan support!");
	}
	std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
	vkEnumeratePhysicalDevices(m_vkInstance, &deviceCount, physicalDevices.data());
	std::multimap<int, VkPhysicalDevice> candidates;

	for (const auto& physicalDevice : physicalDevices) {
		if (!IsDeviceSuitable(physicalDevice, m_requiredDeviceExtensions)) continue;
		int score = RateDeviceSuitability(physicalDevice);
		candidates.insert(std::make_pair(score, physicalDevice));
	}
	// Check if the best candidate is suitable at all
	if (candidates.rbegin()->first > 0) {
		m_vkPhysicalDevice = candidates.rbegin()->second;
		vkGetPhysicalDeviceFeatures(m_vkPhysicalDevice, &m_vkPhysicalDeviceFeatures);
		vkGetPhysicalDeviceProperties(m_vkPhysicalDevice, &m_vkPhysicalDeviceProperties);
		EVOENGINE_LOG("Chose \"" + std::string(m_vkPhysicalDeviceProperties.deviceName) + "\" as physical device.");
	}
	else {
		throw std::runtime_error("failed to find a suitable GPU!");
	}
#pragma endregion
}

void Graphics::CreateLogicalDevice()
{
	std::vector<const char*> cRequiredDeviceExtensions;
	for (const auto& i : m_requiredDeviceExtensions) cRequiredDeviceExtensions.emplace_back(i.c_str());
	std::vector<const char*> cRequiredLayers;
	for (const auto& i : m_requiredLayers) cRequiredLayers.emplace_back(i.c_str());

#pragma region Logical Device
	VkPhysicalDeviceFeatures deviceFeatures{};
	VkDeviceCreateInfo deviceCreateInfo{};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
#pragma region Queues requirement
	m_queueFamilyIndices = FindQueueFamilies(m_vkPhysicalDevice);
	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies;
	if (m_queueFamilyIndices.m_graphicsFamily.has_value()) uniqueQueueFamilies.emplace(m_queueFamilyIndices.m_graphicsFamily.value());
	if (m_queueFamilyIndices.m_presentFamily.has_value()) uniqueQueueFamilies.emplace(m_queueFamilyIndices.m_presentFamily.value());
	float queuePriority = 1.0f;
	for (uint32_t queueFamily : uniqueQueueFamilies) {
		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}
	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
#pragma endregion
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(m_requiredDeviceExtensions.size());
	deviceCreateInfo.ppEnabledExtensionNames = cRequiredDeviceExtensions.data();

#ifndef NDEBUG
	deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(m_requiredLayers.size());
	deviceCreateInfo.ppEnabledLayerNames = cRequiredLayers.data();
#else
	deviceCreateInfo.enabledLayerCount = 0;
#endif
	if (vkCreateDevice(m_vkPhysicalDevice, &deviceCreateInfo, nullptr, &m_vkDevice) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create logical device!");
	}

	if (m_queueFamilyIndices.m_graphicsFamily.has_value()) vkGetDeviceQueue(m_vkDevice, m_queueFamilyIndices.m_graphicsFamily.value(), 0, &m_vkGraphicsQueue);
	if (m_queueFamilyIndices.m_presentFamily.has_value()) vkGetDeviceQueue(m_vkDevice, m_queueFamilyIndices.m_presentFamily.value(), 0, &m_vkPresentQueue);

#pragma endregion
}

void Graphics::SetupVmaAllocator()
{
#pragma region VMA
	VmaVulkanFunctions vulkanFunctions{};
	vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
	vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo vmaAllocatorCreateInfo{};
	vmaAllocatorCreateInfo.physicalDevice = m_vkPhysicalDevice;
	vmaAllocatorCreateInfo.device = m_vkDevice;
	vmaAllocatorCreateInfo.instance = m_vkInstance;
	vmaAllocatorCreateInfo.vulkanApiVersion = volkGetInstanceVersion();
	vmaAllocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;
	vmaCreateAllocator(&vmaAllocatorCreateInfo, &m_vmaAllocator);
#pragma endregion
}

void Graphics::CreateCommandPool()
{
#pragma region Command pool
	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolInfo.queueFamilyIndex = m_queueFamilyIndices.m_graphicsFamily.value();
	m_commandPool.Create(poolInfo);
#pragma endregion
}

void Graphics::CreateCommandBuffers()
{
#pragma region Command buffers
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = m_commandPool.GetVkCommandPool();
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = static_cast<uint32_t>(m_maxFrameInFlight);
	m_vkCommandBuffers.resize(m_maxFrameInFlight);

	if (vkAllocateCommandBuffers(m_vkDevice, &allocInfo, m_vkCommandBuffers.data()) != VK_SUCCESS) {
		throw std::runtime_error("Failed to allocate command buffers!");
	}
#pragma endregion
}

void Graphics::CreateSwapChainSyncObjects()
{
#pragma region Sync Objects
	VkSemaphoreCreateInfo semaphoreCreateInfo{};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	VkFenceCreateInfo fenceCreateInfo{};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
	m_renderFinishedSemaphores.resize(m_maxFrameInFlight);
	m_inFlightFences.resize(m_maxFrameInFlight);
	for (int i = 0; i < m_maxFrameInFlight; i++) {
		m_renderFinishedSemaphores[i].Create(semaphoreCreateInfo);
		m_inFlightFences[i].Create(fenceCreateInfo);
	}
#pragma endregion
}

void Graphics::CreateSwapChain()
{
	auto applicationInfo = Application::GetApplicationInfo();
	auto windowLayer = Application::GetLayer<WindowLayer>();
	SwapChainSupportDetails swapChainSupportDetails = QuerySwapChainSupport(m_vkPhysicalDevice);

	VkSurfaceFormatKHR surfaceFormat = swapChainSupportDetails.m_formats[0];
	for (const auto& availableFormat : swapChainSupportDetails.m_formats) {
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			surfaceFormat = availableFormat;
			break;
		}
	}
	VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
	for (const auto& availablePresentMode : swapChainSupportDetails.m_presentModes) {
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
			presentMode = availablePresentMode;
			break;
		}
	}

	VkExtent2D extent = m_swapchain.GetVkExtent2D();
	if (swapChainSupportDetails.m_capabilities.currentExtent.width != 0
		&& swapChainSupportDetails.m_capabilities.currentExtent.height != 0
		&& swapChainSupportDetails.m_capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		extent = swapChainSupportDetails.m_capabilities.currentExtent;
	}
	else {
		int width, height;
		if (windowLayer) glfwGetFramebufferSize(windowLayer->m_window, &width, &height);
		else
		{
			width = applicationInfo.m_defaultWindowSize.x;
			height = applicationInfo.m_defaultWindowSize.y;
		}
		if (width > 0 && height > 0) {
			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, swapChainSupportDetails.m_capabilities.minImageExtent.width, swapChainSupportDetails.m_capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, swapChainSupportDetails.m_capabilities.minImageExtent.height, swapChainSupportDetails.m_capabilities.maxImageExtent.height);
			extent = actualExtent;
		}
	}

	uint32_t imageCount = swapChainSupportDetails.m_capabilities.minImageCount + 1;
	if (swapChainSupportDetails.m_capabilities.maxImageCount > 0 && imageCount > swapChainSupportDetails.m_capabilities.maxImageCount) {
		imageCount = swapChainSupportDetails.m_capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR swapchainCreateInfo{};
	swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchainCreateInfo.surface = m_vkSurface;

	swapchainCreateInfo.minImageCount = imageCount;
	swapchainCreateInfo.imageFormat = surfaceFormat.format;
	swapchainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
	swapchainCreateInfo.imageExtent = extent;
	swapchainCreateInfo.imageArrayLayers = 1;
	/*
	 * It is also possible that you'll render images to a separate image first to perform operations like post-processing.
	 * In that case you may use a value like VK_IMAGE_USAGE_TRANSFER_DST_BIT instead and use a memory operation to transfer the rendered image to a swap chain image.
	 */
	swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	uint32_t queueFamilyIndices[] = { m_queueFamilyIndices.m_graphicsFamily.value(), m_queueFamilyIndices.m_presentFamily.value() };

	if (m_queueFamilyIndices.m_graphicsFamily != m_queueFamilyIndices.m_presentFamily) {
		swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		swapchainCreateInfo.queueFamilyIndexCount = 2;
		swapchainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else {
		swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	swapchainCreateInfo.preTransform = swapChainSupportDetails.m_capabilities.currentTransform;
	swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swapchainCreateInfo.presentMode = presentMode;
	swapchainCreateInfo.clipped = VK_TRUE;

	swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;
	if (extent.width == 0)
	{
		EVOENGINE_ERROR("WRONG")
	}
	m_swapchain.Create(swapchainCreateInfo);

	VkSemaphoreCreateInfo semaphoreCreateInfo{};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	m_imageAvailableSemaphores.resize(m_maxFrameInFlight);
	for (int i = 0; i < m_maxFrameInFlight; i++) {
		m_imageAvailableSemaphores[i].Create(semaphoreCreateInfo);
	}

	m_swapchainVersion++;
}

void Graphics::RecreateSwapChain()
{
	vkDeviceWaitIdle(m_vkDevice);
	CreateSwapChain();
	
}

void Graphics::OnDestroy()
{
	vkDeviceWaitIdle(m_vkDevice);

#pragma region Vulkan
	for (int i = 0; i < m_maxFrameInFlight; i++) {
		m_inFlightFences[i].Destroy();
		m_renderFinishedSemaphores[i].Destroy();
		m_imageAvailableSemaphores[i].Destroy();
	}
	m_commandPool.Destroy();
	m_swapchain.Destroy();
	vkDestroyDevice(m_vkDevice, nullptr);
#pragma region Debug Messenger
#ifndef NDEBUG
	DestroyDebugUtilsMessengerEXT(m_vkInstance, m_vkDebugMessenger, nullptr);
#endif
#pragma endregion
#pragma region Surface
	vkDestroySurfaceKHR(m_vkInstance, m_vkSurface, nullptr);
#pragma endregion
	vmaDestroyAllocator(m_vmaAllocator);
	vkDestroyInstance(m_vkInstance, nullptr);
#pragma endregion
}

void Graphics::SwapChainSwapImage()
{
	const auto& windowLayer = Application::GetLayer<WindowLayer>();
	if (windowLayer && (windowLayer->m_windowSize.x == 0 || windowLayer->m_windowSize.y == 0)) return;
	if (!m_queueFamilyIndices.m_presentFamily.has_value()) return;
	const VkFence inFlightFences[] = { m_inFlightFences[m_currentFrameIndex].GetVkFence() };
	vkWaitForFences(m_vkDevice, 1, inFlightFences,
		VK_TRUE, UINT64_MAX);
	auto result = vkAcquireNextImageKHR(m_vkDevice,
		m_swapchain.GetVkSwapchain(), UINT64_MAX,
		m_imageAvailableSemaphores[m_currentFrameIndex].GetVkSemaphore(),
		VK_NULL_HANDLE, &m_nextImageIndex);
	while (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_recreateSwapChain) {
		RecreateSwapChain();
		result = vkAcquireNextImageKHR(m_vkDevice,
			m_swapchain.GetVkSwapchain(), UINT64_MAX,
			m_imageAvailableSemaphores[m_currentFrameIndex].GetVkSemaphore(),
			VK_NULL_HANDLE, &m_nextImageIndex);
		m_recreateSwapChain = false;
	}
	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to acquire swap chain image!");
	}
	vkResetFences(m_vkDevice, 1, inFlightFences);
	vkResetCommandBuffer(m_vkCommandBuffers[m_currentFrameIndex], 0);
}

void Graphics::SubmitPresent()
{
	const auto& windowLayer = Application::GetLayer<WindowLayer>();
	if (windowLayer && (windowLayer->m_windowSize.x == 0 || windowLayer->m_windowSize.y == 0)) return;
	if (!m_queueFamilyIndices.m_presentFamily.has_value()) return;
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	VkSemaphore waitSemaphores[] = { m_imageAvailableSemaphores[m_currentFrameIndex].GetVkSemaphore() };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &m_vkCommandBuffers[m_currentFrameIndex];

	VkSemaphore signalSemaphores[] = { m_renderFinishedSemaphores[m_currentFrameIndex].GetVkSemaphore() };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;
	if (vkQueueSubmit(m_vkGraphicsQueue, 1, &submitInfo, m_inFlightFences[m_currentFrameIndex].GetVkFence()) != VK_SUCCESS) {
		throw std::runtime_error("failed to submit draw command buffer!");
	}

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	VkSwapchainKHR swapChains[] = { m_swapchain.GetVkSwapchain() };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;

	presentInfo.pImageIndices = &m_nextImageIndex;

	vkQueuePresentKHR(m_vkPresentQueue, &presentInfo);

	m_currentFrameIndex = (m_currentFrameIndex + 1) % m_maxFrameInFlight;
}


#pragma endregion




void Graphics::Initialize()
{
	auto& graphics = GetInstance();
#pragma region volk
	if (volkInitialize() != VK_SUCCESS)
	{
		throw std::runtime_error("Volk failed to initialize!");
	}
#pragma endregion
#pragma region Vulkan
	graphics.CreateInstance();
	graphics.CreateSurface();
	graphics.CreateDebugMessenger();
	graphics.CreatePhysicalDevice();
	graphics.CreateLogicalDevice();
	graphics.SetupVmaAllocator();

	if (graphics.m_queueFamilyIndices.m_presentFamily.has_value()) {
		graphics.CreateSwapChain();
		const auto swapChainSupportDetails = graphics.QuerySwapChainSupport(graphics.m_vkPhysicalDevice);
		graphics.m_vkSurfaceFormat = swapChainSupportDetails.m_formats[0];
		for (const auto& availableFormat : swapChainSupportDetails.m_formats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				graphics.m_vkSurfaceFormat = availableFormat;
				break;
			}
		}
	}
	if (graphics.m_queueFamilyIndices.m_graphicsFamily.has_value()) {
		graphics.CreateCommandPool();
		graphics.CreateCommandBuffers();
		graphics.CreateSwapChainSyncObjects();
	}
#pragma endregion

	
}

void Graphics::Destroy()
{
	auto& graphics = GetInstance();
	graphics.Destroy();

}

void Graphics::PreUpdate()
{
	auto& graphics = GetInstance();
	graphics.SwapChainSwapImage();
}

void Graphics::LateUpdate()
{
	auto& graphics = GetInstance();
	graphics.SubmitPresent();
}

bool Graphics::CheckExtensionSupport(const std::string& extensionName)
{
	const auto& graphics = GetInstance();
	for (const auto& layerProperties : graphics.m_vkLayers) {
		if (strcmp(extensionName.c_str(), layerProperties.layerName) == 0) {
			return true;
		}
	}
	return false;
}

bool Graphics::CheckLayerSupport(const std::string& layerName)
{
	const auto& graphics = GetInstance();
	for (const auto& layerProperties : graphics.m_vkLayers) {
		if (strcmp(layerName.c_str(), layerProperties.layerName) == 0) {
			return true;
		}
	}
	return false;
}

