#include "Graphics.hpp"
#include "Console.hpp"
#include "Application.hpp"
#include "Utilities.hpp"
#include "WindowLayer.hpp"
#include "Shader.hpp"
#include "Mesh.hpp"
#include "ProjectManager.hpp"
#include "Vertex.hpp"
#include "vk_mem_alloc.h"
#include "EditorLayer.hpp"
#include "GeometryStorage.hpp"
#include "RenderLayer.hpp"
#include "TextureStorage.hpp"
using namespace EvoEngine;
VkBool32 DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData)
{
	if (messageSeverity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) return VK_FALSE;
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
#ifndef NDEBUG
		throw std::runtime_error("Graphics Error!");
#endif
	}
	break;
	}
	msg += std::string(pCallbackData->pMessage);
	EVOENGINE_LOG(msg);
	return VK_FALSE;
}
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
void SelectStageFlagsAccessMask(const VkImageLayout imageLayout, VkAccessFlags& mask, VkPipelineStageFlags& stageFlags)
{
	switch (imageLayout)
	{
	case VK_IMAGE_LAYOUT_UNDEFINED:
	{
		mask = 0;
		stageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	}break;
	case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
	{
		mask = VK_ACCESS_TRANSFER_WRITE_BIT;
		stageFlags = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}break;
	case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
	{
		mask = VK_ACCESS_SHADER_READ_BIT;
		stageFlags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}break;
	case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
	{
		mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		stageFlags = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	}break;
	case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
	{
		mask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		stageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	}break;
	case VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL:
	{
		mask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		stageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	}break;
	case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
	{
		mask = 0;
		stageFlags = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	}break;
	default:
	{
		mask = 0;
		stageFlags = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
	}break;
	}
}


void Graphics::RegisterGraphicsPipeline(const std::string& name, const std::shared_ptr<GraphicsPipeline>& graphicsPipeline)
{
	auto& graphics = GetInstance();
	if (graphics.m_graphicsPipelines.find(name) != graphics.m_graphicsPipelines.end())
	{
		EVOENGINE_ERROR("GraphicsPipeline with same name exists!");
		return;
	}
	graphics.m_graphicsPipelines[name] = graphicsPipeline;
}

const std::shared_ptr<GraphicsPipeline>& Graphics::GetGraphicsPipeline(const std::string& name)
{
	const auto& graphics = GetInstance();
	return graphics.m_graphicsPipelines.at(name);
}

const std::shared_ptr<DescriptorSetLayout>& Graphics::GetDescriptorSetLayout(const std::string& name)
{
	const auto& graphics = GetInstance();
	return graphics.m_descriptorSetLayouts.at(name);
}

void Graphics::RegisterDescriptorSetLayout(const std::string& name,
	const std::shared_ptr<DescriptorSetLayout>& descriptorSetLayout)
{
	auto& graphics = GetInstance();
	if (graphics.m_descriptorSetLayouts.find(name) != graphics.m_descriptorSetLayouts.end())
	{
		EVOENGINE_ERROR("GraphicsPipeline with same name exists!");
		return;
	}
	graphics.m_descriptorSetLayouts[name] = descriptorSetLayout;
}

void Graphics::TransitImageLayout(VkCommandBuffer commandBuffer, VkImage targetImage, VkFormat imageFormat, uint32_t layerCount, VkImageLayout oldLayout,
	VkImageLayout newLayout, uint32_t mipLevels)
{
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = targetImage;
	if (imageFormat == Constants::TEXTURE_2D || imageFormat == Constants::RENDER_TEXTURE_COLOR || imageFormat == Constants::G_BUFFER_COLOR)
	{
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	}
	else if (imageFormat == Constants::RENDER_TEXTURE_DEPTH || imageFormat == Constants::G_BUFFER_DEPTH || imageFormat == Constants::SHADOW_MAP)
	{
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	}
	else if (const auto windowLayer = Application::GetLayer<WindowLayer>(); windowLayer && imageFormat == GetSwapchain()->GetImageFormat())
	{
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	}
	else
	{
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	}

	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = mipLevels;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = layerCount;

	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;

	SelectStageFlagsAccessMask(oldLayout, barrier.srcAccessMask, sourceStage);
	SelectStageFlagsAccessMask(newLayout, barrier.dstAccessMask, destinationStage);

	vkCmdPipelineBarrier(
		commandBuffer,
		sourceStage, destinationStage,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);
}

size_t Graphics::GetMaxBoneAmount()
{
	const auto& graphics = GetInstance();
	return graphics.m_maxBoneAmount;
}

size_t Graphics::GetMaxShadowCascadeAmount()
{
	const auto& graphics = GetInstance();
	return graphics.m_maxShadowCascadeAmount;
}

void Graphics::ImmediateSubmit(const std::function<void(VkCommandBuffer commandBuffer)>& action)
{
	const auto& graphics = GetInstance();
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = graphics.m_commandPool->GetVkCommandPool();
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

	vkFreeCommandBuffers(graphics.m_vkDevice, graphics.m_commandPool->GetVkCommandPool(), 1, &commandBuffer);
}

QueueFamilyIndices Graphics::GetQueueFamilyIndices()
{
	const auto& graphics = GetInstance();
	return graphics.m_queueFamilyIndices;
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

VkInstance Graphics::GetVkInstance()
{
	const auto& graphics = GetInstance();
	return graphics.m_vkInstance;
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
	return graphics.m_commandPool->GetVkCommandPool();
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

const std::unique_ptr<Swapchain>& Graphics::GetSwapchain()
{
	const auto& graphics = GetInstance();
	return graphics.m_swapchain;
}

const std::unique_ptr<DescriptorPool>& Graphics::GetDescriptorPool()
{
	const auto& graphics = GetInstance();
	return graphics.m_descriptorPool;
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

const VkPhysicalDeviceProperties& Graphics::GetVkPhysicalDeviceProperties()
{
	const auto& graphics = GetInstance();
	return graphics.m_vkPhysicalDeviceProperties;
}



VkResult CreateDebugUtilsMessengerExt(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	if (const auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT")); func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerExt(const VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	if (const auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
		vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT")); func != nullptr) {
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



QueueFamilyIndices Graphics::FindQueueFamilies(const VkPhysicalDevice physicalDevice) const
{
	const auto windowLayer = Application::GetLayer<WindowLayer>();
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

SwapChainSupportDetails Graphics::QuerySwapChainSupport(const VkPhysicalDevice physicalDevice) const
{
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

bool Graphics::IsDeviceSuitable(VkPhysicalDevice physicalDevice, const std::vector<std::string>& requiredDeviceExtensions) const
{
	if (!CheckDeviceExtensionSupport(physicalDevice, requiredDeviceExtensions)) return false;
	if (const auto windowLayer = Application::GetLayer<WindowLayer>()) {
		const auto swapChainSupportDetails = QuerySwapChainSupport(physicalDevice);
		if (swapChainSupportDetails.m_formats.empty() || swapChainSupportDetails.m_presentModes.empty()) return false;
	}

	return true;
}

void Graphics::CreateInstance()
{
	auto applicationInfo = Application::GetApplicationInfo();
	const auto windowLayer = Application::GetLayer<WindowLayer>();
	const auto editorLayer = Application::GetLayer<EditorLayer>();
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
		if (editorLayer) windowLayer->m_windowSize = { 250, 50 };
		windowLayer->m_window = glfwCreateWindow(windowLayer->m_windowSize.x, windowLayer->m_windowSize.y, applicationInfo.m_applicationName.c_str(), nullptr, nullptr);

		if (applicationInfo.m_fullScreen)
			glfwMaximizeWindow(windowLayer->m_window);

		glfwSetFramebufferSizeCallback(windowLayer->m_window, windowLayer->FramebufferSizeCallback);
		glfwSetWindowFocusCallback(windowLayer->m_window, windowLayer->WindowFocusCallback);
		glfwSetKeyCallback(windowLayer->m_window, Input::KeyCallBack);
		glfwSetMouseButtonCallback(windowLayer->m_window, Input::MouseButtonCallBack);

		if (windowLayer->m_window == nullptr)
		{
			EVOENGINE_ERROR("Failed to create a window");
		}
#pragma endregion
		m_requiredDeviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	}

	m_requiredLayers = { "VK_LAYER_KHRONOS_validation" };
	std::vector<const char*> cRequiredLayers;
	cRequiredLayers.reserve(m_requiredLayers.size());
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
		const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		for (uint32_t i = 0; i < glfwExtensionCount; i++) {

			requiredExtensions.emplace_back(glfwExtensions[i]);
		}
	}
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
	const auto windowLayer = Application::GetLayer<WindowLayer>();
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
	if (CreateDebugUtilsMessengerExt(m_vkInstance, &debugUtilsMessengerCreateInfo, nullptr, &m_vkDebugMessenger) != VK_SUCCESS) {
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
	if (!deviceFeatures.geometryShader) return 0;
	if (!deviceFeatures.samplerAnisotropy) return 0;

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
	m_requiredDeviceExtensions.emplace_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
	m_requiredDeviceExtensions.emplace_back(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
	m_requiredDeviceExtensions.emplace_back(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
	m_requiredDeviceExtensions.emplace_back(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME);
	m_requiredDeviceExtensions.emplace_back(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);

	std::vector<const char*> cRequiredDeviceExtensions;
	cRequiredDeviceExtensions.reserve(m_requiredDeviceExtensions.size());
	for (const auto& i : m_requiredDeviceExtensions) cRequiredDeviceExtensions.emplace_back(i.c_str());
	std::vector<const char*> cRequiredLayers;
	cRequiredLayers.reserve(m_requiredLayers.size());
	for (const auto& i : m_requiredLayers) cRequiredLayers.emplace_back(i.c_str());

#pragma region Logical Device

	VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeaturesExt{};
	meshShaderFeaturesExt.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
	meshShaderFeaturesExt.pNext = nullptr;
	meshShaderFeaturesExt.meshShader = VK_TRUE;
	meshShaderFeaturesExt.taskShader = VK_TRUE;

	VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamicRenderingFeatures{};
	dynamicRenderingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
	dynamicRenderingFeatures.dynamicRendering = VK_TRUE;
	dynamicRenderingFeatures.pNext = &meshShaderFeaturesExt;

	VkPhysicalDeviceSynchronization2Features physicalDeviceSynchronization2Features{};
	physicalDeviceSynchronization2Features.synchronization2 = VK_TRUE;
	physicalDeviceSynchronization2Features.pNext = &dynamicRenderingFeatures;
	physicalDeviceSynchronization2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;

	VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT extendedVertexInputDynamicStateFeatures{};
	extendedVertexInputDynamicStateFeatures.vertexInputDynamicState = VK_TRUE;
	extendedVertexInputDynamicStateFeatures.pNext = &physicalDeviceSynchronization2Features;
	extendedVertexInputDynamicStateFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;

	VkPhysicalDeviceExtendedDynamicState3FeaturesEXT extendedDynamicState3Features{};
	extendedDynamicState3Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
	extendedDynamicState3Features.extendedDynamicState3PolygonMode = VK_TRUE;
	extendedDynamicState3Features.extendedDynamicState3DepthClampEnable = VK_TRUE;
	extendedDynamicState3Features.extendedDynamicState3ColorBlendEnable = VK_TRUE;
	extendedDynamicState3Features.extendedDynamicState3LogicOpEnable = VK_TRUE;
	extendedDynamicState3Features.extendedDynamicState3ColorBlendEquation = VK_TRUE;
	extendedDynamicState3Features.extendedDynamicState3ColorWriteMask = VK_TRUE;

	extendedDynamicState3Features.pNext = &extendedVertexInputDynamicStateFeatures;

	VkPhysicalDeviceExtendedDynamicState2FeaturesEXT extendedDynamicState2Features{};
	extendedDynamicState2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT;
	extendedDynamicState2Features.extendedDynamicState2 = VK_TRUE;
	extendedDynamicState2Features.extendedDynamicState2PatchControlPoints = VK_TRUE;
	extendedDynamicState2Features.extendedDynamicState2LogicOp = VK_TRUE;
	extendedDynamicState2Features.pNext = &extendedDynamicState3Features;

	VkPhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicStateFeatures{};
	extendedDynamicStateFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT;
	extendedDynamicStateFeatures.extendedDynamicState = VK_TRUE;
	extendedDynamicStateFeatures.pNext = &extendedDynamicState2Features;

	VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeatures{};
	descriptorIndexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
	descriptorIndexingFeatures.pNext = &extendedDynamicStateFeatures;
	descriptorIndexingFeatures.descriptorBindingPartiallyBound = VK_TRUE;
	descriptorIndexingFeatures.runtimeDescriptorArray = VK_TRUE;

	VkPhysicalDeviceSeparateDepthStencilLayoutsFeatures separateDepthStencilLayoutsFeatures{};
	separateDepthStencilLayoutsFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES;
	separateDepthStencilLayoutsFeatures.pNext = &descriptorIndexingFeatures;
	separateDepthStencilLayoutsFeatures.separateDepthStencilLayouts = VK_TRUE;

	VkPhysicalDeviceFeatures2 deviceFeatures2{};
	deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	deviceFeatures2.pNext = &descriptorIndexingFeatures;
	vkGetPhysicalDeviceFeatures2(m_vkPhysicalDevice, &deviceFeatures2);

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
	deviceCreateInfo.pEnabledFeatures = nullptr;
	deviceCreateInfo.pNext = &deviceFeatures2;

	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(m_requiredDeviceExtensions.size());
	deviceCreateInfo.ppEnabledExtensionNames = cRequiredDeviceExtensions.data();

	deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(m_requiredLayers.size());
	deviceCreateInfo.ppEnabledLayerNames = cRequiredLayers.data();
	if (vkCreateDevice(m_vkPhysicalDevice, &deviceCreateInfo, nullptr, &m_vkDevice) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create logical device!");
	}

	if (m_queueFamilyIndices.m_graphicsFamily.has_value()) vkGetDeviceQueue(m_vkDevice, m_queueFamilyIndices.m_graphicsFamily.value(), 0, &m_vkGraphicsQueue);
	if (m_queueFamilyIndices.m_presentFamily.has_value()) vkGetDeviceQueue(m_vkDevice, m_queueFamilyIndices.m_presentFamily.value(), 0, &m_vkPresentQueue);

#pragma endregion
}


void Graphics::EverythingBarrier(VkCommandBuffer commandBuffer)
{
	VkMemoryBarrier2 memoryBarrier{};
	memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
	memoryBarrier.srcStageMask = memoryBarrier.dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	memoryBarrier.srcAccessMask = memoryBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
	VkDependencyInfo dependencyInfo{};
	dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependencyInfo.memoryBarrierCount = 1;
	dependencyInfo.pMemoryBarriers = &memoryBarrier;

	vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
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


std::string Graphics::StringifyResultVk(const VkResult& result)
{
	switch (result)
	{
	case VK_SUCCESS:
		return "Success";
	case VK_NOT_READY:
		return "A fence or query has not yet completed";
	case VK_TIMEOUT:
		return "A wait operation has not completed in the specified time";
	case VK_EVENT_SET:
		return "An event is signaled";
	case VK_EVENT_RESET:
		return "An event is not signaled";
	case VK_INCOMPLETE:
		return "A return array was too small for the result";
	case VK_ERROR_OUT_OF_HOST_MEMORY:
		return "A host memory allocation has failed";
	case VK_ERROR_OUT_OF_DEVICE_MEMORY:
		return "A device memory allocation has failed";
	case VK_ERROR_INITIALIZATION_FAILED:
		return "Initialization of an object could not be completed for implementation-specific reasons";
	case VK_ERROR_DEVICE_LOST:
		return "The logical or physical device has been lost";
	case VK_ERROR_MEMORY_MAP_FAILED:
		return "Mapping of a memory object has failed";
	case VK_ERROR_LAYER_NOT_PRESENT:
		return "A requested layer is not present or could not be loaded";
	case VK_ERROR_EXTENSION_NOT_PRESENT:
		return "A requested extension is not supported";
	case VK_ERROR_FEATURE_NOT_PRESENT:
		return "A requested feature is not supported";
	case VK_ERROR_INCOMPATIBLE_DRIVER:
		return "The requested version of Vulkan is not supported by the driver or is otherwise incompatible";
	case VK_ERROR_TOO_MANY_OBJECTS:
		return "Too many objects of the type have already been created";
	case VK_ERROR_FORMAT_NOT_SUPPORTED:
		return "A requested format is not supported on this device";
	case VK_ERROR_SURFACE_LOST_KHR:
		return "A surface is no longer available";
		//case VK_ERROR_OUT_OF_POOL_MEMORY:
		//return "A allocation failed due to having no more space in the descriptor pool";
	case VK_SUBOPTIMAL_KHR:
		return "A swapchain no longer matches the surface properties exactly, but can still be used";
	case VK_ERROR_OUT_OF_DATE_KHR:
		return "A surface has changed in such a way that it is no longer compatible with the swapchain";
	case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
		return "The display used by a swapchain does not use the same presentable image layout";
	case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
		return "The requested window is already connected to a VkSurfaceKHR, or to some other non-Vulkan API";
	case VK_ERROR_VALIDATION_FAILED_EXT:
		return "A validation layer found an error";
	default:
		return "Unknown Vulkan error";
	}

}

void Graphics::CheckVk(const VkResult& result)
{
	if (result >= 0)
	{
		return;
	}
	const std::string failure = StringifyResultVk(result);
	throw std::runtime_error("Vulkan error: " + failure);
}

void Graphics::AppendCommands(const std::function<void(VkCommandBuffer commandBuffer)>& action)
{
	auto& graphics = GetInstance();
	const unsigned commandBufferIndex = graphics.m_usedCommandBufferSize;
	if (commandBufferIndex >= graphics.m_commandBufferPool[graphics.m_currentFrameIndex].size())
	{
		graphics.m_commandBufferPool[graphics.m_currentFrameIndex].emplace_back();
		graphics.m_commandBufferPool[graphics.m_currentFrameIndex].back().Allocate();
	}
	auto& commandBuffer = graphics.m_commandBufferPool[graphics.m_currentFrameIndex][commandBufferIndex];
	commandBuffer.Begin();
	GraphicsPipelineStates graphicsGlobalState{};
	action(commandBuffer.GetVkCommandBuffer());
	commandBuffer.End();
	graphics.m_usedCommandBufferSize++;
}

void Graphics::CreateSwapChain()
{
	auto applicationInfo = Application::GetApplicationInfo();
	auto windowLayer = Application::GetLayer<WindowLayer>();
	SwapChainSupportDetails swapChainSupportDetails = QuerySwapChainSupport(m_vkPhysicalDevice);

	VkSurfaceFormatKHR surfaceFormat = swapChainSupportDetails.m_formats[0];
	for (const auto& availableFormat : swapChainSupportDetails.m_formats) {
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
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

	VkExtent2D extent = {};
	if (m_swapchain) extent = m_swapchain->GetImageExtent();
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

	if (m_swapchain)
	{
		swapchainCreateInfo.oldSwapchain = m_swapchain->GetVkSwapchain();
	}
	else
	{
		swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;
	}

	if (extent.width == 0)
	{
		EVOENGINE_ERROR("WRONG")
	}
	m_swapchain = std::make_unique<Swapchain>(swapchainCreateInfo);

	VkSemaphoreCreateInfo semaphoreCreateInfo{};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	m_imageAvailableSemaphores.clear();
	for (int i = 0; i < m_maxFrameInFlight; i++) {
		m_imageAvailableSemaphores.emplace_back(std::make_unique<Semaphore>(semaphoreCreateInfo));
	}

	m_swapchainVersion++;
}

void Graphics::CreateSwapChainSyncObjects()
{
	VkSemaphoreCreateInfo semaphoreCreateInfo{};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	VkFenceCreateInfo fenceCreateInfo{};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
	m_renderFinishedSemaphores.clear();
	m_inFlightFences.clear();
	for (int i = 0; i < m_maxFrameInFlight; i++) {
		m_renderFinishedSemaphores.emplace_back(std::make_unique<Semaphore>(semaphoreCreateInfo));
		m_inFlightFences.emplace_back(std::make_unique<Fence>(fenceCreateInfo));
	}
}


void Graphics::RecreateSwapChain()
{
	vkDeviceWaitIdle(m_vkDevice);
	CreateSwapChain();
}

void Graphics::OnDestroy()
{
	vkDeviceWaitIdle(m_vkDevice);



	m_descriptorPool.reset();

#pragma region Vulkan
	m_imageAvailableSemaphores.clear();
	m_commandPool.reset();
	m_swapchain.reset();

	vkDestroyDevice(m_vkDevice, nullptr);
#pragma region Debug Messenger
#ifndef NDEBUG
	DestroyDebugUtilsMessengerExt(m_vkInstance, m_vkDebugMessenger, nullptr);
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
	if (windowLayer && (windowLayer->m_windowSize.x == 0 || windowLayer->m_windowSize.y == 0))
	{
		ResetCommandBuffers();
		return;
	}
	vkDeviceWaitIdle(m_vkDevice);
	const VkFence inFlightFences[] = { m_inFlightFences[m_currentFrameIndex]->GetVkFence() };
	vkWaitForFences(m_vkDevice, 1, inFlightFences,
		VK_TRUE, UINT64_MAX);

	auto result = vkAcquireNextImageKHR(m_vkDevice,
		m_swapchain->GetVkSwapchain(), UINT64_MAX,
		m_imageAvailableSemaphores[m_currentFrameIndex]->GetVkSemaphore(),
		VK_NULL_HANDLE, &m_nextImageIndex);
	while (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_recreateSwapChain) {
		RecreateSwapChain();
		result = vkAcquireNextImageKHR(m_vkDevice,
			m_swapchain->GetVkSwapchain(), UINT64_MAX,
			m_imageAvailableSemaphores[m_currentFrameIndex]->GetVkSemaphore(),
			VK_NULL_HANDLE, &m_nextImageIndex);
		m_recreateSwapChain = false;
	}
	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to acquire swap chain image!");
	}
	vkResetFences(m_vkDevice, 1, inFlightFences);
}

void Graphics::SubmitPresent()
{
	const auto& windowLayer = Application::GetLayer<WindowLayer>();
	if (windowLayer && (windowLayer->m_windowSize.x == 0 || windowLayer->m_windowSize.y == 0)) {
		ResetCommandBuffers();
		return;
	}
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	const VkSemaphore waitSemaphores[] = { m_imageAvailableSemaphores[m_currentFrameIndex]->GetVkSemaphore() };
	const VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	std::vector<VkCommandBuffer> commandBuffers;

	commandBuffers.reserve(m_usedCommandBufferSize);
	for (int i = 0; i < m_usedCommandBufferSize; i++)
	{
		commandBuffers.emplace_back(m_commandBufferPool[m_currentFrameIndex][i].GetVkCommandBuffer());
	}

	submitInfo.commandBufferCount = commandBuffers.size();
	submitInfo.pCommandBuffers = commandBuffers.data();

	const VkSemaphore signalSemaphores[] = { m_renderFinishedSemaphores[m_currentFrameIndex]->GetVkSemaphore() };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;
	if (vkQueueSubmit(m_vkGraphicsQueue, 1, &submitInfo, m_inFlightFences[m_currentFrameIndex]->GetVkFence()) != VK_SUCCESS) {
		throw std::runtime_error("failed to submit draw command buffer!");
	}


	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	const VkSwapchainKHR swapChains[] = { m_swapchain->GetVkSwapchain() };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;

	presentInfo.pImageIndices = &m_nextImageIndex;

	vkQueuePresentKHR(m_vkPresentQueue, &presentInfo);

	m_currentFrameIndex = (m_currentFrameIndex + 1) % m_maxFrameInFlight;
}

void Graphics::WaitForCommandsComplete()
{
}

void Graphics::ResetCommandBuffers()
{
	m_usedCommandBufferSize = 0;
	for (auto& commandBuffer : m_commandBufferPool[m_currentFrameIndex])
	{
		if (commandBuffer.m_status == CommandBufferStatus::Recorded) commandBuffer.Reset();
	}
}


#pragma endregion

void Graphics::PrepareDescriptorSetLayouts() const
{
	const auto perFrameLayout = std::make_shared<DescriptorSetLayout>();
	perFrameLayout->PushDescriptorBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_MESH_BIT_EXT, 0);
	perFrameLayout->PushDescriptorBinding(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_MESH_BIT_EXT, 0);
	perFrameLayout->PushDescriptorBinding(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_MESH_BIT_EXT, 0);
	perFrameLayout->PushDescriptorBinding(13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, Constants::MAX_TEXTURE_2D_RESOURCE_SIZE);
	perFrameLayout->PushDescriptorBinding(14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, Constants::MAX_CUBEMAP_RESOURCE_SIZE);

	perFrameLayout->Initialize();
	RegisterDescriptorSetLayout("PER_FRAME_LAYOUT", perFrameLayout);

	const auto cameraGBufferLayout = std::make_shared<DescriptorSetLayout>();
	cameraGBufferLayout->PushDescriptorBinding(18, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	cameraGBufferLayout->PushDescriptorBinding(19, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	cameraGBufferLayout->PushDescriptorBinding(20, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	cameraGBufferLayout->PushDescriptorBinding(21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	cameraGBufferLayout->Initialize();
	RegisterDescriptorSetLayout("CAMERA_GBUFFER_LAYOUT", cameraGBufferLayout);

	const auto lightLayout = std::make_shared<DescriptorSetLayout>();
	lightLayout->PushDescriptorBinding(15, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	lightLayout->PushDescriptorBinding(16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	lightLayout->PushDescriptorBinding(17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	lightLayout->Initialize();
	RegisterDescriptorSetLayout("LIGHTING_LAYOUT", lightLayout);

	const auto renderTexturePresent = std::make_shared<DescriptorSetLayout>();
	renderTexturePresent->PushDescriptorBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	renderTexturePresent->Initialize();
	RegisterDescriptorSetLayout("RENDER_TEXTURE_PRESENT", renderTexturePresent);

	const auto boneMatricesLayout = std::make_shared<DescriptorSetLayout>();
	boneMatricesLayout->PushDescriptorBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
	boneMatricesLayout->Initialize();
	RegisterDescriptorSetLayout("BONE_MATRICES_LAYOUT", boneMatricesLayout);

	const auto instancedDataLayout = std::make_shared<DescriptorSetLayout>();
	instancedDataLayout->PushDescriptorBinding(18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	instancedDataLayout->Initialize();
	RegisterDescriptorSetLayout("INSTANCED_DATA_LAYOUT", instancedDataLayout);

}

void Graphics::CreateGraphicsPipelines() const
{
	auto perFrameLayout = GetDescriptorSetLayout("PER_FRAME_LAYOUT");
	auto cameraGBufferLayout = GetDescriptorSetLayout("CAMERA_GBUFFER_LAYOUT");
	auto lightingLayout = GetDescriptorSetLayout("LIGHTING_LAYOUT");

	auto boneMatricesLayout = GetDescriptorSetLayout("BONE_MATRICES_LAYOUT");
	auto instancedDataLayout = GetDescriptorSetLayout("INSTANCED_DATA_LAYOUT");

	if (const auto windowLayer = Application::GetLayer<WindowLayer>()) {
		const auto renderTexturePassThrough = std::make_shared<GraphicsPipeline>();
		renderTexturePassThrough->m_vertexShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
		renderTexturePassThrough->m_fragmentShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_FRAG");
		renderTexturePassThrough->m_geometryType = GeometryType::Mesh;
		renderTexturePassThrough->m_descriptorSetLayouts.emplace_back(GetDescriptorSetLayout("RENDER_TEXTURE_PRESENT"));

		renderTexturePassThrough->m_depthAttachmentFormat = VK_FORMAT_UNDEFINED;
		renderTexturePassThrough->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		renderTexturePassThrough->m_colorAttachmentFormats = { 1, m_swapchain->GetImageFormat() };

		renderTexturePassThrough->PreparePipeline();
		RegisterGraphicsPipeline("RENDER_TEXTURE_PRESENT", renderTexturePassThrough);
	}
	{
		const auto standardDeferredPrepass = std::make_shared<GraphicsPipeline>();
		standardDeferredPrepass->m_vertexShader = Resources::GetResource<Shader>("STANDARD_VERT");
		standardDeferredPrepass->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_FRAG");
		standardDeferredPrepass->m_geometryType = GeometryType::Mesh;
		standardDeferredPrepass->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		standardDeferredPrepass->m_depthAttachmentFormat = Constants::G_BUFFER_DEPTH;
		standardDeferredPrepass->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		standardDeferredPrepass->m_colorAttachmentFormats = { 3, Constants::G_BUFFER_COLOR };

		auto& pushConstantRange = standardDeferredPrepass->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		standardDeferredPrepass->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_DEFERRED_PREPASS", standardDeferredPrepass);
	}
	{
		const auto standardSkinnedDeferredPrepass = std::make_shared<GraphicsPipeline>();
		standardSkinnedDeferredPrepass->m_vertexShader = Resources::GetResource<Shader>("STANDARD_SKINNED_VERT");
		standardSkinnedDeferredPrepass->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_FRAG");
		standardSkinnedDeferredPrepass->m_geometryType = GeometryType::SkinnedMesh;
		standardSkinnedDeferredPrepass->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		standardSkinnedDeferredPrepass->m_descriptorSetLayouts.emplace_back(boneMatricesLayout);

		standardSkinnedDeferredPrepass->m_depthAttachmentFormat = Constants::G_BUFFER_DEPTH;
		standardSkinnedDeferredPrepass->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		standardSkinnedDeferredPrepass->m_colorAttachmentFormats = { 3, Constants::G_BUFFER_COLOR };

		auto& pushConstantRange = standardSkinnedDeferredPrepass->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
		standardSkinnedDeferredPrepass->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_SKINNED_DEFERRED_PREPASS", standardSkinnedDeferredPrepass);
	}
	{
		const auto standardInstancedDeferredPrepass = std::make_shared<GraphicsPipeline>();
		standardInstancedDeferredPrepass->m_vertexShader = Resources::GetResource<Shader>("STANDARD_INSTANCED_VERT");
		standardInstancedDeferredPrepass->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_FRAG");
		standardInstancedDeferredPrepass->m_geometryType = GeometryType::Mesh;
		standardInstancedDeferredPrepass->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		standardInstancedDeferredPrepass->m_descriptorSetLayouts.emplace_back(instancedDataLayout);

		standardInstancedDeferredPrepass->m_depthAttachmentFormat = Constants::G_BUFFER_DEPTH;
		standardInstancedDeferredPrepass->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		standardInstancedDeferredPrepass->m_colorAttachmentFormats = { 3, Constants::G_BUFFER_COLOR };

		auto& pushConstantRange = standardInstancedDeferredPrepass->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
		standardInstancedDeferredPrepass->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_INSTANCED_DEFERRED_PREPASS", standardInstancedDeferredPrepass);
	}
	{
		const auto standardStrandsDeferredPrepass = std::make_shared<GraphicsPipeline>();
		standardStrandsDeferredPrepass->m_vertexShader = Resources::GetResource<Shader>("STANDARD_STRANDS_VERT");
		standardStrandsDeferredPrepass->m_tessellationControlShader = Resources::GetResource<Shader>("STANDARD_STRANDS_TESC");
		standardStrandsDeferredPrepass->m_tessellationEvaluationShader = Resources::GetResource<Shader>("STANDARD_STRANDS_TESE");
		standardStrandsDeferredPrepass->m_geometryShader = Resources::GetResource<Shader>("STANDARD_STRANDS_GEOM");
		standardStrandsDeferredPrepass->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_FRAG");
		standardStrandsDeferredPrepass->m_geometryType = GeometryType::Strands;
		standardStrandsDeferredPrepass->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		standardStrandsDeferredPrepass->m_descriptorSetLayouts.emplace_back(instancedDataLayout);

		standardStrandsDeferredPrepass->m_depthAttachmentFormat = Constants::G_BUFFER_DEPTH;
		standardStrandsDeferredPrepass->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		standardStrandsDeferredPrepass->m_colorAttachmentFormats = { 3, Constants::G_BUFFER_COLOR };

		auto& pushConstantRange = standardStrandsDeferredPrepass->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
		standardStrandsDeferredPrepass->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_STRANDS_DEFERRED_PREPASS", standardStrandsDeferredPrepass);
	}
	{
		const auto standardDeferredLighting = std::make_shared<GraphicsPipeline>();
		standardDeferredLighting->m_vertexShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
		standardDeferredLighting->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_LIGHTING_FRAG");
		standardDeferredLighting->m_geometryType = GeometryType::Mesh;
		standardDeferredLighting->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		standardDeferredLighting->m_descriptorSetLayouts.emplace_back(cameraGBufferLayout);
		standardDeferredLighting->m_descriptorSetLayouts.emplace_back(lightingLayout);

		standardDeferredLighting->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		standardDeferredLighting->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		standardDeferredLighting->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };

		auto& pushConstantRange = standardDeferredLighting->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		standardDeferredLighting->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_DEFERRED_LIGHTING", standardDeferredLighting);
	}
	{
		const auto standardDeferredLightingSceneCamera = std::make_shared<GraphicsPipeline>();
		standardDeferredLightingSceneCamera->m_vertexShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
		standardDeferredLightingSceneCamera->m_fragmentShader = Resources::GetResource<Shader>("STANDARD_DEFERRED_LIGHTING_SCENE_CAMERA_FRAG");
		standardDeferredLightingSceneCamera->m_geometryType = GeometryType::Mesh;
		standardDeferredLightingSceneCamera->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		standardDeferredLightingSceneCamera->m_descriptorSetLayouts.emplace_back(cameraGBufferLayout);
		standardDeferredLightingSceneCamera->m_descriptorSetLayouts.emplace_back(lightingLayout);

		standardDeferredLightingSceneCamera->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		standardDeferredLightingSceneCamera->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		standardDeferredLightingSceneCamera->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };

		auto& pushConstantRange = standardDeferredLightingSceneCamera->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
		standardDeferredLightingSceneCamera->PreparePipeline();
		RegisterGraphicsPipeline("STANDARD_DEFERRED_LIGHTING_SCENE_CAMERA", standardDeferredLightingSceneCamera);
	}
	{
		const auto directionalLightShadowMap = std::make_shared<GraphicsPipeline>();
		directionalLightShadowMap->m_vertexShader = Resources::GetResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_VERT");
		directionalLightShadowMap->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		directionalLightShadowMap->m_geometryType = GeometryType::Mesh;
		directionalLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		directionalLightShadowMap->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		directionalLightShadowMap->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = directionalLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		directionalLightShadowMap->PreparePipeline();
		RegisterGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP", directionalLightShadowMap);
	}
	{
		const auto directionalLightShadowMapSkinned = std::make_shared<GraphicsPipeline>();
		directionalLightShadowMapSkinned->m_vertexShader = Resources::GetResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_SKINNED_VERT");
		directionalLightShadowMapSkinned->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		directionalLightShadowMapSkinned->m_geometryType = GeometryType::SkinnedMesh;
		directionalLightShadowMapSkinned->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		directionalLightShadowMapSkinned->m_descriptorSetLayouts.emplace_back(boneMatricesLayout);
		directionalLightShadowMapSkinned->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		directionalLightShadowMapSkinned->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = directionalLightShadowMapSkinned->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		directionalLightShadowMapSkinned->PreparePipeline();
		RegisterGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP_SKINNED", directionalLightShadowMapSkinned);
	}
	{
		const auto directionalLightShadowMapInstanced = std::make_shared<GraphicsPipeline>();
		directionalLightShadowMapInstanced->m_vertexShader = Resources::GetResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_INSTANCED_VERT");
		directionalLightShadowMapInstanced->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		directionalLightShadowMapInstanced->m_geometryType = GeometryType::Mesh;
		directionalLightShadowMapInstanced->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		directionalLightShadowMapInstanced->m_descriptorSetLayouts.emplace_back(instancedDataLayout);
		directionalLightShadowMapInstanced->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		directionalLightShadowMapInstanced->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = directionalLightShadowMapInstanced->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		directionalLightShadowMapInstanced->PreparePipeline();
		RegisterGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP_INSTANCED", directionalLightShadowMapInstanced);
	}
	{
		const auto directionalLightShadowMapStrand = std::make_shared<GraphicsPipeline>();
		directionalLightShadowMapStrand->m_vertexShader = Resources::GetResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_STRANDS_VERT");
		directionalLightShadowMapStrand->m_tessellationControlShader = Resources::GetResource<Shader>("SHADOW_MAP_STRANDS_TESC");
		directionalLightShadowMapStrand->m_tessellationEvaluationShader = Resources::GetResource<Shader>("SHADOW_MAP_STRANDS_TESE");
		directionalLightShadowMapStrand->m_geometryShader = Resources::GetResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_STRANDS_GEOM");
		directionalLightShadowMapStrand->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		directionalLightShadowMapStrand->m_geometryType = GeometryType::Mesh;
		directionalLightShadowMapStrand->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		directionalLightShadowMapStrand->m_descriptorSetLayouts.emplace_back(instancedDataLayout);
		directionalLightShadowMapStrand->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		directionalLightShadowMapStrand->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = directionalLightShadowMapStrand->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		directionalLightShadowMapStrand->PreparePipeline();
		RegisterGraphicsPipeline("DIRECTIONAL_LIGHT_SHADOW_MAP_STRANDS", directionalLightShadowMapStrand);
	}
	{
		const auto pointLightShadowMap = std::make_shared<GraphicsPipeline>();
		pointLightShadowMap->m_vertexShader = Resources::GetResource<Shader>("POINT_LIGHT_SHADOW_MAP_VERT");
		pointLightShadowMap->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		pointLightShadowMap->m_geometryType = GeometryType::Mesh;
		pointLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		pointLightShadowMap->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		pointLightShadowMap->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = pointLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		pointLightShadowMap->PreparePipeline();
		RegisterGraphicsPipeline("POINT_LIGHT_SHADOW_MAP", pointLightShadowMap);
	}
	{
		const auto pointLightShadowMapSkinned = std::make_shared<GraphicsPipeline>();
		pointLightShadowMapSkinned->m_vertexShader = Resources::GetResource<Shader>("POINT_LIGHT_SHADOW_MAP_SKINNED_VERT");
		pointLightShadowMapSkinned->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		pointLightShadowMapSkinned->m_geometryType = GeometryType::SkinnedMesh;
		pointLightShadowMapSkinned->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		pointLightShadowMapSkinned->m_descriptorSetLayouts.emplace_back(boneMatricesLayout);
		pointLightShadowMapSkinned->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		pointLightShadowMapSkinned->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = pointLightShadowMapSkinned->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		pointLightShadowMapSkinned->PreparePipeline();
		RegisterGraphicsPipeline("POINT_LIGHT_SHADOW_MAP_SKINNED", pointLightShadowMapSkinned);
	}
	{
		const auto pointLightShadowMapInstanced = std::make_shared<GraphicsPipeline>();
		pointLightShadowMapInstanced->m_vertexShader = Resources::GetResource<Shader>("POINT_LIGHT_SHADOW_MAP_INSTANCED_VERT");
		pointLightShadowMapInstanced->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		pointLightShadowMapInstanced->m_geometryType = GeometryType::Mesh;
		pointLightShadowMapInstanced->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		pointLightShadowMapInstanced->m_descriptorSetLayouts.emplace_back(instancedDataLayout);
		pointLightShadowMapInstanced->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		pointLightShadowMapInstanced->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = pointLightShadowMapInstanced->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		pointLightShadowMapInstanced->PreparePipeline();
		RegisterGraphicsPipeline("POINT_LIGHT_SHADOW_MAP_INSTANCED", pointLightShadowMapInstanced);
	}
	{
		const auto pointLightShadowMapStrand = std::make_shared<GraphicsPipeline>();
		pointLightShadowMapStrand->m_vertexShader = Resources::GetResource<Shader>("POINT_LIGHT_SHADOW_MAP_STRANDS_VERT");
		pointLightShadowMapStrand->m_tessellationControlShader = Resources::GetResource<Shader>("SHADOW_MAP_STRANDS_TESC");
		pointLightShadowMapStrand->m_tessellationEvaluationShader = Resources::GetResource<Shader>("SHADOW_MAP_STRANDS_TESE");
		pointLightShadowMapStrand->m_geometryShader = Resources::GetResource<Shader>("POINT_LIGHT_SHADOW_MAP_STRANDS_GEOM");
		pointLightShadowMapStrand->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		pointLightShadowMapStrand->m_geometryType = GeometryType::Mesh;
		pointLightShadowMapStrand->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		pointLightShadowMapStrand->m_descriptorSetLayouts.emplace_back(instancedDataLayout);
		pointLightShadowMapStrand->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		pointLightShadowMapStrand->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = pointLightShadowMapStrand->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		pointLightShadowMapStrand->PreparePipeline();
		RegisterGraphicsPipeline("POINT_LIGHT_SHADOW_MAP_STRANDS", pointLightShadowMapStrand);
	}
	{
		const auto spotLightShadowMap = std::make_shared<GraphicsPipeline>();
		spotLightShadowMap->m_vertexShader = Resources::GetResource<Shader>("SPOT_LIGHT_SHADOW_MAP_VERT");
		spotLightShadowMap->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		spotLightShadowMap->m_geometryType = GeometryType::Mesh;
		spotLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		spotLightShadowMap->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		spotLightShadowMap->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = spotLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		spotLightShadowMap->PreparePipeline();
		RegisterGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP", spotLightShadowMap);
	}
	{
		const auto spotLightShadowMap = std::make_shared<GraphicsPipeline>();
		spotLightShadowMap->m_vertexShader = Resources::GetResource<Shader>("SPOT_LIGHT_SHADOW_MAP_SKINNED_VERT");
		spotLightShadowMap->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		spotLightShadowMap->m_geometryType = GeometryType::SkinnedMesh;
		spotLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		spotLightShadowMap->m_descriptorSetLayouts.emplace_back(boneMatricesLayout);
		spotLightShadowMap->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		spotLightShadowMap->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = spotLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		spotLightShadowMap->PreparePipeline();
		RegisterGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP_SKINNED", spotLightShadowMap);
	}
	{
		const auto spotLightShadowMap = std::make_shared<GraphicsPipeline>();
		spotLightShadowMap->m_vertexShader = Resources::GetResource<Shader>("SPOT_LIGHT_SHADOW_MAP_INSTANCED_VERT");
		spotLightShadowMap->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		spotLightShadowMap->m_geometryType = GeometryType::Mesh;
		spotLightShadowMap->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		spotLightShadowMap->m_descriptorSetLayouts.emplace_back(instancedDataLayout);
		spotLightShadowMap->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		spotLightShadowMap->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = spotLightShadowMap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		spotLightShadowMap->PreparePipeline();
		RegisterGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP_INSTANCED", spotLightShadowMap);
	}
	{
		const auto spotLightShadowMapStrand = std::make_shared<GraphicsPipeline>();
		spotLightShadowMapStrand->m_vertexShader = Resources::GetResource<Shader>("SPOT_LIGHT_SHADOW_MAP_STRANDS_VERT");
		spotLightShadowMapStrand->m_tessellationControlShader = Resources::GetResource<Shader>("SHADOW_MAP_STRANDS_TESC");
		spotLightShadowMapStrand->m_tessellationEvaluationShader = Resources::GetResource<Shader>("SHADOW_MAP_STRANDS_TESE");
		spotLightShadowMapStrand->m_geometryShader = Resources::GetResource<Shader>("SPOT_LIGHT_SHADOW_MAP_STRANDS_GEOM");
		spotLightShadowMapStrand->m_fragmentShader = Resources::GetResource<Shader>("EMPTY_FRAG");
		spotLightShadowMapStrand->m_geometryType = GeometryType::Mesh;
		spotLightShadowMapStrand->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		spotLightShadowMapStrand->m_descriptorSetLayouts.emplace_back(instancedDataLayout);
		spotLightShadowMapStrand->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		spotLightShadowMapStrand->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		auto& pushConstantRange = spotLightShadowMapStrand->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(RenderInstancePushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		spotLightShadowMapStrand->PreparePipeline();
		RegisterGraphicsPipeline("SPOT_LIGHT_SHADOW_MAP_STRANDS", spotLightShadowMapStrand);
	}
	{
		const auto brdfLut = std::make_shared<GraphicsPipeline>();
		brdfLut->m_vertexShader = Resources::GetResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
		brdfLut->m_fragmentShader = Resources::GetResource<Shader>("ENVIRONMENTAL_MAP_BRDF_FRAG");
		brdfLut->m_geometryType = GeometryType::Mesh;

		brdfLut->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		brdfLut->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
		brdfLut->m_colorAttachmentFormats = { 1, VK_FORMAT_R16G16_SFLOAT };

		brdfLut->PreparePipeline();
		RegisterGraphicsPipeline("ENVIRONMENTAL_MAP_BRDF", brdfLut);
	}
	{
		const auto equirectangularToCubemap = std::make_shared<GraphicsPipeline>();
		equirectangularToCubemap->m_vertexShader = Resources::GetResource<Shader>("EQUIRECTANGULAR_MAP_TO_CUBEMAP_VERT");
		equirectangularToCubemap->m_fragmentShader = Resources::GetResource<Shader>("EQUIRECTANGULAR_MAP_TO_CUBEMAP_FRAG");
		equirectangularToCubemap->m_geometryType = GeometryType::Mesh;

		equirectangularToCubemap->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		equirectangularToCubemap->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		equirectangularToCubemap->m_colorAttachmentFormats = { 1, Constants::TEXTURE_2D };
		equirectangularToCubemap->m_descriptorSetLayouts.emplace_back(GetDescriptorSetLayout("RENDER_TEXTURE_PRESENT"));

		auto& pushConstantRange = equirectangularToCubemap->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(glm::mat4) + sizeof(float);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		equirectangularToCubemap->PreparePipeline();
		RegisterGraphicsPipeline("EQUIRECTANGULAR_TO_CUBEMAP", equirectangularToCubemap);
	}
	{
		const auto irradianceConstruct = std::make_shared<GraphicsPipeline>();
		irradianceConstruct->m_vertexShader = Resources::GetResource<Shader>("EQUIRECTANGULAR_MAP_TO_CUBEMAP_VERT");
		irradianceConstruct->m_fragmentShader = Resources::GetResource<Shader>("IRRADIANCE_CONSTRUCT_FRAG");
		irradianceConstruct->m_geometryType = GeometryType::Mesh;

		irradianceConstruct->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		irradianceConstruct->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		irradianceConstruct->m_colorAttachmentFormats = { 1, Constants::TEXTURE_2D };
		irradianceConstruct->m_descriptorSetLayouts.emplace_back(GetDescriptorSetLayout("RENDER_TEXTURE_PRESENT"));

		auto& pushConstantRange = irradianceConstruct->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(glm::mat4) + sizeof(float);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		irradianceConstruct->PreparePipeline();
		RegisterGraphicsPipeline("IRRADIANCE_CONSTRUCT", irradianceConstruct);
	}
	{
		const auto prefilterConstruct = std::make_shared<GraphicsPipeline>();
		prefilterConstruct->m_vertexShader = Resources::GetResource<Shader>("EQUIRECTANGULAR_MAP_TO_CUBEMAP_VERT");
		prefilterConstruct->m_fragmentShader = Resources::GetResource<Shader>("PREFILTER_CONSTRUCT_FRAG");
		prefilterConstruct->m_geometryType = GeometryType::Mesh;

		prefilterConstruct->m_depthAttachmentFormat = Constants::SHADOW_MAP;
		prefilterConstruct->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		prefilterConstruct->m_colorAttachmentFormats = { 1, Constants::TEXTURE_2D };
		prefilterConstruct->m_descriptorSetLayouts.emplace_back(GetDescriptorSetLayout("RENDER_TEXTURE_PRESENT"));

		auto& pushConstantRange = prefilterConstruct->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(glm::mat4) + sizeof(float);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		prefilterConstruct->PreparePipeline();
		RegisterGraphicsPipeline("PREFILTER_CONSTRUCT", prefilterConstruct);
	}
	{
		const auto gizmos = std::make_shared<GraphicsPipeline>();
		gizmos->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_VERT");
		gizmos->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_FRAG");
		gizmos->m_geometryType = GeometryType::Mesh;

		gizmos->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmos->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmos->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmos->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		auto& pushConstantRange = gizmos->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmos->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS", gizmos);
	}
	{
		const auto gizmosStrands = std::make_shared<GraphicsPipeline>();
		gizmosStrands->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_VERT");
		gizmosStrands->m_tessellationControlShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_TESC");
		gizmosStrands->m_tessellationEvaluationShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_TESE");
		gizmosStrands->m_geometryShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_GEOM");
		gizmosStrands->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_FRAG");
		gizmosStrands->m_geometryType = GeometryType::Strands;

		gizmosStrands->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmosStrands->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmosStrands->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmosStrands->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		auto& pushConstantRange = gizmosStrands->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmosStrands->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS_STRANDS", gizmosStrands);
	}
	{
		const auto gizmosNormalColored = std::make_shared<GraphicsPipeline>();
		gizmosNormalColored->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_NORMAL_COLORED_VERT");
		gizmosNormalColored->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_COLORED_FRAG");
		gizmosNormalColored->m_geometryType = GeometryType::Mesh;

		gizmosNormalColored->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmosNormalColored->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmosNormalColored->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmosNormalColored->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		auto& pushConstantRange = gizmosNormalColored->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmosNormalColored->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS_NORMAL_COLORED", gizmosNormalColored);
	}
	{
		const auto gizmosStrandsNormalColored = std::make_shared<GraphicsPipeline>();
		gizmosStrandsNormalColored->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_NORMAL_COLORED_VERT");
		gizmosStrandsNormalColored->m_tessellationControlShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_COLORED_TESC");
		gizmosStrandsNormalColored->m_tessellationEvaluationShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_COLORED_TESE");
		gizmosStrandsNormalColored->m_geometryShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_COLORED_GEOM");
		gizmosStrandsNormalColored->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_COLORED_FRAG");
		gizmosStrandsNormalColored->m_geometryType = GeometryType::Strands;

		gizmosStrandsNormalColored->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmosStrandsNormalColored->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmosStrandsNormalColored->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmosStrandsNormalColored->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		auto& pushConstantRange = gizmosStrandsNormalColored->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmosStrandsNormalColored->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS_STRANDS_NORMAL_COLORED", gizmosStrandsNormalColored);
	}
	{
		const auto gizmosVertexColored = std::make_shared<GraphicsPipeline>();
		gizmosVertexColored->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_VERTEX_COLORED_VERT");
		gizmosVertexColored->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_COLORED_FRAG");
		gizmosVertexColored->m_geometryType = GeometryType::Mesh;

		gizmosVertexColored->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmosVertexColored->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmosVertexColored->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmosVertexColored->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		auto& pushConstantRange = gizmosVertexColored->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmosVertexColored->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS_VERTEX_COLORED", gizmosVertexColored);
	}
	{
		const auto gizmosStrandsVertexColored = std::make_shared<GraphicsPipeline>();
		gizmosStrandsVertexColored->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_VERTEX_COLORED_VERT");
		gizmosStrandsVertexColored->m_tessellationControlShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_COLORED_TESC");
		gizmosStrandsVertexColored->m_tessellationEvaluationShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_COLORED_TESE");
		gizmosStrandsVertexColored->m_geometryShader = Resources::GetResource<Shader>("GIZMOS_STRANDS_COLORED_GEOM");
		gizmosStrandsVertexColored->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_COLORED_FRAG");
		gizmosStrandsVertexColored->m_geometryType = GeometryType::Strands;

		gizmosStrandsVertexColored->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmosStrandsVertexColored->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmosStrandsVertexColored->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmosStrandsVertexColored->m_descriptorSetLayouts.emplace_back(perFrameLayout);

		auto& pushConstantRange = gizmosStrandsVertexColored->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmosStrandsVertexColored->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS_STRANDS_VERTEX_COLORED", gizmosStrandsVertexColored);
	}
	{
		const auto gizmosInstancedColored = std::make_shared<GraphicsPipeline>();
		gizmosInstancedColored->m_vertexShader = Resources::GetResource<Shader>("GIZMOS_INSTANCED_COLORED_VERT");
		gizmosInstancedColored->m_fragmentShader = Resources::GetResource<Shader>("GIZMOS_COLORED_FRAG");
		gizmosInstancedColored->m_geometryType = GeometryType::Mesh;

		gizmosInstancedColored->m_depthAttachmentFormat = Constants::RENDER_TEXTURE_DEPTH;
		gizmosInstancedColored->m_stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

		gizmosInstancedColored->m_colorAttachmentFormats = { 1, Constants::RENDER_TEXTURE_COLOR };
		gizmosInstancedColored->m_descriptorSetLayouts.emplace_back(perFrameLayout);
		gizmosInstancedColored->m_descriptorSetLayouts.emplace_back(instancedDataLayout);

		auto& pushConstantRange = gizmosInstancedColored->m_pushConstantRanges.emplace_back();
		pushConstantRange.size = sizeof(GizmosPushConstant);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;

		gizmosInstancedColored->PreparePipeline();
		RegisterGraphicsPipeline("GIZMOS_INSTANCED_COLORED", gizmosInstancedColored);
	}
}

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
#pragma region Command pool
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = graphics.m_queueFamilyIndices.m_graphicsFamily.value();
		graphics.m_commandPool = std::make_unique<CommandPool>(poolInfo);
#pragma endregion
		graphics.m_usedCommandBufferSize = 0;
		graphics.m_commandBufferPool.resize(graphics.m_maxFrameInFlight);
		graphics.CreateSwapChainSyncObjects();

		constexpr VkDescriptorPoolSize renderLayerDescriptorPoolSizes[] =
		{
			{ VK_DESCRIPTOR_TYPE_SAMPLER, 65536 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 65536 },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 65536 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 65536 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 65536 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 65536 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 65536 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 65536 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 65536 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 65536 },
			{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 65536 }
		};

		VkDescriptorPoolCreateInfo renderLayerDescriptorPoolInfo{};
		renderLayerDescriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		renderLayerDescriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		renderLayerDescriptorPoolInfo.poolSizeCount = std::size(renderLayerDescriptorPoolSizes);
		renderLayerDescriptorPoolInfo.pPoolSizes = renderLayerDescriptorPoolSizes;
		renderLayerDescriptorPoolInfo.maxSets = 65536;
		graphics.m_descriptorPool = std::make_unique<DescriptorPool>(renderLayerDescriptorPoolInfo);


	}
#pragma endregion
	const auto& windowLayer = Application::GetLayer<WindowLayer>();
	const auto& editorLayer = Application::GetLayer<EditorLayer>();
	if (windowLayer && editorLayer)
	{
		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleFonts;
		//io.ConfigFlags |= ImGuiConfigFlags_IsSRGB;
		ImGui::StyleColorsDark();
		ImGui::CreateContext();
		ImGui_ImplGlfw_InitForVulkan(windowLayer->GetGlfwWindow(), true);
		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = graphics.m_vkInstance;
		init_info.PhysicalDevice = graphics.m_vkPhysicalDevice;
		init_info.Device = graphics.m_vkDevice;
		init_info.QueueFamily = graphics.m_queueFamilyIndices.m_graphicsFamily.value();
		init_info.Queue = graphics.m_vkGraphicsQueue;
		init_info.PipelineCache = VK_NULL_HANDLE;
		init_info.DescriptorPool = graphics.m_descriptorPool->GetVkDescriptorPool();
		init_info.MinImageCount = graphics.m_swapchain->GetAllImageViews().size();
		init_info.ImageCount = graphics.m_swapchain->GetAllImageViews().size();
		init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		init_info.UseDynamicRendering = true;
		init_info.ColorAttachmentFormat = graphics.m_swapchain->GetImageFormat();

		ImGui_ImplVulkan_LoadFunctions([](const char* function_name, void*) { return vkGetInstanceProcAddr(Graphics::GetVkInstance(), function_name); });
		ImGui_ImplVulkan_Init(&init_info, VK_NULL_HANDLE);
		ImGui::StyleColorsDark();
		ImmediateSubmit([&](const VkCommandBuffer cmd) {
			ImGui_ImplVulkan_CreateFontsTexture(cmd);
			});
		//clear font textures from cpu data
		ImGui_ImplVulkan_DestroyFontUploadObjects();
	}

	GeometryStorage::Initialize();
	TextureStorage::Initialize();
}

void Graphics::PostResourceLoadingInitialization()
{
	const auto& graphics = GetInstance();
	graphics.PrepareDescriptorSetLayouts();
	graphics.CreateGraphicsPipelines();
}

void Graphics::Destroy()
{
	auto& graphics = GetInstance();
	graphics.OnDestroy();

}

void Graphics::PreUpdate()
{
	auto& graphics = GetInstance();
	const auto windowLayer = Application::GetLayer<WindowLayer>();
	if (windowLayer)
	{
		if (glfwWindowShouldClose(windowLayer->m_window))
		{
			Application::End();
		}
		if (Application::GetLayer<RenderLayer>() || Application::GetLayer<EditorLayer>())
		{
			graphics.SwapChainSwapImage();
		}
	}
	else
	{
		if (Application::GetLayer<RenderLayer>())
		{
			graphics.WaitForCommandsComplete();
		}
	}

	graphics.ResetCommandBuffers();
	graphics.m_triangles = 0;
	graphics.m_strandsSegments = 0;
	graphics.m_drawCall = 0;
	GeometryStorage::PreUpdate();
	TextureStorage::PreUpdate();
	if (Application::GetLayer<RenderLayer>() && !Application::GetLayer<EditorLayer>()) {
		if (const auto scene = Application::GetActiveScene()) {
			if (const auto mainCamera = scene->m_mainCamera.Get<Camera>(); mainCamera && mainCamera->IsEnabled())
			{
				mainCamera->SetRequireRendering(true);
				if (windowLayer) mainCamera->Resize({ graphics.m_swapchain->GetImageExtent().width, graphics.m_swapchain->GetImageExtent().height });
			}
		}
	}
}

void Graphics::LateUpdate()
{
	auto& graphics = GetInstance();
	if (const auto windowLayer = Application::GetLayer<WindowLayer>())
	{
		if (Application::GetLayer<RenderLayer>() && !Application::GetLayer<EditorLayer>()) {
			if (const auto scene = Application::GetActiveScene()) {
				if (const auto mainCamera = scene->m_mainCamera.Get<Camera>();
					mainCamera->IsEnabled() && mainCamera->m_rendered)
				{
					const auto& renderTexturePresent = graphics.m_graphicsPipelines["RENDER_TEXTURE_PRESENT"];
					AppendCommands([&](VkCommandBuffer commandBuffer)
						{
							EverythingBarrier(commandBuffer);
							TransitImageLayout(commandBuffer,
								graphics.m_swapchain->GetVkImage(), graphics.m_swapchain->GetImageFormat(), 1,
								VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR);

							constexpr VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
							VkRect2D renderArea;
							renderArea.offset = { 0, 0 };
							renderArea.extent = graphics.m_swapchain->GetImageExtent();


							VkRenderingAttachmentInfo colorAttachmentInfo{};
							colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
							colorAttachmentInfo.imageView = graphics.m_swapchain->GetVkImageView();
							colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
							colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
							colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
							colorAttachmentInfo.clearValue = clearColor;

							VkRenderingInfo renderInfo{};
							renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
							renderInfo.renderArea = renderArea;
							renderInfo.layerCount = 1;
							renderInfo.colorAttachmentCount = 1;
							renderInfo.pColorAttachments = &colorAttachmentInfo;
							VkViewport viewport;
							viewport.x = 0.0f;
							viewport.y = 0.0f;
							viewport.width = renderArea.extent.width;
							viewport.height = renderArea.extent.height;
							viewport.minDepth = 0.0f;
							viewport.maxDepth = 1.0f;

							VkRect2D scissor;
							scissor.offset = { 0, 0 };
							scissor.extent.width = renderArea.extent.width;
							scissor.extent.height = renderArea.extent.height;

							renderTexturePresent->m_states.m_viewPort = viewport;
							renderTexturePresent->m_states.m_scissor = scissor;
							renderTexturePresent->m_states.m_colorBlendAttachmentStates.clear();
							renderTexturePresent->m_states.m_colorBlendAttachmentStates.resize(1);
							for (auto& i : renderTexturePresent->m_states.m_colorBlendAttachmentStates)
							{
								i.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
								i.blendEnable = VK_FALSE;
							}
							renderTexturePresent->m_states.m_depthTest = VK_FALSE;
							renderTexturePresent->m_states.m_depthWrite = VK_FALSE;
							vkCmdBeginRendering(commandBuffer, &renderInfo);
							//From main camera to swap chain.
							renderTexturePresent->Bind(commandBuffer);
							renderTexturePresent->BindDescriptorSet(commandBuffer, 0, mainCamera->GetRenderTexture()->m_descriptorSet->GetVkDescriptorSet());

							const auto mesh = Resources::GetResource<Mesh>("PRIMITIVE_TEX_PASS_THROUGH");
							mesh->Bind(commandBuffer);
							mesh->DrawIndexed(commandBuffer, renderTexturePresent->m_states, 1, false);
							vkCmdEndRendering(commandBuffer);
							TransitImageLayout(commandBuffer,
								graphics.m_swapchain->GetVkImage(), graphics.m_swapchain->GetImageFormat(), 1,
								VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
						});
				}
			}
		}
		graphics.SubmitPresent();
	}
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

