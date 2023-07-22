#include "Graphics.hpp"
#include "Console.hpp"
#include "Application.hpp"
#include "Utilities.hpp"
#include "WindowLayer.hpp"

#include "Mesh.hpp"
#include "ProjectManager.hpp"
#include "Vertex.hpp"
#include "vk_mem_alloc.h"
#include "EditorLayer.hpp"
#include "RenderLayer.hpp"
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

GlobalPipelineState& Graphics::GlobalState()
{
	auto& graphics = GetInstance();
	return graphics.m_globalPipelineState;
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

VkCommandBuffer Graphics::GetCurrentVkCommandBuffer()
{
	const auto& graphics = GetInstance();
	return graphics.m_vkCommandBuffers[graphics.m_currentFrameIndex];
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


void GlobalPipelineState::ResetAllStates(VkCommandBuffer commandBuffer)
{
	m_viewPort = {};
	m_viewPort.width = 1;
	m_viewPort.height = 1;
	m_scissor = {};
	m_patchControlPoints = 1;
	m_depthClamp = false;
	m_rasterizerDiscard = false;
	m_polygonMode = VK_POLYGON_MODE_FILL;
	m_cullMode = VK_CULL_MODE_BACK_BIT;
	m_frontFace = VK_FRONT_FACE_CLOCKWISE;
	m_depthBias = false;
	m_depthBiasConstantClampSlope = glm::vec3(0.0f);
	m_lineWidth = 1.0f;
	m_depthTest = true;
	m_depthWrite = true;
	m_depthCompare = VK_COMPARE_OP_LESS;
	m_depthBoundTest = false;
	m_minMaxDepthBound = glm::vec2(0.0f, 1.0f);
	m_stencilTest = false;
	m_stencilFaceMask = VK_STENCIL_FACE_FRONT_BIT;
	m_stencilFailOp = VK_STENCIL_OP_ZERO;
	m_stencilPassOp = VK_STENCIL_OP_ZERO;
	m_stencilDepthFailOp = VK_STENCIL_OP_ZERO;
	m_stencilCompareOp = VK_COMPARE_OP_LESS;

	m_vertexShader = {};
	m_tessellationControlShader = {};
	m_tessellationEvaluationShader = {};
	m_geometryShader = {};
	m_fragShader = {};
	m_computeShader = {};

	ApplyAllStates(commandBuffer, true);
}

void GlobalPipelineState::ApplyAllStates(const VkCommandBuffer commandBuffer, const bool forceSet)
{
	m_viewPortApplied = m_viewPort;
	m_viewPort.width = glm::max(1.0f, m_viewPort.width);
	m_viewPort.height = glm::max(1.0f, m_viewPort.height);
	m_scissorApplied = m_scissor;
	vkCmdSetViewport(commandBuffer, 0, 1, &m_viewPortApplied);
	vkCmdSetScissor(commandBuffer, 0, 1, &m_scissorApplied);
	if (forceSet || m_patchControlPointsApplied != m_patchControlPoints) {
		m_patchControlPointsApplied = m_patchControlPoints;
		vkCmdSetPatchControlPointsEXT(commandBuffer, m_patchControlPointsApplied);
	}
	if (forceSet || m_depthClampApplied != m_depthClamp) {
		m_depthClampApplied = m_depthClamp;
		vkCmdSetDepthClampEnableEXT(commandBuffer, m_depthClampApplied);
	}
	if (forceSet || m_rasterizerDiscardApplied != m_rasterizerDiscard) {
		m_rasterizerDiscardApplied = m_rasterizerDiscard;
		vkCmdSetRasterizerDiscardEnable(commandBuffer, m_rasterizerDiscardApplied);
	}
	if (forceSet || m_polygonModeApplied != m_polygonMode) {
		m_polygonModeApplied = m_polygonMode;
		vkCmdSetPolygonModeEXT(commandBuffer, m_polygonModeApplied);
	}
	if (forceSet || m_cullModeApplied != m_cullMode) {
		m_cullModeApplied = m_cullMode;
		vkCmdSetCullModeEXT(commandBuffer, m_cullModeApplied);
	}
	if (forceSet || m_frontFaceApplied != m_frontFace) {
		m_frontFaceApplied = m_frontFace;
		vkCmdSetFrontFace(commandBuffer, m_frontFaceApplied);
	}
	if (forceSet || m_depthBiasApplied != m_depthBias) {
		m_depthBiasApplied = m_depthBias;
		vkCmdSetDepthBiasEnable(commandBuffer, m_depthBiasApplied);
	}
	if (forceSet || m_depthBiasConstantClampSlopeApplied != m_depthBiasConstantClampSlope) {
		m_depthBiasConstantClampSlopeApplied = m_depthBiasConstantClampSlope;
		vkCmdSetDepthBias(commandBuffer, m_depthBiasConstantClampSlopeApplied.x, m_depthBiasConstantClampSlopeApplied.y, m_depthBiasConstantClampSlopeApplied.z);
	}
	if (forceSet || m_lineWidthApplied != m_lineWidth) {
		m_lineWidthApplied = m_lineWidth;
		vkCmdSetLineWidth(commandBuffer, m_lineWidthApplied);
	}
	if (forceSet || m_depthTestApplied != m_depthTest) {
		m_depthTestApplied = m_depthTest;
		vkCmdSetDepthTestEnableEXT(commandBuffer, m_depthTestApplied);
	}
	if (forceSet || m_depthWriteApplied != m_depthWrite) {
		m_depthWriteApplied = m_depthWrite;
		vkCmdSetDepthWriteEnableEXT(commandBuffer, m_depthWriteApplied);
	}
	if (forceSet || m_depthCompareApplied != m_depthCompare) {
		m_depthCompareApplied = m_depthCompare;
		vkCmdSetDepthCompareOpEXT(commandBuffer, m_depthCompareApplied);
	}
	if (forceSet || m_depthBoundTestApplied != m_depthBoundTest) {
		m_depthBoundTestApplied = m_depthBoundTest;
		vkCmdSetDepthBoundsTestEnableEXT(commandBuffer, m_depthBoundTestApplied);
	}
	if (forceSet || m_minMaxDepthBoundApplied != m_minMaxDepthBound) {
		m_minMaxDepthBoundApplied = m_minMaxDepthBound;
		vkCmdSetDepthBounds(commandBuffer, m_minMaxDepthBoundApplied.x, m_minMaxDepthBoundApplied.y);
	}
	if (forceSet || m_stencilTestApplied != m_stencilTest) {
		m_stencilTestApplied = m_stencilTest;
		vkCmdSetStencilTestEnableEXT(commandBuffer, m_stencilTestApplied);
	}
	if (forceSet || 
		m_frontFaceApplied != m_frontFace
		|| m_stencilFailOpApplied != m_stencilFailOp
		|| m_stencilPassOpApplied != m_stencilPassOp
		|| m_stencilDepthFailOpApplied != m_stencilDepthFailOp
		|| m_stencilCompareOpApplied != m_stencilCompareOp) {
		m_stencilFaceMaskApplied = m_stencilFaceMask;
		m_stencilFailOpApplied = m_stencilFailOp;
		m_stencilPassOpApplied = m_stencilPassOp;
		m_stencilDepthFailOpApplied = m_stencilDepthFailOp;
		m_stencilCompareOpApplied = m_stencilCompareOp;
		vkCmdSetStencilOpEXT(commandBuffer, m_stencilFaceMaskApplied, m_stencilFailOpApplied, m_stencilPassOpApplied, m_stencilDepthFailOpApplied, m_stencilCompareOpApplied);
	}
	constexpr  VkShaderStageFlagBits stages[6] =
	{
		VK_SHADER_STAGE_VERTEX_BIT,
		VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
		VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
		VK_SHADER_STAGE_GEOMETRY_BIT,
		VK_SHADER_STAGE_FRAGMENT_BIT,
		VK_SHADER_STAGE_COMPUTE_BIT
	};
	if (forceSet || m_vertexShaderApplied != m_vertexShader) {
		m_vertexShaderApplied = m_vertexShader;
		if (m_vertexShaderApplied) vkCmdBindShadersEXT(commandBuffer, 1, &stages[0], &m_vertexShaderApplied->GetVkShaderEXT());
		else vkCmdBindShadersEXT(commandBuffer, 1, &stages[0], nullptr);
	}
	if (forceSet || m_tessellationControlShaderApplied != m_tessellationControlShader) {
		m_tessellationControlShaderApplied = m_tessellationControlShader;
		if (m_tessellationControlShaderApplied) vkCmdBindShadersEXT(commandBuffer, 1, &stages[1], &m_tessellationControlShaderApplied->GetVkShaderEXT());
		else vkCmdBindShadersEXT(commandBuffer, 1, &stages[1], nullptr);
	}
	if (forceSet || m_tessellationEvaluationShaderApplied != m_tessellationEvaluationShader) {
		m_tessellationEvaluationShaderApplied = m_tessellationEvaluationShader;
		if (m_tessellationEvaluationShaderApplied) vkCmdBindShadersEXT(commandBuffer, 1, &stages[2], &m_tessellationEvaluationShaderApplied->GetVkShaderEXT());
		else vkCmdBindShadersEXT(commandBuffer, 1, &stages[2], nullptr);
	}
	if (forceSet || m_geometryShaderApplied != m_geometryShader) {
		m_geometryShaderApplied = m_geometryShader;
		if (m_geometryShaderApplied) vkCmdBindShadersEXT(commandBuffer, 1, &stages[3], &m_geometryShaderApplied->GetVkShaderEXT());
		else vkCmdBindShadersEXT(commandBuffer, 1, &stages[3], nullptr);
	}
	if (forceSet || m_fragShaderApplied != m_fragShader) {
		m_fragShaderApplied = m_fragShader;
		if (m_fragShaderApplied) vkCmdBindShadersEXT(commandBuffer, 1, &stages[4], &m_fragShaderApplied->GetVkShaderEXT());
		else vkCmdBindShadersEXT(commandBuffer, 1, &stages[4], nullptr);
	}
	if (forceSet || m_computeShaderApplied != m_computeShader) {
		m_computeShaderApplied = m_computeShader;
		if (m_computeShader) vkCmdBindShadersEXT(commandBuffer, 1, &stages[5], &m_computeShader->GetVkShaderEXT());
		else vkCmdBindShadersEXT(commandBuffer, 1, &stages[5], nullptr);
	}
}

void GlobalPipelineState::ClearShaders()
{
	m_vertexShader = {};
	m_tessellationControlShader = {};
	m_tessellationEvaluationShader = {};
	m_geometryShader = {};
	m_fragShader = {};
	m_computeShader = {};
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
	m_requiredDeviceExtensions.emplace_back(VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
	m_requiredDeviceExtensions.emplace_back(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
	m_requiredDeviceExtensions.emplace_back(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME);
	m_requiredDeviceExtensions.emplace_back(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);

	std::vector<const char*> cRequiredDeviceExtensions;
	for (const auto& i : m_requiredDeviceExtensions) cRequiredDeviceExtensions.emplace_back(i.c_str());
	std::vector<const char*> cRequiredLayers;
	for (const auto& i : m_requiredLayers) cRequiredLayers.emplace_back(i.c_str());

#pragma region Logical Device
	VkPhysicalDeviceFeatures deviceFeatures{};
	deviceFeatures.samplerAnisotropy = VK_TRUE;
	VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT extendedVertexInputDynamicStateFeatures{};
	extendedVertexInputDynamicStateFeatures.vertexInputDynamicState = VK_TRUE;
	extendedVertexInputDynamicStateFeatures.pNext = nullptr;
	extendedVertexInputDynamicStateFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;

	VkPhysicalDeviceExtendedDynamicState3FeaturesEXT extendedDynamicState3Features{};
	extendedDynamicState3Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
	extendedDynamicState3Features.extendedDynamicState3PolygonMode = VK_TRUE;
	extendedDynamicState3Features.extendedDynamicState3DepthClampEnable = VK_TRUE;
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

	VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{};
	shaderObjectFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
	shaderObjectFeatures.shaderObject = VK_TRUE;
	shaderObjectFeatures.pNext = &extendedDynamicStateFeatures;

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
	deviceCreateInfo.pNext = &shaderObjectFeatures;

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
void Graphics::CreateCommandBuffers(const std::unique_ptr<CommandPool>& commandPool, std::vector<VkCommandBuffer>& commandBuffers)
{
	const auto& graphics = GetInstance();
#pragma region Command buffers
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = commandPool->GetVkCommandPool();
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = static_cast<uint32_t>(graphics.m_maxFrameInFlight);
	commandBuffers.resize(graphics.m_maxFrameInFlight);

	if (vkAllocateCommandBuffers(graphics.m_vkDevice, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
		throw std::runtime_error("Failed to allocate command buffers!");
	}
#pragma endregion
}

void Graphics::AppendCommands(const std::function<void(VkCommandBuffer commandBuffer, GlobalPipelineState& globalPipelineState)>& action)
{
	auto& graphics = GetInstance();
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	if (vkBeginCommandBuffer(graphics.m_vkCommandBuffers[graphics.m_currentFrameIndex], &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("Failed to begin recording command buffer!");
	}

	action(graphics.m_vkCommandBuffers[graphics.m_currentFrameIndex], graphics.m_globalPipelineState);
	if (vkEndCommandBuffer(graphics.m_vkCommandBuffers[graphics.m_currentFrameIndex]) != VK_SUCCESS) {
		throw std::runtime_error("Failed to record command buffer!");
	}
}

void Graphics::CreateSwapChainSyncObjects()
{
#pragma region Sync Objects
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
#pragma endregion
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
	m_inFlightFences.clear();
	m_renderFinishedSemaphores.clear();
	m_imageAvailableSemaphores.clear();
	m_commandPool.reset();
	m_swapchain.reset();

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
	vkResetCommandBuffer(m_vkCommandBuffers[m_currentFrameIndex], 0);
}

void Graphics::SubmitPresent()
{
	const auto& windowLayer = Application::GetLayer<WindowLayer>();
	if (windowLayer && (windowLayer->m_windowSize.x == 0 || windowLayer->m_windowSize.y == 0)) return;
	if (!m_queueFamilyIndices.m_presentFamily.has_value()) return;
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	VkSemaphore waitSemaphores[] = { m_imageAvailableSemaphores[m_currentFrameIndex]->GetVkSemaphore() };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &m_vkCommandBuffers[m_currentFrameIndex];

	VkSemaphore signalSemaphores[] = { m_renderFinishedSemaphores[m_currentFrameIndex]->GetVkSemaphore() };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;
	if (vkQueueSubmit(m_vkGraphicsQueue, 1, &submitInfo, m_inFlightFences[m_currentFrameIndex]->GetVkFence()) != VK_SUCCESS) {
		throw std::runtime_error("failed to submit draw command buffer!");
	}

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	VkSwapchainKHR swapChains[] = { m_swapchain->GetVkSwapchain() };
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
#pragma region Command pool
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = graphics.m_queueFamilyIndices.m_graphicsFamily.value();
		graphics.m_commandPool = std::make_unique<CommandPool>(poolInfo);
#pragma endregion
		CreateCommandBuffers(graphics.m_commandPool, graphics.m_vkCommandBuffers);
		graphics.CreateSwapChainSyncObjects();

		const VkDescriptorPoolSize renderLayerDescriptorPoolSizes[] =
		{
			{ VK_DESCRIPTOR_TYPE_SAMPLER, 1024 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024 },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1024 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1024 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1024 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1024 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1024 },
			{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1024 }
		};

		VkDescriptorPoolCreateInfo renderLayerDescriptorPoolInfo{};
		renderLayerDescriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		renderLayerDescriptorPoolInfo.poolSizeCount = std::size(renderLayerDescriptorPoolSizes);
		renderLayerDescriptorPoolInfo.pPoolSizes = renderLayerDescriptorPoolSizes;
		renderLayerDescriptorPoolInfo.maxSets = 1024;
		graphics.m_descriptorPool = std::make_unique<DescriptorPool>(renderLayerDescriptorPoolInfo);

	}
#pragma endregion


}

void Graphics::Destroy()
{
	auto& graphics = GetInstance();
	graphics.OnDestroy();

}

void Graphics::PreUpdate()
{
	auto& graphics = GetInstance();
	if (const auto windowLayer = Application::GetLayer<WindowLayer>())
	{
		glfwPollEvents();
		if (glfwWindowShouldClose(windowLayer->m_window))
		{
			Application::End();
		}
		if (Application::GetLayer<RenderLayer>() || Application::GetLayer<EditorLayer>())
		{
			graphics.SwapChainSwapImage();
		}
	}
	AppendCommands([&](const VkCommandBuffer commandBuffer, GlobalPipelineState& globalPipelineState)
		{
			globalPipelineState.ResetAllStates(commandBuffer);
		}
	);
}

void Graphics::LateUpdate()
{
	auto& graphics = GetInstance();
	if (const auto windowLayer = Application::GetLayer<WindowLayer>())
	{
		if (Application::GetLayer<RenderLayer>() || Application::GetLayer<EditorLayer>())
		{
			graphics.SubmitPresent();
		}
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

