#pragma once
#include "shaderc/shaderc.h"
namespace EvoEngine
{
	class Fence
	{
		VkFence m_vkFence = VK_NULL_HANDLE;
	public:
		void Create(const VkFenceCreateInfo& vkFenceCreateInfo);
		void Destroy(); 

		VkFence GetVkFence() const;
	};

	class Semaphore
	{
		VkSemaphore m_vkSemaphore = VK_NULL_HANDLE;
	public:
		void Create(const VkSemaphoreCreateInfo& semaphoreCreateInfo);
		void Destroy();
		VkSemaphore GetVkSemaphore() const;
	};

	

	class Image
	{
		VkImage m_vkImage = VK_NULL_HANDLE;

	public:
		void Create(const VkImageCreateInfo& imageCreateInfo);
		void Destroy();

		VkImage GetVkImage() const;
	};

	class ImageView
	{
		VkImageView m_vkImageView = VK_NULL_HANDLE;
	public:
		void Create(const VkImageViewCreateInfo& imageViewCreateInfo);
		void Destroy();

		VkImageView GetVkImageView() const;
	};

	class Framebuffer
	{
		VkFramebuffer m_vkFramebuffer = VK_NULL_HANDLE;
	public:
		void Create(const VkFramebufferCreateInfo& framebufferCreateInfo);
		void Destroy();

		VkFramebuffer GetVkFrameBuffer() const;
	};

	class Swapchain
	{
		VkSwapchainKHR m_vkSwapchain = VK_NULL_HANDLE;
		std::vector<VkImage> m_vkImages;

		VkFormat m_vkFormat = {};
		VkExtent2D m_vkExtent2D = {};

		std::vector<ImageView> m_imageViews;

		
	public:
		void Create(const VkSwapchainCreateInfoKHR& swapchainCreateInfo);
		void Destroy();

		VkSwapchainKHR GetVkSwapchain() const;

		const std::vector<VkImage>& GetVkImages() const;
		const std::vector<ImageView>& GetImageViews() const;
		VkFormat GetVkFormat() const;

		VkExtent2D GetVkExtent2D() const;
	};

	class ShaderModule
	{
		shaderc_shader_kind m_shaderKind = shaderc_glsl_infer_from_source;
		VkShaderModule m_vkShaderModule = VK_NULL_HANDLE;
	public:
		void Create(shaderc_shader_kind shaderKind, const std::vector<char>& code);
		void Destroy();

		void Create(shaderc_shader_kind shaderKind, const std::string& code);

		VkShaderModule GetVkShaderModule() const;
	};

	class RenderPass
	{
		VkRenderPass m_vkRenderPass = VK_NULL_HANDLE;
	public:
		void Create(const VkRenderPassCreateInfo& renderPassCreateInfo);
		void Destroy();

		VkRenderPass GetVkRenderPass() const;
	};

	class PipelineLayout
	{
		VkPipelineLayout m_vkPipelineLayout = VK_NULL_HANDLE;
	public:
		void Create(const VkPipelineLayoutCreateInfo& pipelineLayoutCreateInfo);
		void Destroy();

		VkPipelineLayout GetVkPipelineLayout() const;
	};

	class GraphicsPipeline
	{
		VkPipeline m_vkGraphicsPipeline = VK_NULL_HANDLE;
	public:
		void Create(const VkGraphicsPipelineCreateInfo& graphicsPipelineCreateInfo);

		void Destroy();

		VkPipeline GetVkPipeline() const;
	};

	class CommandPool
	{
		VkCommandPool m_vkCommandPool = VK_NULL_HANDLE;
	public:
		void Create(const VkCommandPoolCreateInfo& commandPoolCreateInfo);

		void Destroy();

		VkCommandPool GetVkCommandPool() const;
	};
}
