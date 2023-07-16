#pragma once
#include "shaderc/shaderc.h"
namespace EvoEngine
{
	class IGraphicsResource
	{
	public:
		IGraphicsResource() = default;
		virtual void Destroy();
		virtual ~IGraphicsResource();

		IGraphicsResource(const IGraphicsResource&) = delete;
		IGraphicsResource& operator=(IGraphicsResource&) = delete;
	};
	class Fence final : public IGraphicsResource
	{
		VkFence m_vkFence = VK_NULL_HANDLE;
	public:
		void Create(const VkFenceCreateInfo& vkFenceCreateInfo);
		void Destroy() override;

		VkFence GetVkFence() const;
	};

	class Semaphore final : public IGraphicsResource
	{
		VkSemaphore m_vkSemaphore = VK_NULL_HANDLE;
	public:
		void Create(const VkSemaphoreCreateInfo& semaphoreCreateInfo);
		void Destroy() override;
		VkSemaphore GetVkSemaphore() const;
	};

	

	class Image final : public IGraphicsResource
	{
		VkImage m_vkImage = VK_NULL_HANDLE;

	public:
		void Create(const VkImageCreateInfo& imageCreateInfo);
		void Destroy() override;

		VkImage GetVkImage() const;
	};

	class ImageView final : public IGraphicsResource
	{
		VkImageView m_vkImageView = VK_NULL_HANDLE;
	public:
		void Create(const VkImageViewCreateInfo& imageViewCreateInfo);
		void Destroy() override;

		VkImageView GetVkImageView() const;
	};

	class Framebuffer final : public IGraphicsResource
	{
		VkFramebuffer m_vkFramebuffer = VK_NULL_HANDLE;
	public:
		void Create(const VkFramebufferCreateInfo& framebufferCreateInfo);
		void Destroy() override;

		VkFramebuffer GetVkFrameBuffer() const;
	};

	class Swapchain final : public IGraphicsResource
	{
		VkSwapchainKHR m_vkSwapchain = VK_NULL_HANDLE;
		std::vector<VkImage> m_vkImages;

		VkFormat m_vkFormat = {};
		VkExtent2D m_vkExtent2D = {};

		std::vector<VkImageView> m_vkImageViews;

		
	public:
		void Create(const VkSwapchainCreateInfoKHR& swapchainCreateInfo);
		void Destroy() override;

		VkSwapchainKHR GetVkSwapchain() const;

		const std::vector<VkImage>& GetVkImages() const;
		const std::vector<VkImageView>& GetVkImageViews() const;
		VkFormat GetVkFormat() const;

		VkExtent2D GetVkExtent2D() const;
	};

	class ShaderModule final : public IGraphicsResource
	{
		shaderc_shader_kind m_shaderKind = shaderc_glsl_infer_from_source;
		VkShaderModule m_vkShaderModule = VK_NULL_HANDLE;
	public:
		void Create(shaderc_shader_kind shaderKind, const std::vector<char>& code);
		void Destroy() override;

		void Create(shaderc_shader_kind shaderKind, const std::string& code);

		VkShaderModule GetVkShaderModule() const;
	};

	class RenderPass final : public IGraphicsResource
	{
		VkRenderPass m_vkRenderPass = VK_NULL_HANDLE;
	public:
		void Create(const VkRenderPassCreateInfo& renderPassCreateInfo);
		void Destroy() override;

		VkRenderPass GetVkRenderPass() const;
	};

	class PipelineLayout final : public IGraphicsResource
	{
		VkPipelineLayout m_vkPipelineLayout = VK_NULL_HANDLE;
	public:
		void Create(const VkPipelineLayoutCreateInfo& pipelineLayoutCreateInfo);
		void Destroy() override;

		VkPipelineLayout GetVkPipelineLayout() const;
	};

	class GraphicsPipeline final : public IGraphicsResource
	{
		VkPipeline m_vkGraphicsPipeline = VK_NULL_HANDLE;
	public:
		void Create(const VkGraphicsPipelineCreateInfo& graphicsPipelineCreateInfo);

		void Destroy() override;

		VkPipeline GetVkPipeline() const;
	};

	class CommandPool final : public IGraphicsResource
	{
		VkCommandPool m_vkCommandPool = VK_NULL_HANDLE;
	public:
		void Create(const VkCommandPoolCreateInfo& commandPoolCreateInfo);

		void Destroy() override;

		VkCommandPool GetVkCommandPool() const;
	};
}
