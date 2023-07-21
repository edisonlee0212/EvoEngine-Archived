#pragma once
#include "shaderc/shaderc.h"
namespace EvoEngine
{
	class IGraphicsResource
	{
	protected:
		IGraphicsResource() = default;
	public:
		IGraphicsResource& operator=(IGraphicsResource&) = delete;
		IGraphicsResource& operator=(const IGraphicsResource&) = delete;
		virtual ~IGraphicsResource() = default;
	};
	class Fence final : public IGraphicsResource
	{
		VkFence m_vkFence = VK_NULL_HANDLE;
	public:
		explicit Fence(const VkFenceCreateInfo& vkFenceCreateInfo);
		~Fence() override;

		[[nodiscard]] VkFence GetVkFence() const;
	};

	class Semaphore final : public IGraphicsResource
	{
		VkSemaphore m_vkSemaphore = VK_NULL_HANDLE;
	public:
		explicit Semaphore(const VkSemaphoreCreateInfo& semaphoreCreateInfo);
		~Semaphore() override;
		[[nodiscard]] VkSemaphore GetVkSemaphore() const;
	};

	class ImageView final : public IGraphicsResource
	{
		VkImageView m_vkImageView = VK_NULL_HANDLE;
	public:
		explicit ImageView(const VkImageViewCreateInfo& imageViewCreateInfo);
		~ImageView() override;

		[[nodiscard]] VkImageView GetVkImageView() const;
	};

	class Framebuffer final : public IGraphicsResource
	{
		VkFramebuffer m_vkFramebuffer = VK_NULL_HANDLE;
	public:
		explicit Framebuffer(const VkFramebufferCreateInfo& framebufferCreateInfo);
		~Framebuffer() override;

		[[nodiscard]] VkFramebuffer GetVkFrameBuffer() const;
	};

	class Swapchain final : public IGraphicsResource
	{
		VkSwapchainKHR m_vkSwapchain = VK_NULL_HANDLE;
		std::vector<VkImage> m_vkImages;

		VkFormat m_vkFormat = {};
		VkExtent2D m_vkExtent2D = {};

		std::vector<VkImageView> m_vkImageViews;


	public:
		explicit Swapchain(const VkSwapchainCreateInfoKHR& swapchainCreateInfo);
		~Swapchain() override;

		[[nodiscard]] VkSwapchainKHR GetVkSwapchain() const;

		[[nodiscard]] const std::vector<VkImage>& GetVkImages() const;
		[[nodiscard]] const std::vector<VkImageView>& GetVkImageViews() const;
		[[nodiscard]] VkFormat GetVkFormat() const;

		[[nodiscard]] VkExtent2D GetVkExtent2D() const;
	};

	class ShaderModule final : public IGraphicsResource
	{
		shaderc_shader_kind m_shaderKind = shaderc_glsl_infer_from_source;
		VkShaderModule m_vkShaderModule = VK_NULL_HANDLE;
	public:
		ShaderModule(shaderc_shader_kind shaderKind, const std::vector<char>& code);
		~ShaderModule() override;

		ShaderModule(shaderc_shader_kind shaderKind, const std::string& code);

		[[nodiscard]] VkShaderModule GetVkShaderModule() const;
	};

	class RenderPass final : public IGraphicsResource
	{
		VkRenderPass m_vkRenderPass = VK_NULL_HANDLE;
	public:
		RenderPass(const VkRenderPassCreateInfo& renderPassCreateInfo);
		~RenderPass() override;

		[[nodiscard]] VkRenderPass GetVkRenderPass() const;
	};

	class PipelineLayout final : public IGraphicsResource
	{
		VkPipelineLayout m_vkPipelineLayout = VK_NULL_HANDLE;
	public:
		PipelineLayout(const VkPipelineLayoutCreateInfo& pipelineLayoutCreateInfo);
		~PipelineLayout() override;

		[[nodiscard]] VkPipelineLayout GetVkPipelineLayout() const;
	};

	class GraphicsPipeline final : public IGraphicsResource
	{
		VkPipeline m_vkGraphicsPipeline = VK_NULL_HANDLE;
	public:
		explicit GraphicsPipeline(const VkGraphicsPipelineCreateInfo& graphicsPipelineCreateInfo);

		~GraphicsPipeline() override;

		[[nodiscard]] VkPipeline GetVkPipeline() const;
	};

	class CommandPool final : public IGraphicsResource
	{
		VkCommandPool m_vkCommandPool = VK_NULL_HANDLE;
	public:
		explicit CommandPool(const VkCommandPoolCreateInfo& commandPoolCreateInfo);

		~CommandPool() override;

		[[nodiscard]] VkCommandPool GetVkCommandPool() const;
	};

	class Buffer final : public IGraphicsResource
	{
		VkBuffer m_vkBuffer = VK_NULL_HANDLE;
		VmaAllocation m_vmaAllocation = VK_NULL_HANDLE;
		VmaAllocationInfo m_vmaAllocationInfo = {};
	public:
		explicit Buffer(const VkBufferCreateInfo& bufferCreateInfo);
		~Buffer() override;
		Buffer(const VkBufferCreateInfo& bufferCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo);

		void Copy(const Buffer& srcBuffer, VkDeviceSize size, VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0) const;

		[[nodiscard]] VkBuffer GetVkBuffer() const;

		[[nodiscard]] VmaAllocation GetVmaAllocation() const;

		[[nodiscard]] const VmaAllocationInfo& GetVmaAllocationInfo() const;
	};

	class Image final : public IGraphicsResource
	{
		VkImage m_vkImage = VK_NULL_HANDLE;
		VmaAllocation m_vmaAllocation = VK_NULL_HANDLE;
		VmaAllocationInfo m_vmaAllocationInfo = {};
		VkImageLayout m_vkImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VkExtent3D m_extent = {0, 0, 0};
	public:
		explicit Image(const VkImageCreateInfo& imageCreateInfo);
		Image(const VkImageCreateInfo& imageCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo);

		~Image() override;
		void TransitionImageLayout(VkImageLayout newLayout);
		void Copy(const VkBuffer& srcBuffer, VkDeviceSize srcOffset = 0) const;

		[[nodiscard]] VkImage GetVkImage() const;

		[[nodiscard]] VmaAllocation GetVmaAllocation() const;
		[[nodiscard]] VkExtent3D GetVkExtent3D() const;
		[[nodiscard]] const VmaAllocationInfo& GetVmaAllocationInfo() const;
	};

	class Sampler final : public IGraphicsResource
	{
		VkSampler m_vkSampler;
	public:
		explicit Sampler(const VkSamplerCreateInfo& samplerCreateInfo);
		~Sampler() override;
		[[nodiscard]] VkSampler GetVkSampler() const;
	};

	class DescriptorSetLayout final : public IGraphicsResource
	{
		VkDescriptorSetLayout m_vkDescriptorSetLayout = VK_NULL_HANDLE;
	public:
		explicit DescriptorSetLayout(const VkDescriptorSetLayoutCreateInfo& descriptorSetLayoutCreateInfo);
		~DescriptorSetLayout() override;
		[[nodiscard]] VkDescriptorSetLayout GetVkDescriptorSetLayout() const;
	};

	class DescriptorPool final : public IGraphicsResource
	{
		VkDescriptorPool m_vkDescriptorPool = VK_NULL_HANDLE;
	public:
		explicit DescriptorPool(const VkDescriptorPoolCreateInfo& descriptorPoolCreateInfo);
		~DescriptorPool() override;
		[[nodiscard]] VkDescriptorPool GetVkDescriptorPool() const;
	};
}
