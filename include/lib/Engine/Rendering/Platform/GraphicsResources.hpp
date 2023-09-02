#pragma once
#include "Console.hpp"
#include "shaderc/shaderc.h"
namespace EvoEngine
{
	class IGraphicsResource
	{
	protected:
		IGraphicsResource() = default;
	public:
		template<typename T>
		static void ApplyVector(std::vector<T>& target, uint32_t size, const T* data);
		IGraphicsResource& operator=(IGraphicsResource&) = delete;
		IGraphicsResource& operator=(const IGraphicsResource&) = delete;
		virtual ~IGraphicsResource() = default;
	};

	template <typename T>
	void IGraphicsResource::ApplyVector(std::vector<T>& target, uint32_t size, const T* data)
	{
		if (size == 0 || data == nullptr) return;
		target.resize(size);
		memcpy(target.data(), data,
			sizeof(T) * size);
	}

	class Fence final : public IGraphicsResource
	{
		VkFence m_vkFence = VK_NULL_HANDLE;
		VkFenceCreateFlags m_flags = {};
	public:
		explicit Fence(const VkFenceCreateInfo& vkFenceCreateInfo);
		~Fence() override;

		[[nodiscard]] const VkFence& GetVkFence() const;
	};

	class Semaphore final : public IGraphicsResource
	{
		VkSemaphore m_vkSemaphore = VK_NULL_HANDLE;
		VkSemaphoreCreateFlags m_flags = {};
	public:
		explicit Semaphore(const VkSemaphoreCreateInfo& semaphoreCreateInfo);
		~Semaphore() override;
		[[nodiscard]] const VkSemaphore& GetVkSemaphore() const;
#ifdef _WIN64
		void* GetVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType) const;
#else
		int GetVkSemaphoreHandle(
			VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType) const;
#endif
	};

	class Image final : public IGraphicsResource
	{
		VkImage m_vkImage = VK_NULL_HANDLE;
		VmaAllocation m_vmaAllocation = VK_NULL_HANDLE;
		VmaAllocationInfo m_vmaAllocationInfo = {};

		VkImageCreateFlags       m_flags;
		VkImageType              m_imageType;
		VkFormat                 m_format;
		VkExtent3D               m_extent;
		uint32_t                 m_mipLevels;
		uint32_t                 m_arrayLayers;
		VkSampleCountFlagBits    m_samples;
		VkImageTiling            m_tiling;
		VkImageUsageFlags        m_usage;
		VkSharingMode            m_sharingMode;
		std::vector<uint32_t>	 m_queueFamilyIndices;
		VkImageLayout            m_initialLayout;

		VkImageLayout			 m_layout;

	public:
		[[nodiscard]] uint32_t GetMipLevels() const ;
		explicit Image(VkImageCreateInfo imageCreateInfo);
		Image(VkImageCreateInfo imageCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo);
		bool HasStencilComponent() const;
		~Image() override;
		void TransitImageLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout);
		void CopyFromBuffer(VkCommandBuffer commandBuffer, const VkBuffer& srcBuffer, VkDeviceSize srcOffset = 0) const;

		void GenerateMipmaps(VkCommandBuffer commandBuffer);

		[[nodiscard]] VkImage GetVkImage() const;
		[[nodiscard]] VkFormat GetFormat() const;
		[[nodiscard]] VmaAllocation GetVmaAllocation() const;
		[[nodiscard]] VkExtent3D GetExtent() const;
		[[nodiscard]] VkImageLayout GetLayout() const;
		[[nodiscard]] const VmaAllocationInfo& GetVmaAllocationInfo() const;

#ifdef _WIN64
		void* GetVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType) const;
#else
		int GetVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType) const;
#endif
	};

	class ImageView final : public IGraphicsResource
	{
		VkImageView m_vkImageView = VK_NULL_HANDLE;

		VkImageViewCreateFlags     m_flags;
		std::shared_ptr<Image>	   m_image;
		VkImageViewType            m_viewType;
		VkFormat                   m_format;
		VkComponentMapping         m_components;
		VkImageSubresourceRange    m_subresourceRange;
		friend class Swapchain;
		friend class Graphics;

	public:
		explicit ImageView(const VkImageViewCreateInfo& imageViewCreateInfo);
		explicit ImageView(const VkImageViewCreateInfo& imageViewCreateInfo, const std::shared_ptr<Image>& image);
		~ImageView() override;
		[[nodiscard]] VkImageView GetVkImageView() const;

		[[nodiscard]] const std::shared_ptr<Image>& GetImage() const;
	};

	class Swapchain final : public IGraphicsResource
	{
		VkSwapchainKHR m_vkSwapchain = VK_NULL_HANDLE;
		std::vector<VkImage> m_vkImages;

		VkSwapchainCreateFlagsKHR m_flags;
		VkSurfaceKHR m_surface;
		uint32_t m_minImageCount;
		VkFormat                         m_imageFormat;
		VkColorSpaceKHR                  m_imageColorSpace;
		VkExtent2D                       m_imageExtent;
		uint32_t                         m_imageArrayLayers;
		VkImageUsageFlags                m_imageUsage;
		VkSharingMode                    m_imageSharingMode;
		std::vector<uint32_t>			m_queueFamilyIndices;
		VkSurfaceTransformFlagBitsKHR    m_preTransform;
		VkCompositeAlphaFlagBitsKHR      m_compositeAlpha;
		VkPresentModeKHR                 m_presentMode;
		VkBool32                         m_clipped;

		std::vector<std::shared_ptr<ImageView>> m_vkImageViews;
	public:
		explicit Swapchain(const VkSwapchainCreateInfoKHR& swapchainCreateInfo);
		~Swapchain() override;

		[[nodiscard]] VkSwapchainKHR GetVkSwapchain() const;

		[[nodiscard]] const std::vector<VkImage>& GetAllVkImages() const;
		[[nodiscard]] const VkImage &GetVkImage() const;
		[[nodiscard]] const VkImageView& GetVkImageView() const;
		[[nodiscard]] const std::vector<std::shared_ptr<ImageView>>& GetAllImageViews() const;

		[[nodiscard]] VkFormat GetImageFormat() const;

		[[nodiscard]] VkExtent2D GetImageExtent() const;
	};

	class ShaderModule final : public IGraphicsResource
	{
		VkShaderModule m_vkShaderModule = VK_NULL_HANDLE;

	public:
		~ShaderModule() override;

		ShaderModule(const VkShaderModuleCreateInfo& createInfo);

		[[nodiscard]] VkShaderModule GetVkShaderModule() const;
	};




	class PipelineLayout final : public IGraphicsResource
	{
		VkPipelineLayout m_vkPipelineLayout = VK_NULL_HANDLE;

		VkPipelineLayoutCreateFlags     m_flags;
		std::vector<VkDescriptorSetLayout> m_setLayouts;
		std::vector<VkPushConstantRange> m_pushConstantRanges;
	public:
		PipelineLayout(const VkPipelineLayoutCreateInfo& pipelineLayoutCreateInfo);
		~PipelineLayout() override;

		[[nodiscard]] VkPipelineLayout GetVkPipelineLayout() const;
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

		VkBufferCreateFlags    m_flags = {};
		VkDeviceSize           m_size = {};
		VkBufferUsageFlags     m_usage = {};
		VkSharingMode          m_sharingMode = {};
		std::vector<uint32_t>	m_queueFamilyIndices = {};
		VmaAllocationCreateInfo m_vmaAllocationCreateInfo = {};

		void UploadData(size_t size, const void* src);
		void DownloadData(size_t size, void* dst);
		void Allocate(VkBufferCreateInfo bufferCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo);
	public:
		explicit Buffer(size_t stagingBufferSize, bool randomAccess = false);
		explicit Buffer(const VkBufferCreateInfo& bufferCreateInfo);
		~Buffer() override;
		Buffer(const VkBufferCreateInfo& bufferCreateInfo, const VmaAllocationCreateInfo& vmaAllocationCreateInfo);
		void Resize(VkDeviceSize newSize);
		template<typename T>
		void UploadVector(const std::vector<T>& data);
		template<typename T>
		void Upload(const T& data);
		template<typename T>
		void DownloadVector(std::vector<T>& data, size_t elementSize);
		template<typename T>
		void Download(T& data);
		void CopyFromBuffer(const Buffer& srcBuffer, VkDeviceSize size, VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0);
		void CopyFromImage(Image& srcImage, const VkBufferImageCopy& imageCopyInfo) const;
		void CopyFromImage(Image& srcImage);
		[[nodiscard]] const VkBuffer& GetVkBuffer() const;

		[[nodiscard]] VmaAllocation GetVmaAllocation() const;

		[[nodiscard]] const VmaAllocationInfo& GetVmaAllocationInfo() const;
	};

	template <typename T>
	void Buffer::UploadVector(const std::vector<T>& data)
	{
		if (data.empty()) return;
		const T* address = data.data();
		UploadData(data.size() * sizeof(T), static_cast<const void*>(address));
	}

	template <typename T>
	void Buffer::Upload(const T& data)
	{
		UploadData(sizeof(T), static_cast<const void*>(&data));
	}

	template <typename T>
	void Buffer::DownloadVector(std::vector<T>& data, size_t elementSize)
	{
		data.resize(elementSize);
		T* address = data.data();
		DownloadData(data.size() * sizeof(T), address);
	}

	template <typename T>
	void Buffer::Download(T& data)
	{
		DownloadData(sizeof(T), static_cast<void*>(&data));
	}

	class Sampler final : public IGraphicsResource
	{
		VkSampler m_vkSampler;
	public:
		explicit Sampler(const VkSamplerCreateInfo& samplerCreateInfo);
		~Sampler() override;
		[[nodiscard]] VkSampler GetVkSampler() const;
	};

	struct DescriptorBinding
	{
		VkDescriptorSetLayoutBinding m_binding;
		VkDescriptorBindingFlags m_bindingFlags;
	};
	class DescriptorSetLayout final : public IGraphicsResource
	{
		friend class DescriptorSet;
		std::unordered_map<uint32_t, DescriptorBinding> m_descriptorSetLayoutBindings;
		VkDescriptorSetLayout m_vkDescriptorSetLayout = VK_NULL_HANDLE;
	public:
		~DescriptorSetLayout() override;
		[[nodiscard]] const VkDescriptorSetLayout& GetVkDescriptorSetLayout() const;

		void PushDescriptorBinding(uint32_t bindingIndex, VkDescriptorType type, VkShaderStageFlags stageFlags, VkDescriptorBindingFlags bindingFlags, uint32_t descriptorCount = 1);
		void Initialize();
	};

	class DescriptorSet final : public IGraphicsResource
	{
		std::shared_ptr<DescriptorSetLayout> m_descriptorSetLayout;
		VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
	public:
		[[nodiscard]] const VkDescriptorSet& GetVkDescriptorSet() const;
		~DescriptorSet() override;
		DescriptorSet(const std::shared_ptr<DescriptorSetLayout>& targetLayout);
		/**
		 * \brief UpdateImageDescriptorBinding
		 * \param bindingIndex Target binding
		 * \param imageInfos The image info for update. Make sure the size is max frame size.
		 */
		void UpdateImageDescriptorBinding(uint32_t bindingIndex, const VkDescriptorImageInfo& imageInfo, uint32_t arrayElement = 0) const;
		void UpdateBufferDescriptorBinding(uint32_t bindingIndex, const VkDescriptorBufferInfo& bufferInfo, uint32_t arrayElement = 0) const;
	};

	class DescriptorPool final : public IGraphicsResource
	{
		VkDescriptorPool m_vkDescriptorPool = VK_NULL_HANDLE;
	public:
		explicit DescriptorPool(const VkDescriptorPoolCreateInfo& descriptorPoolCreateInfo);
		~DescriptorPool() override;
		[[nodiscard]] VkDescriptorPool GetVkDescriptorPool() const;
	};

	class ShaderEXT final : public IGraphicsResource
	{
		VkShaderEXT m_shaderExt = VK_NULL_HANDLE;

		VkShaderCreateFlagsEXT          m_flags;
		VkShaderStageFlagBits           m_stage;
		VkShaderStageFlags              m_nextStage;
		VkShaderCodeTypeEXT             m_codeType;
		std::string						m_name;
		std::vector<VkDescriptorSetLayout> m_setLayouts;
		std::vector<VkPushConstantRange>	m_pushConstantRanges;
		std::optional<VkSpecializationInfo> m_specializationInfo;
	public:
		explicit ShaderEXT(const VkShaderCreateInfoEXT& shaderCreateInfoExt);
		~ShaderEXT() override;
		[[nodiscard]] const VkShaderEXT& GetVkShaderEXT() const;
	};

	enum class CommandBufferStatus
	{
		Ready,
		Recording,
		Recorded,
		Invalid
	};
	class CommandBuffer
	{
		friend class Graphics;
		CommandBufferStatus m_status = CommandBufferStatus::Invalid;
		VkCommandBuffer m_vkCommandBuffer = VK_NULL_HANDLE;
	public:
		CommandBufferStatus GetStatus() const;
		void Allocate(const VkQueueFlagBits& queueType = VK_QUEUE_GRAPHICS_BIT,
			const VkCommandBufferLevel& bufferLevel = VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		void Free();
		[[nodiscard]] const VkCommandBuffer& GetVkCommandBuffer() const;
		/**
		 * Begins the recording state for this command buffer.
		 * @param usage How this command buffer will be used.
		 */
		void Begin(const VkCommandBufferUsageFlags& usage = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

		/**
		 * Ends the recording state for this command buffer.
		 */
		void End();
		/**
		 * Submits the command buffer to the queue and will hold the current thread idle until it has finished.
		 */
		void SubmitIdle();

		/**
		 * Submits the command buffer.
		 * @param waitSemaphore A optional semaphore that will waited upon before the command buffer is executed.
		 * @param signalSemaphore A optional that is signaled once the command buffer has been executed.
		 * @param fence A optional fence that is signaled once the command buffer has completed.
		 */
		void Submit(const VkSemaphore& waitSemaphore = VK_NULL_HANDLE, const VkSemaphore& signalSemaphore = VK_NULL_HANDLE, VkFence fence = VK_NULL_HANDLE);

		void Reset();
	};
}
