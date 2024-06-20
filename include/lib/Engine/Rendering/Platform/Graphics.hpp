#pragma once
#include "GraphicsPipeline.hpp"
#include "GraphicsResources.hpp"
#include "ISingleton.hpp"

namespace evo_engine {
struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;
  [[nodiscard]] bool IsComplete() const {
    return graphics_family.has_value() && present_family.has_value();
  }
};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

class Graphics final : public ISingleton<Graphics> {
  friend class Application;
  friend class Resources;
  friend class Lighting;
  friend class PointLightShadowMap;
  friend class SpotLightShadowMap;
#pragma region Vulkan
  VkInstance vk_instance_ = VK_NULL_HANDLE;
  std::vector<std::string> required_device_extensions_ = {};
  std::vector<std::string> required_layers_ = {};
  std::vector<VkExtensionProperties> vk_extensions_;
  std::vector<VkLayerProperties> vk_layers_;
  VkDebugUtilsMessengerEXT vk_debug_messenger_ = {};
  VkPhysicalDeviceFeatures vk_physical_device_features_ = {};
  VkPhysicalDeviceProperties vk_physical_device_properties_ = {};
  VkPhysicalDeviceProperties2 vk_physical_device_properties2_ = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceVulkan11Properties vk_physical_device_vulkan11_properties_{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
  VkPhysicalDeviceVulkan12Properties vk_physical_device_vulkan12_properties_{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
  VkPhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties_ext_ = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
  VkPhysicalDeviceSubgroupSizeControlProperties subgroup_size_control_properties_ = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};

  VkPhysicalDeviceMemoryProperties vk_physical_device_memory_properties_ = {};
  VkSurfaceKHR vk_surface_ = VK_NULL_HANDLE;
  VkPhysicalDevice vk_physical_device_ = VK_NULL_HANDLE;
  VkDevice vk_device_ = VK_NULL_HANDLE;

  VmaAllocator vma_allocator_ = VK_NULL_HANDLE;

  QueueFamilyIndices queue_family_indices_ = {};

  VkQueue vk_graphics_queue_ = VK_NULL_HANDLE;
  VkQueue vk_present_queue_ = VK_NULL_HANDLE;

  std::unique_ptr<Swapchain> swapchain_ = {};

  VkSurfaceFormatKHR vk_surface_format_ = {};

#pragma endregion
#pragma region Internals
  std::unique_ptr<CommandPool> command_pool_ = {};
  std::unique_ptr<DescriptorPool> descriptor_pool_ = {};

  int max_frame_in_flight_ = 2;

  int used_command_buffer_size_ = 0;
  std::vector<std::vector<CommandBuffer>> command_buffer_pool_ = {};

  std::vector<std::unique_ptr<Semaphore>> image_available_semaphores_ = {};
  std::vector<std::unique_ptr<Semaphore>> render_finished_semaphores_ = {};
  std::vector<std::unique_ptr<Fence>> in_flight_fences_ = {};

  uint32_t current_frame_index_ = 0;

  uint32_t next_image_index_ = 0;

#pragma endregion
#pragma region Shader related
  std::string shader_basic_;
  std::string shader_ssr_constants_;
  std::string shader_basic_constants_;
  std::string shader_gizmos_constants_;
  std::string shader_lighting_;
  std::string shader_skybox_;
  size_t max_bone_amount_ = 65536;
  size_t max_shadow_cascade_amount_ = 4;
  friend class RenderLayer;

#pragma endregion

  QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice physical_device) const;
  SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice physical_device) const;
  bool IsDeviceSuitable(VkPhysicalDevice physical_device,
                        const std::vector<std::string>& required_device_extensions) const;

  void CreateInstance();
  void CreateSurface();
  void CreateDebugMessenger();
  void SelectPhysicalDevice();
  void CreateLogicalDevice();
  void SetupVmaAllocator();

  void CreateSwapChain();

  void CreateSwapChainSyncObjects();

  void RecreateSwapChain();

  void OnDestroy();
  void SwapChainSwapImage();
  void SubmitPresent();
  void WaitForCommandsComplete() const;
  void Submit();

  void ResetCommandBuffers();
  static void Initialize();
  static void PostResourceLoadingInitialization();
  static void Destroy();
  static void PreUpdate();
  static void LateUpdate();

  bool recreate_swap_chain_ = false;
  unsigned swapchain_version_ = 0;
  static uint32_t FindMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties);

  std::unordered_map<std::string, std::shared_ptr<GraphicsPipeline>> graphics_pipelines_;
  std::unordered_map<std::string, std::shared_ptr<DescriptorSetLayout>> descriptor_set_layouts_;
  void CreateGraphicsPipelines() const;
  static void PrepareDescriptorSetLayouts();

 public:
  double cpu_wait_time = 0.0f;
  static void WaitForDeviceIdle();

  static void RegisterGraphicsPipeline(const std::string& name,
                                       const std::shared_ptr<GraphicsPipeline>& graphics_pipeline);
  [[nodiscard]] static const std::shared_ptr<GraphicsPipeline>& GetGraphicsPipeline(const std::string& name);
  static void RegisterDescriptorSetLayout(const std::string& name,
                                          const std::shared_ptr<DescriptorSetLayout>& descriptor_set_layout);
  [[nodiscard]] static const std::shared_ptr<DescriptorSetLayout>& GetDescriptorSetLayout(const std::string& name);

  std::vector<size_t> triangles;
  std::vector<size_t> strands_segments;
  std::vector<size_t> draw_call;

  class Settings {
   public:
    inline static bool use_mesh_shader = false;
    inline static uint32_t directional_light_shadow_map_resolution = 2048;
    inline static uint32_t point_light_shadow_map_resolution = 1024;
    inline static uint32_t spot_light_shadow_map_resolution = 1024;
    inline static uint32_t max_texture_2d_resource_size = 2048;
    inline static uint32_t max_cubemap_resource_size = 256;

    inline static uint32_t max_directional_light_size = 16;
    inline static uint32_t max_point_light_size = 16;
    inline static uint32_t max_spot_light_size = 16;
  };

  class Constants {
   public:
    inline static bool enable_mesh_shader = true;
    constexpr static uint32_t initial_descriptor_pool_max_size = 16384;
    constexpr static uint32_t initial_descriptor_pool_max_sets = 16384;
    constexpr static uint32_t initial_camera_size = 1;
    constexpr static uint32_t initial_material_size = 1;
    constexpr static uint32_t initial_instance_size = 1;
    constexpr static uint32_t initial_render_task_size = 1;
    constexpr static uint32_t max_kernel_amount = 64;

    constexpr static VkFormat texture_2d = VK_FORMAT_R32G32B32A32_SFLOAT;
    constexpr static VkFormat render_texture_depth = VK_FORMAT_D32_SFLOAT;
    constexpr static VkFormat render_texture_color = VK_FORMAT_R32G32B32A32_SFLOAT;
    constexpr static VkFormat g_buffer_depth = VK_FORMAT_D32_SFLOAT;
    constexpr static VkFormat g_buffer_color = VK_FORMAT_R16G16B16A16_SFLOAT;
    constexpr static VkFormat g_buffer_material = VK_FORMAT_R16G16B16A16_SFLOAT;
    constexpr static VkFormat shadow_map = VK_FORMAT_D32_SFLOAT;
    constexpr static uint32_t meshlet_max_vertices_size = 64;
    constexpr static uint32_t meshlet_max_triangles_size = 40;
  };

  static void EverythingBarrier(VkCommandBuffer command_buffer);

  static void TransitImageLayout(VkCommandBuffer command_buffer, VkImage target_image, VkFormat image_format,
                                 uint32_t layer_count, VkImageLayout old_layout, VkImageLayout new_layout,
                                 uint32_t mip_levels = 1);

  static std::string StringifyResultVk(const VkResult& result);
  static void CheckVk(const VkResult& result);

  static size_t GetMaxBoneAmount();
  static size_t GetMaxShadowCascadeAmount();
  static void AppendCommands(const std::function<void(VkCommandBuffer command_buffer)>& action);
  static void ImmediateSubmit(const std::function<void(VkCommandBuffer command_buffer)>& action);
  static QueueFamilyIndices GetQueueFamilyIndices();
  static int GetMaxFramesInFlight();
  static void NotifyRecreateSwapChain();
  static VkInstance GetVkInstance();
  static VkPhysicalDevice GetVkPhysicalDevice();
  static VkDevice GetVkDevice();
  static uint32_t GetCurrentFrameIndex();
  static uint32_t GetNextImageIndex();
  static VkCommandPool GetVkCommandPool();
  static VkQueue GetGraphicsVkQueue();
  static VkQueue GetPresentVkQueue();
  static VmaAllocator GetVmaAllocator();
  static const std::unique_ptr<Swapchain>& GetSwapchain();
  static const std::unique_ptr<DescriptorPool>& GetDescriptorPool();
  static unsigned GetSwapchainVersion();
  static VkSurfaceFormatKHR GetVkSurfaceFormat();
  static const VkPhysicalDeviceProperties& GetVkPhysicalDeviceProperties();
  [[nodiscard]] static bool CheckExtensionSupport(const std::string& extension_name);
  [[nodiscard]] static bool CheckLayerSupport(const std::string& layer_name);
};
}  // namespace evo_engine
