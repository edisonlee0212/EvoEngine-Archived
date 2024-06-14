#include "Graphics.hpp"
#include "Utilities.hpp"
using namespace evo_engine;

void Graphics::PrepareDescriptorSetLayouts() const
{
	const auto renderTexturePresent = std::make_shared<DescriptorSetLayout>();
	renderTexturePresent->PushDescriptorBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	renderTexturePresent->Initialize();
	RegisterDescriptorSetLayout("RENDER_TEXTURE_PRESENT_LAYOUT", renderTexturePresent);

	const auto perFrameLayout = std::make_shared<DescriptorSetLayout>();
	perFrameLayout->PushDescriptorBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	perFrameLayout->PushDescriptorBinding(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_MESH_BIT_EXT, 0);
	perFrameLayout->PushDescriptorBinding(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_MESH_BIT_EXT, 0);

	perFrameLayout->PushDescriptorBinding(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_MESH_BIT_EXT, 0);
	perFrameLayout->PushDescriptorBinding(13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, Settings::MAX_TEXTURE_2D_RESOURCE_SIZE);
	perFrameLayout->PushDescriptorBinding(14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, Settings::MAX_CUBEMAP_RESOURCE_SIZE);

	perFrameLayout->Initialize();
	RegisterDescriptorSetLayout("PER_FRAME_LAYOUT", perFrameLayout);

	const auto lightLayout = std::make_shared<DescriptorSetLayout>();
	lightLayout->PushDescriptorBinding(15, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	lightLayout->PushDescriptorBinding(16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	lightLayout->PushDescriptorBinding(17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	lightLayout->Initialize();
	RegisterDescriptorSetLayout("LIGHTING_LAYOUT", lightLayout);

	const auto boneMatricesLayout = std::make_shared<DescriptorSetLayout>();
	boneMatricesLayout->PushDescriptorBinding(18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
	boneMatricesLayout->Initialize();
	RegisterDescriptorSetLayout("BONE_MATRICES_LAYOUT", boneMatricesLayout);

	const auto instancedDataLayout = std::make_shared<DescriptorSetLayout>();
	instancedDataLayout->PushDescriptorBinding(18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_ALL, 0);
	instancedDataLayout->Initialize();
	RegisterDescriptorSetLayout("INSTANCED_DATA_LAYOUT", instancedDataLayout);

	const auto cameraGBufferLayout = std::make_shared<DescriptorSetLayout>();
	cameraGBufferLayout->PushDescriptorBinding(18, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	cameraGBufferLayout->PushDescriptorBinding(19, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	cameraGBufferLayout->PushDescriptorBinding(20, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	cameraGBufferLayout->PushDescriptorBinding(21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	cameraGBufferLayout->Initialize();
	RegisterDescriptorSetLayout("CAMERA_GBUFFER_LAYOUT", cameraGBufferLayout);

	const auto SSRReflectLayout = std::make_shared<DescriptorSetLayout>();
	SSRReflectLayout->PushDescriptorBinding(18, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	SSRReflectLayout->PushDescriptorBinding(19, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	SSRReflectLayout->PushDescriptorBinding(20, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	SSRReflectLayout->PushDescriptorBinding(21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	SSRReflectLayout->Initialize();
	RegisterDescriptorSetLayout("SSR_REFLECT_LAYOUT", SSRReflectLayout);

	const auto SSRBlur = std::make_shared<DescriptorSetLayout>();
	SSRBlur->PushDescriptorBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	SSRBlur->Initialize();
	RegisterDescriptorSetLayout("SSR_BLUR_LAYOUT", SSRBlur);

	const auto SSRCombineLayout = std::make_shared<DescriptorSetLayout>();
	SSRCombineLayout->PushDescriptorBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	SSRCombineLayout->PushDescriptorBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	SSRCombineLayout->Initialize();
	RegisterDescriptorSetLayout("SSR_COMBINE_LAYOUT", SSRCombineLayout);

}
