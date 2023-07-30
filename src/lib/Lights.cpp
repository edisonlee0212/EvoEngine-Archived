#include "Lights.hpp"

#include "Application.hpp"
#include "Graphics.hpp"
#include "RenderLayer.hpp"
#include "Serialization.hpp"
using namespace EvoEngine;


void SpotLight::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
    ImGui::Checkbox("Cast Shadow", &m_castShadow);
    ImGui::ColorEdit3("Color", &m_diffuse[0]);
    ImGui::DragFloat("Intensity", &m_diffuseBrightness, 0.1f, 0.0f, 999.0f);
    ImGui::DragFloat("Bias", &m_bias, 0.001f, 0.0f, 999.0f);

    ImGui::DragFloat("Constant", &m_constant, 0.01f, 0.0f, 999.0f);
    ImGui::DragFloat("Linear", &m_linear, 0.001f, 0, 1, "%.3f");
    ImGui::DragFloat("Quadratic", &m_quadratic, 0.001f, 0, 10, "%.4f");

    ImGui::DragFloat("Inner Degrees", &m_innerDegrees, 0.1f, 0.0f, m_outerDegrees);
    ImGui::DragFloat("Outer Degrees", &m_outerDegrees, 0.1f, m_innerDegrees, 180.0f);
    ImGui::DragFloat("Light Size", &m_lightSize, 0.01f, 0.0f, 999.0f);
}

void SpotLight::OnCreate()
{
    SetEnabled(true);
}

void SpotLight::Serialize(YAML::Emitter& out)
{
    out << YAML::Key << "m_castShadow" << YAML::Value << m_castShadow;
    out << YAML::Key << "m_innerDegrees" << YAML::Value << m_innerDegrees;
    out << YAML::Key << "m_outerDegrees" << YAML::Value << m_outerDegrees;
    out << YAML::Key << "m_constant" << YAML::Value << m_constant;
    out << YAML::Key << "m_linear" << YAML::Value << m_linear;
    out << YAML::Key << "m_quadratic" << YAML::Value << m_quadratic;
    out << YAML::Key << "m_bias" << YAML::Value << m_bias;
    out << YAML::Key << "m_diffuse" << YAML::Value << m_diffuse;
    out << YAML::Key << "m_diffuseBrightness" << YAML::Value << m_diffuseBrightness;
    out << YAML::Key << "m_lightSize" << YAML::Value << m_lightSize;
}

void SpotLight::Deserialize(const YAML::Node& in)
{
    m_castShadow = in["m_castShadow"].as<bool>();
    m_innerDegrees = in["m_innerDegrees"].as<float>();
    m_outerDegrees = in["m_outerDegrees"].as<float>();
    m_constant = in["m_constant"].as<float>();
    m_linear = in["m_linear"].as<float>();
    m_quadratic = in["m_quadratic"].as<float>();
    m_bias = in["m_bias"].as<float>();
    m_diffuse = in["m_diffuse"].as<glm::vec3>();
    m_diffuseBrightness = in["m_diffuseBrightness"].as<float>();
    m_lightSize = in["m_lightSize"].as<float>();
}

float PointLight::GetFarPlane() const
{
    float lightMax = glm::max(glm::max(m_diffuse.x, m_diffuse.y), m_diffuse.z);
    return (-m_linear + glm::sqrt(m_linear * m_linear - 4 * m_quadratic * (m_constant - (256.0 / 5.0) * lightMax)))
        / (2 * m_quadratic);
}

float SpotLight::GetFarPlane() const
{
    float lightMax = glm::max(glm::max(m_diffuse.x, m_diffuse.y), m_diffuse.z);
    return (-m_linear + glm::sqrt(m_linear * m_linear - 4 * m_quadratic * (m_constant - (256.0 / 5.0) * lightMax)))
        / (2 * m_quadratic);
}

void PointLight::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
    ImGui::Checkbox("Cast Shadow", &m_castShadow);
    ImGui::ColorEdit3("Color", &m_diffuse[0]);
    ImGui::DragFloat("Intensity", &m_diffuseBrightness, 0.1f, 0.0f, 999.0f);
    ImGui::DragFloat("Bias", &m_bias, 0.001f, 0.0f, 999.0f);

    ImGui::DragFloat("Constant", &m_constant, 0.01f, 0.0f, 999.0f);
    ImGui::DragFloat("Linear", &m_linear, 0.0001f, 0, 1, "%.4f");
    ImGui::DragFloat("Quadratic", &m_quadratic, 0.00001f, 0, 10, "%.5f");

    // ImGui::InputFloat("Normal Offset", &dl->normalOffset, 0.01f);
    ImGui::DragFloat("Light Size", &m_lightSize, 0.01f, 0.0f, 999.0f);
}

void PointLight::OnCreate()
{
    SetEnabled(true);
}

void PointLight::Serialize(YAML::Emitter& out)
{
    out << YAML::Key << "m_castShadow" << YAML::Value << m_castShadow;
    out << YAML::Key << "m_constant" << YAML::Value << m_constant;
    out << YAML::Key << "m_linear" << YAML::Value << m_linear;
    out << YAML::Key << "m_quadratic" << YAML::Value << m_quadratic;
    out << YAML::Key << "m_bias" << YAML::Value << m_bias;
    out << YAML::Key << "m_diffuse" << YAML::Value << m_diffuse;
    out << YAML::Key << "m_diffuseBrightness" << YAML::Value << m_diffuseBrightness;
    out << YAML::Key << "m_lightSize" << YAML::Value << m_lightSize;
}

void PointLight::Deserialize(const YAML::Node& in)
{
    m_castShadow = in["m_castShadow"].as<bool>();
    m_constant = in["m_constant"].as<float>();
    m_linear = in["m_linear"].as<float>();
    m_quadratic = in["m_quadratic"].as<float>();
    m_bias = in["m_bias"].as<float>();
    m_diffuse = in["m_diffuse"].as<glm::vec3>();
    m_diffuseBrightness = in["m_diffuseBrightness"].as<float>();
    m_lightSize = in["m_lightSize"].as<float>();
}

void DirectionalLight::OnCreate()
{
    SetEnabled(true);
}

void DirectionalLight::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
    ImGui::Checkbox("Cast Shadow", &m_castShadow);
    ImGui::ColorEdit3("Color", &m_diffuse[0]);
    ImGui::DragFloat("Intensity", &m_diffuseBrightness, 0.1f, 0.0f, 999.0f);
    ImGui::DragFloat("Bias", &m_bias, 0.001f, 0.0f, 999.0f);
    ImGui::DragFloat("Normal Offset", &m_normalOffset, 0.001f, 0.0f, 999.0f);
    ImGui::DragFloat("Light Size", &m_lightSize, 0.01f, 0.0f, 999.0f);
}

void DirectionalLight::Serialize(YAML::Emitter& out)
{
    out << YAML::Key << "m_castShadow" << YAML::Value << m_castShadow;
    out << YAML::Key << "m_bias" << YAML::Value << m_bias;
    out << YAML::Key << "m_diffuse" << YAML::Value << m_diffuse;
    out << YAML::Key << "m_diffuseBrightness" << YAML::Value << m_diffuseBrightness;
    out << YAML::Key << "m_lightSize" << YAML::Value << m_lightSize;
    out << YAML::Key << "m_normalOffset" << YAML::Value << m_normalOffset;
}

void DirectionalLight::Deserialize(const YAML::Node& in)
{
    m_castShadow = in["m_castShadow"].as<bool>();
    m_bias = in["m_bias"].as<float>();
    m_diffuse = in["m_diffuse"].as<glm::vec3>();
    m_diffuseBrightness = in["m_diffuseBrightness"].as<float>();
    m_lightSize = in["m_lightSize"].as<float>();
    m_normalOffset = in["m_normalOffset"].as<float>();
}
void DirectionalLight::PostCloneAction(const std::shared_ptr<IPrivateComponent>& target)
{
}

void PointLight::PostCloneAction(const std::shared_ptr<IPrivateComponent>& target)
{
}

void SpotLight::PostCloneAction(const std::shared_ptr<IPrivateComponent>& target)
{
}

ShadowMaps::ShadowMaps()
{
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = Graphics::GetDescriptorPool()->GetVkDescriptorPool();
    const auto renderLayer = Application::GetLayer<RenderLayer>();

    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &renderLayer->m_lightingLayout->GetVkDescriptorSetLayout();
    if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, &m_lightingDescriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }
}

void ShadowMaps::Initialize()
{
    /*
    m_environmentalBRDFSampler.reset();
    m_environmentalBRDFView.reset();
    m_environmentalBRDFLut.reset();

    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = 512;
        imageInfo.extent.height = 512;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_R16G16_SFLOAT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        m_environmentalBRDFLut = std::make_shared<Image>(imageInfo);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_environmentalBRDFLut->GetVkImage();
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        viewInfo.format = Graphics::ImageFormats::m_gBufferDepth;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        m_environmentalBRDFView = std::make_shared<ImageView>(viewInfo);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        m_environmentalBRDFSampler = std::make_shared<Sampler>(samplerInfo);
    }
    */
    m_directionalShadowMapSampler.reset();
    m_directionalLightShadowMapView.reset();
    m_directionalLightShadowMap.reset();

    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_3D;
        imageInfo.extent.width = Graphics::StorageSizes::m_directionalLightShadowMapResolution;
        imageInfo.extent.height = Graphics::StorageSizes::m_directionalLightShadowMapResolution;
        imageInfo.extent.depth = 4;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = Graphics::ImageFormats::m_shadowMap;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        m_directionalLightShadowMap = std::make_shared<Image>(imageInfo);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_directionalLightShadowMap->GetVkImage();
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        viewInfo.format = Graphics::ImageFormats::m_shadowMap;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        m_directionalLightShadowMapView = std::make_shared<ImageView>(viewInfo);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        m_directionalShadowMapSampler = std::make_shared<Sampler>(samplerInfo);
    }

    m_pointLightShadowMapSampler.reset();
    m_pointLightShadowMapView.reset();
    m_pointLightShadowMap.reset();

    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_3D;
        imageInfo.extent.width = Graphics::StorageSizes::m_pointLightShadowMapResolution;
        imageInfo.extent.height = Graphics::StorageSizes::m_pointLightShadowMapResolution;
        imageInfo.extent.depth = 6;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = Graphics::ImageFormats::m_shadowMap;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        m_pointLightShadowMap = std::make_shared<Image>(imageInfo);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_pointLightShadowMap->GetVkImage();
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        viewInfo.format = Graphics::ImageFormats::m_shadowMap;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        m_pointLightShadowMapView = std::make_shared<ImageView>(viewInfo);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        m_pointLightShadowMapSampler = std::make_shared<Sampler>(samplerInfo);
    }

    m_spotLightShadowMapSampler.reset();
    m_spotLightShadowMapView.reset();
    m_spotLightShadowMap.reset();

    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = Graphics::StorageSizes::m_spotLightShadowMapResolution;
        imageInfo.extent.height = Graphics::StorageSizes::m_spotLightShadowMapResolution;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = Graphics::ImageFormats::m_shadowMap;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        m_spotLightShadowMap = std::make_shared<Image>(imageInfo);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_spotLightShadowMap->GetVkImage();
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = Graphics::ImageFormats::m_shadowMap;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        m_spotLightShadowMapView = std::make_shared<ImageView>(viewInfo);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = Graphics::GetVkPhysicalDeviceProperties().limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        m_spotLightShadowMapSampler = std::make_shared<Sampler>(samplerInfo);
    }


    {
        VkDescriptorImageInfo imageInfo{};
        VkWriteDescriptorSet writeInfo{};
        
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        //imageInfo.imageView = m_environmentalBRDFView->GetVkImageView();
        //imageInfo.sampler = m_environmentalBRDFSampler->GetVkSampler();

        writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeInfo.dstSet = m_lightingDescriptorSet;
        writeInfo.dstBinding = 18;
        writeInfo.dstArrayElement = 0;
        writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeInfo.descriptorCount = 1;
        writeInfo.pImageInfo = &imageInfo;
        //vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
        imageInfo.imageView = m_directionalLightShadowMapView->GetVkImageView();
        imageInfo.sampler = m_directionalShadowMapSampler->GetVkSampler();
        writeInfo.dstBinding = 19;
        vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
        imageInfo.imageView = m_pointLightShadowMapView->GetVkImageView();
        imageInfo.sampler = m_pointLightShadowMapSampler->GetVkSampler();
        writeInfo.dstBinding = 20;
        vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
        imageInfo.imageView = m_spotLightShadowMapView->GetVkImageView();
        imageInfo.sampler = m_spotLightShadowMapSampler->GetVkSampler();
        writeInfo.dstBinding = 21;
        vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
    }
}

VkRenderingAttachmentInfo ShadowMaps::GetDirectionalLightDepthAttachmentInfo() const
{
    VkRenderingAttachmentInfo attachment{};
    attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

    attachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    attachment.clearValue.depthStencil.depth = 1.0f;
    attachment.imageView = m_directionalLightShadowMapView->GetVkImageView();
    return attachment;
}

VkRenderingAttachmentInfo ShadowMaps::GetPointLightDepthAttachmentInfo() const
{
    VkRenderingAttachmentInfo attachment{};
    attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

    attachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    attachment.clearValue.depthStencil.depth = 1.0f;
    attachment.imageView = m_pointLightShadowMapView->GetVkImageView();
    return attachment;
}

VkRenderingAttachmentInfo ShadowMaps::GetSpotLightDepthAttachmentInfo() const
{
    VkRenderingAttachmentInfo attachment{};
    attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

    attachment.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    attachment.clearValue.depthStencil.depth = 1.0f;
    attachment.imageView = m_spotLightShadowMapView->GetVkImageView();
    return attachment;
}
