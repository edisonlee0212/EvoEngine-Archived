#include "Material.hpp"

#include "Texture2D.hpp"
#include "EditorLayer.hpp"
#include "RenderLayer.hpp"
using namespace EvoEngine;

static const char* PolygonMode[]{ "Point", "Line", "Fill" };
static const char* CullingMode[]{ "Front", "Back", "FrontAndBack" };
static const char* BlendingFactor[]{ "Zero", "One", "SrcColor", "OneMinusSrcColor", "DstColor", "OneMinusDstColor",
                                    "SrcAlpha", "OneMinusSrcAlpha",
                                    "DstAlpha", "OneMinusDstAlpha", "ConstantColor", "OneMinusConstantColor",
                                    "ConstantAlpha", "OneMinusConstantAlpha", "SrcAlphaSaturate",
                                    "Src1Color", "OneMinusSrc1Color", "Src1Alpha", "OneMinusSrc1Alpha" };

void Material::CollectAssetRef(std::vector<AssetRef>& list) {
    list.push_back(m_albedoTexture);
    list.push_back(m_normalTexture);
    list.push_back(m_metallicTexture);
    list.push_back(m_roughnessTexture);
    list.push_back(m_aoTexture);
}


bool DrawSettings::OnInspect() {
    bool changed = false;
    int polygonMode = 0;
    switch (m_polygonMode)
    {
    case VK_POLYGON_MODE_POINT: polygonMode = 0; break;
    case VK_POLYGON_MODE_LINE: polygonMode = 1; break;
    case VK_POLYGON_MODE_FILL: polygonMode = 2; break;
    }
    if (ImGui::Combo(
        "Polygon Mode",
        &polygonMode,
        PolygonMode,
        IM_ARRAYSIZE(PolygonMode))) {
        changed = true;
        switch (polygonMode)
        {
        case 0: m_polygonMode = VK_POLYGON_MODE_POINT; break;
        case 1: m_polygonMode = VK_POLYGON_MODE_LINE; break;
        case 2: m_polygonMode = VK_POLYGON_MODE_FILL; break;
        }
    }
    if (m_polygonMode == VK_POLYGON_MODE_LINE)
    {
        ImGui::DragFloat("Line width", &m_lineWidth, 0.1f, 0.0f, 100.0f);
    }
    int cullFaceMode = 0;
    switch (m_cullMode)
    {
    case VK_CULL_MODE_FRONT_BIT: cullFaceMode = 0; break;
    case VK_CULL_MODE_BACK_BIT: cullFaceMode = 1; break;
    case VK_CULL_MODE_FRONT_AND_BACK: cullFaceMode = 2; break;
    }
    if (ImGui::Combo(
        "Cull Face Mode",
        &cullFaceMode,
        CullingMode,
        IM_ARRAYSIZE(CullingMode))) {
        changed = true;
        switch (cullFaceMode)
        {
        case 0: m_cullMode = VK_CULL_MODE_FRONT_BIT; break;
        case 1: m_cullMode = VK_CULL_MODE_BACK_BIT; break;
        case 2: m_cullMode = VK_CULL_MODE_FRONT_AND_BACK; break;
        }
    }

    if (ImGui::Checkbox("Blending", &m_blending)) changed = true;

    if (false && m_blending) {
        if (ImGui::Combo(
            "Blending Source Factor",
            reinterpret_cast<int*>(&m_blendingSrcFactor),
            BlendingFactor,
            IM_ARRAYSIZE(BlendingFactor))) {
            changed = true;
        }
        if (ImGui::Combo(
            "Blending Destination Factor",
            reinterpret_cast<int*>(&m_blendingDstFactor),
            BlendingFactor,
            IM_ARRAYSIZE(BlendingFactor))) {
            changed = true;
        }
    }
    return changed;
}

void DrawSettings::ApplySettings(GraphicsGlobalStates& globalPipelineState) const
{
    globalPipelineState.m_cullMode = m_cullMode;
    globalPipelineState.m_polygonMode = m_polygonMode;
    globalPipelineState.m_lineWidth = m_lineWidth;
}

void DrawSettings::Save(const std::string& name, YAML::Emitter& out) const
{
    out << YAML::Key << name << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "m_cullMode" << YAML::Value << m_cullMode;
    out << YAML::Key << "m_lineWidth" << YAML::Value << m_lineWidth;
    out << YAML::Key << "m_polygonMode" << YAML::Value << static_cast<unsigned>(m_polygonMode);
    out << YAML::Key << "m_blending" << YAML::Value << m_blending;
    out << YAML::Key << "m_blendingSrcFactor" << YAML::Value << static_cast<unsigned>(m_blendingSrcFactor);
    out << YAML::Key << "m_blendingDstFactor" << YAML::Value << static_cast<unsigned>(m_blendingDstFactor);
    out << YAML::EndMap;
}

void DrawSettings::Load(const std::string& name, const YAML::Node& in) {
    if (in[name]) {
        const auto& drawSettings = in[name];
        if (drawSettings["m_cullMode"]) m_cullMode = drawSettings["m_cullMode"].as<unsigned>();
        if (drawSettings["m_lineWidth"]) m_lineWidth = drawSettings["m_lineWidth"].as<float>();
        if (drawSettings["m_polygonMode"]) m_polygonMode = static_cast<VkPolygonMode>(drawSettings["m_polygonMode"].as<unsigned>());

        if (drawSettings["m_blending"]) m_blending = drawSettings["m_blending"].as<bool>();
        if (drawSettings["m_blendingSrcFactor"]) m_blendingSrcFactor = static_cast<VkBlendFactor>(drawSettings["m_blendingSrcFactor"].as<unsigned>());
        if (drawSettings["m_blendingDstFactor"]) m_blendingDstFactor = static_cast<VkBlendFactor>(drawSettings["m_blendingDstFactor"].as<unsigned>());
    }
}

void Material::SetAlbedoTexture(const std::shared_ptr<Texture2D>& texture)
{
    m_albedoTexture = texture;
    m_needUpdate = true;
}

void Material::SetNormalTexture(const std::shared_ptr<Texture2D>& texture)
{
    m_normalTexture = texture;
    m_needUpdate = true;
}

void Material::SetMetallicTexture(const std::shared_ptr<Texture2D>& texture)
{
    m_metallicTexture = texture;
    m_needUpdate = true;
}

void Material::SetRoughnessTexture(const std::shared_ptr<Texture2D>& texture)
{
    m_roughnessTexture = texture;
    m_needUpdate = true;
}

void Material::SetAOTexture(const std::shared_ptr<Texture2D>& texture)
{
    m_aoTexture = texture;
    m_needUpdate = true;
}


void Material::OnCreate()
{
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = Graphics::GetDescriptorPool()->GetVkDescriptorPool();
    const auto renderLayer = Application::GetLayer<RenderLayer>();

    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &renderLayer->m_materialLayout->GetVkDescriptorSetLayout();
    if (vkAllocateDescriptorSets(Graphics::GetVkDevice(), &allocInfo, &m_descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }
}

Material::~Material()
{
    if(m_descriptorSet != VK_NULL_HANDLE) vkFreeDescriptorSets(Graphics::GetVkDevice(), Graphics::GetDescriptorPool()->GetVkDescriptorPool(), 1, &m_descriptorSet);
    m_descriptorSet = VK_NULL_HANDLE;
}

void Material::UpdateMaterialInfoBlock(MaterialInfoBlock& materialInfoBlock)
{
	if (const auto albedoTexture = m_albedoTexture.Get<Texture2D>(); albedoTexture && albedoTexture->GetVkSampler())
    {
        materialInfoBlock.m_albedoEnabled = true;
    }else
    {
        materialInfoBlock.m_albedoEnabled = false;
    }
	if (const auto normalTexture = m_normalTexture.Get<Texture2D>(); normalTexture && normalTexture->GetVkSampler())
    {
        materialInfoBlock.m_normalEnabled = true;
    }
    else
    {
        materialInfoBlock.m_normalEnabled = false;
    }
	if (const auto metallicTexture = m_metallicTexture.Get<Texture2D>(); metallicTexture && metallicTexture->GetVkSampler())
    {
        materialInfoBlock.m_metallicEnabled = true;
    }
    else
    {
        materialInfoBlock.m_metallicEnabled = false;
    }
	if (const auto roughnessTexture = m_roughnessTexture.Get<Texture2D>(); roughnessTexture && roughnessTexture->GetVkSampler())
    {
        materialInfoBlock.m_roughnessEnabled = true;
    }
    else
    {
        materialInfoBlock.m_roughnessEnabled = false;
    }
	if (const auto aoTexture = m_aoTexture.Get<Texture2D>(); aoTexture && aoTexture->GetVkSampler())
    {
        materialInfoBlock.m_aoEnabled = true;
    }
    else
    {
        materialInfoBlock.m_aoEnabled = false;
    }
    materialInfoBlock.m_castShadow = true;
    materialInfoBlock.m_subsurfaceColor = { m_materialProperties.m_subsurfaceColor, 0.0f };
    materialInfoBlock.m_subsurfaceRadius = { m_materialProperties.m_subsurfaceRadius, 0.0f };
    materialInfoBlock.m_albedoColorVal = glm::vec4(m_materialProperties.m_albedoColor, m_drawSettings.m_blending ? (1.0f - m_materialProperties.m_transmission) : 1.0f);
    materialInfoBlock.m_metallicVal = m_materialProperties.m_metallic;
    materialInfoBlock.m_roughnessVal = m_materialProperties.m_roughness;
    materialInfoBlock.m_aoVal = 1.0f;
    materialInfoBlock.m_emissionVal = m_materialProperties.m_emission;
    UpdateDescriptorBindings();
}

void Material::UpdateDescriptorBindings(bool forceUpdate)
{
    if(!m_needUpdate) return;
    VkDescriptorImageInfo imageInfo;
    std::vector<VkWriteDescriptorSet> writeInfos;
    VkWriteDescriptorSet writeInfo{};
    auto missingTexture = std::dynamic_pointer_cast<Texture2D>(Resources::GetResource("TEXTURE_MISSING"));
	if(const auto texture = m_albedoTexture.Get<Texture2D>(); texture && texture->GetVkImageView() && texture->GetVkSampler())
    {
        imageInfo.imageLayout = texture->GetLayout();
        imageInfo.imageView = texture->GetVkImageView();
        imageInfo.sampler = texture->GetVkSampler();
    }else
    {
        imageInfo.imageLayout = missingTexture->GetLayout();
        imageInfo.imageView = missingTexture->GetVkImageView();
        imageInfo.sampler = missingTexture->GetVkSampler();
    }
    
    writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeInfo.dstSet = m_descriptorSet;
    writeInfo.dstBinding = 10;
    writeInfo.dstArrayElement = 0;
    writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writeInfo.descriptorCount = 1;
    writeInfo.pImageInfo = &imageInfo;
    writeInfos.emplace_back(writeInfo);
    vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
    if (const auto texture = m_normalTexture.Get<Texture2D>(); texture && texture->GetVkImageView() && texture->GetVkSampler())
    {
        imageInfo.imageLayout = texture->GetLayout();
        imageInfo.imageView = texture->GetVkImageView();
        imageInfo.sampler = texture->GetVkSampler();
    }
    else
    {
        imageInfo.imageLayout = missingTexture->GetLayout();
        imageInfo.imageView = missingTexture->GetVkImageView();
        imageInfo.sampler = missingTexture->GetVkSampler();
    }
    writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeInfo.dstBinding = 11;
    writeInfo.pImageInfo = &imageInfo;
    writeInfos.emplace_back(writeInfo);
    vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
    if (const auto texture = m_metallicTexture.Get<Texture2D>(); texture && texture->GetVkImageView() && texture->GetVkSampler())
    {
        imageInfo.imageLayout = texture->GetLayout();
        imageInfo.imageView = texture->GetVkImageView();
        imageInfo.sampler = texture->GetVkSampler();
    }
    else
    {
        imageInfo.imageLayout = missingTexture->GetLayout();
        imageInfo.imageView = missingTexture->GetVkImageView();
        imageInfo.sampler = missingTexture->GetVkSampler();
    }
    writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeInfo.dstBinding = 12;
    writeInfo.pImageInfo = &imageInfo;
    writeInfos.emplace_back(writeInfo);
    vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
    if (const auto texture = m_roughnessTexture.Get<Texture2D>(); texture && texture->GetVkImageView() && texture->GetVkSampler())
    {
        imageInfo.imageLayout = texture->GetLayout();
        imageInfo.imageView = texture->GetVkImageView();
        imageInfo.sampler = texture->GetVkSampler();
    }
    else
    {
        imageInfo.imageLayout = missingTexture->GetLayout();
        imageInfo.imageView = missingTexture->GetVkImageView();
        imageInfo.sampler = missingTexture->GetVkSampler();
    }
    writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeInfo.dstBinding = 13;
    writeInfo.pImageInfo = &imageInfo;
    writeInfos.emplace_back(writeInfo);
    vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
    if (const auto texture = m_aoTexture.Get<Texture2D>(); texture && texture->GetVkImageView() && texture->GetVkSampler())
    {
        imageInfo.imageLayout = texture->GetLayout();
        imageInfo.imageView = texture->GetVkImageView();
        imageInfo.sampler = texture->GetVkSampler();
    }
    else
    {
        imageInfo.imageLayout = missingTexture->GetLayout();
        imageInfo.imageView = missingTexture->GetVkImageView();
        imageInfo.sampler = missingTexture->GetVkSampler();
    }
    writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeInfo.dstBinding = 14;
    writeInfo.pImageInfo = &imageInfo;
    writeInfos.emplace_back(writeInfo);
    vkUpdateDescriptorSets(Graphics::GetVkDevice(), 1, &writeInfo, 0, nullptr);
    m_needUpdate = false;
    //vkUpdateDescriptorSets(Graphics::GetVkDevice(), writeInfos.size(), writeInfos.data(), 0, nullptr);
}

void Material::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) {
    bool changed = false;
    
    if (ImGui::Checkbox("Vertex color only", &m_vertexColorOnly)) {
        changed = true;
    }

    ImGui::Separator();
    if (ImGui::TreeNodeEx("PBR##Material", ImGuiTreeNodeFlags_DefaultOpen)) {

        if (ImGui::ColorEdit3("Albedo##Material", &m_materialProperties.m_albedoColor.x)) {
            changed = true;
        }
        if (ImGui::DragFloat("Subsurface##Material", &m_materialProperties.m_subsurfaceFactor, 0.01f, 0.0f, 1.0f)) {
            changed = true;
        }
        if (m_materialProperties.m_subsurfaceFactor > 0.0f) {
            if (ImGui::DragFloat3("Subsurface Radius##Material", &m_materialProperties.m_subsurfaceRadius.x, 0.01f, 0.0f, 999.0f)) {
                changed = true;
            }
            if (ImGui::ColorEdit3("Subsurface Color##Material", &m_materialProperties.m_subsurfaceColor.x)) {
                changed = true;
            }
        }
        if (ImGui::DragFloat("Metallic##Material", &m_materialProperties.m_metallic, 0.01f, 0.0f, 1.0f)) {
            changed = true;
        }

        if (ImGui::DragFloat("Specular##Material", &m_materialProperties.m_specular, 0.01f, 0.0f, 1.0f)) {
            changed = true;
        }
        if (ImGui::DragFloat("Specular Tint##Material", &m_materialProperties.m_specularTint, 0.01f, 0.0f, 1.0f)) {
            changed = true;
        }
        if (ImGui::DragFloat("Roughness##Material", &m_materialProperties.m_roughness, 0.01f, 0.0f, 1.0f)) {
            changed = true;
        }
        if (ImGui::DragFloat("Sheen##Material", &m_materialProperties.m_sheen, 0.01f, 0.0f, 1.0f)) {
            changed = true;
        }
        if (ImGui::DragFloat("Sheen Tint##Material", &m_materialProperties.m_sheenTint, 0.01f, 0.0f, 1.0f)) {
            changed = true;
        }
        if (ImGui::DragFloat("Clear Coat##Material", &m_materialProperties.m_clearCoat, 0.01f, 0.0f, 1.0f)) {
            changed = true;
        }
        if (ImGui::DragFloat("Clear Coat Roughness##Material", &m_materialProperties.m_clearCoatRoughness, 0.01f, 0.0f, 1.0f)) {
            changed = true;
        }
        if (ImGui::DragFloat("IOR##Material", &m_materialProperties.m_IOR, 0.01f, 0.0f, 5.0f)) {
            changed = true;
        }
        if (ImGui::DragFloat("Transmission##Material", &m_materialProperties.m_transmission, 0.01f, 0.0f, 1.0f)) {
            changed = true;
        }
        if (ImGui::DragFloat("Transmission Roughness##Material", &m_materialProperties.m_transmissionRoughness, 0.01f, 0.0f, 1.0f)) {
            changed = true;
        }
        if (ImGui::DragFloat("Emission##Material", &m_materialProperties.m_emission, 0.01f, 0.0f, 10.0f)) {
            changed = true;
        }


        ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Others##Material")) {
        if (m_drawSettings.OnInspect()) changed = true;
        ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Textures##Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (editorLayer->DragAndDropButton<Texture2D>(m_albedoTexture, "Albedo Tex")) {
            changed = true;
        }
        if (editorLayer->DragAndDropButton<Texture2D>(m_normalTexture, "Normal Tex")) {
            changed = true;
        }
        if (editorLayer->DragAndDropButton<Texture2D>(m_metallicTexture, "Metallic Tex")) {
            changed = true;
        }
        if (editorLayer->DragAndDropButton<Texture2D>(m_roughnessTexture, "Roughness Tex")) {
            changed = true;
        }
        if (editorLayer->DragAndDropButton<Texture2D>(m_aoTexture, "AO Tex")) {
            changed = true;
        }
        ImGui::TreePop();
    }
    if (changed) {
        m_needUpdate = true;
        m_saved = false;
        m_version++;
    }
}

void Material::Serialize(YAML::Emitter& out) {
    m_albedoTexture.Save("m_albedoTexture", out);
    m_normalTexture.Save("m_normalTexture", out);
    m_metallicTexture.Save("m_metallicTexture", out);
    m_roughnessTexture.Save("m_roughnessTexture", out);
    m_aoTexture.Save("m_aoTexture", out);

    m_drawSettings.Save("m_drawSettings", out);

    out << YAML::Key << "m_materialProperties.m_albedoColor" << YAML::Value << m_materialProperties.m_albedoColor;
    out << YAML::Key << "m_materialProperties.m_subsurfaceColor" << YAML::Value << m_materialProperties.m_subsurfaceColor;
    out << YAML::Key << "m_materialProperties.m_subsurfaceFactor" << YAML::Value << m_materialProperties.m_subsurfaceFactor;
    out << YAML::Key << "m_materialProperties.m_subsurfaceRadius" << YAML::Value << m_materialProperties.m_subsurfaceRadius;

    out << YAML::Key << "m_materialProperties.m_metallic" << YAML::Value << m_materialProperties.m_metallic;
    out << YAML::Key << "m_materialProperties.m_specular" << YAML::Value << m_materialProperties.m_specular;
    out << YAML::Key << "m_materialProperties.m_specularTint" << YAML::Value << m_materialProperties.m_specularTint;
    out << YAML::Key << "m_materialProperties.m_roughness" << YAML::Value << m_materialProperties.m_roughness;
    out << YAML::Key << "m_materialProperties.m_sheen" << YAML::Value << m_materialProperties.m_sheen;
    out << YAML::Key << "m_materialProperties.m_sheenTint" << YAML::Value << m_materialProperties.m_sheenTint;
    out << YAML::Key << "m_materialProperties.m_clearCoat" << YAML::Value << m_materialProperties.m_clearCoat;
    out << YAML::Key << "m_materialProperties.m_clearCoatRoughness" << YAML::Value << m_materialProperties.m_clearCoatRoughness;
    out << YAML::Key << "m_materialProperties.m_IOR" << YAML::Value << m_materialProperties.m_IOR;
    out << YAML::Key << "m_materialProperties.m_transmission" << YAML::Value << m_materialProperties.m_transmission;
    out << YAML::Key << "m_materialProperties.m_transmissionRoughness" << YAML::Value << m_materialProperties.m_transmissionRoughness;
    out << YAML::Key << "m_materialProperties.m_emission" << YAML::Value << m_materialProperties.m_emission;

    out << YAML::Key << "m_vertexColorOnly" << YAML::Value << m_vertexColorOnly;
}

void Material::Deserialize(const YAML::Node& in) {
    m_albedoTexture.Load("m_albedoTexture", in);
    m_normalTexture.Load("m_normalTexture", in);
    m_metallicTexture.Load("m_metallicTexture", in);
    m_roughnessTexture.Load("m_roughnessTexture", in);
    m_aoTexture.Load("m_aoTexture", in);

    m_drawSettings.Load("m_drawSettings", in);
    if (in["m_materialProperties.m_albedoColor"])
        m_materialProperties.m_albedoColor = in["m_materialProperties.m_albedoColor"].as<glm::vec3>();
    if (in["m_materialProperties.m_subsurfaceColor"])
        m_materialProperties.m_subsurfaceColor = in["m_materialProperties.m_subsurfaceColor"].as<glm::vec3>();
    if (in["m_materialProperties.m_subsurfaceFactor"])
        m_materialProperties.m_subsurfaceFactor = in["m_materialProperties.m_subsurfaceFactor"].as<float>();
    if (in["m_materialProperties.m_subsurfaceRadius"])
        m_materialProperties.m_subsurfaceRadius = in["m_materialProperties.m_subsurfaceRadius"].as<glm::vec3>();
    if (in["m_materialProperties.m_metallic"])
        m_materialProperties.m_metallic = in["m_materialProperties.m_metallic"].as<float>();
    if (in["m_materialProperties.m_specular"])
        m_materialProperties.m_specular = in["m_materialProperties.m_specular"].as<float>();
    if (in["m_materialProperties.m_specularTint"])
        m_materialProperties.m_specularTint = in["m_materialProperties.m_specularTint"].as<float>();
    if (in["m_materialProperties.m_roughness"])
        m_materialProperties.m_roughness = in["m_materialProperties.m_roughness"].as<float>();
    if (in["m_materialProperties.m_sheen"])
        m_materialProperties.m_sheen = in["m_materialProperties.m_sheen"].as<float>();
    if (in["m_materialProperties.m_sheenTint"])
        m_materialProperties.m_sheenTint = in["m_materialProperties.m_sheenTint"].as<float>();
    if (in["m_materialProperties.m_clearCoat"])
        m_materialProperties.m_clearCoat = in["m_materialProperties.m_clearCoat"].as<float>();
    if (in["m_materialProperties.m_clearCoatRoughness"])
        m_materialProperties.m_clearCoatRoughness = in["m_materialProperties.m_clearCoatRoughness"].as<float>();
    if (in["m_materialProperties.m_IOR"])
        m_materialProperties.m_IOR = in["m_materialProperties.m_IOR"].as<float>();
    if (in["m_materialProperties.m_transmission"])
        m_materialProperties.m_transmission = in["m_materialProperties.m_transmission"].as<float>();
    if (in["m_materialProperties.m_transmissionRoughness"])
        m_materialProperties.m_transmissionRoughness = in["m_materialProperties.m_transmissionRoughness"].as<float>();
    if (in["m_materialProperties.m_emission"])
        m_materialProperties.m_emission = in["m_materialProperties.m_emission"].as<float>();

    if (in["m_vertexColorOnly"])
        m_vertexColorOnly = in["m_vertexColorOnly"].as<bool>();
    m_version = 0;
}
