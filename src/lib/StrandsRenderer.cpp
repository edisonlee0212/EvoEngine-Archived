#include "EditorLayer.hpp"
#include "StrandsRenderer.hpp"
#include "Strands.hpp"
#include "ClassRegistry.hpp"
#include "Prefab.hpp"
using namespace EvoEngine;
void StrandsRenderer::RenderBound(const std::shared_ptr<EditorLayer>& editorLayer, glm::vec4& color)
{
    const auto transform = GetScene()->GetDataComponent<GlobalTransform>(GetOwner()).m_value;
    glm::vec3 size = m_strands.Get<Strands>()->m_bound.Size();
    if (size.x < 0.01f)
        size.x = 0.01f;
    if (size.z < 0.01f)
        size.z = 0.01f;
    if (size.y < 0.01f)
        size.y = 0.01f;
    GizmoSettings gizmoSettings;
    gizmoSettings.m_drawSettings.m_cullMode = VK_CULL_MODE_NONE;
    gizmoSettings.m_drawSettings.m_blending = true;
    gizmoSettings.m_drawSettings.m_polygonMode = VK_POLYGON_MODE_LINE;
    gizmoSettings.m_drawSettings.m_lineWidth = 3.0f;
    editorLayer->DrawGizmoMesh(Resources::GetResource<Mesh>("PRIMITIVE_CUBE"),
        color,
        transform * (glm::translate(m_strands.Get<Strands>()->m_bound.Center()) * glm::scale(size)),
        1, gizmoSettings);
}

void StrandsRenderer::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
    ImGui::Checkbox("Cast shadow##StrandsRenderer:", &m_castShadow);
    editorLayer->DragAndDropButton<Material>(m_material, "Material");
    editorLayer->DragAndDropButton<Strands>(m_strands, "Strands");
    if (m_strands.Get<Strands>())
    {
        if (ImGui::TreeNode("Strands##StrandsRenderer"))
        {
            static bool displayBound = true;
            ImGui::Checkbox("Display bounds##StrandsRenderer", &displayBound);
            if (displayBound)
            {
                static auto displayBoundColor = glm::vec4(0.0f, 1.0f, 0.0f, 0.2f);
                ImGui::ColorEdit4("Color:##StrandsRenderer", (float*)(void*)&displayBoundColor);
                RenderBound(editorLayer, displayBoundColor);
            }
            ImGui::TreePop();
        }
    }
}

void StrandsRenderer::OnCreate()
{
    SetEnabled(true);
}


void StrandsRenderer::Serialize(YAML::Emitter& out)
{
    out << YAML::Key << "m_castShadow" << m_castShadow;

    m_strands.Save("m_strands", out);
    m_material.Save("m_material", out);
}

void StrandsRenderer::Deserialize(const YAML::Node& in)
{
    m_castShadow = in["m_castShadow"].as<bool>();

    m_strands.Load("m_strands", in);
    m_material.Load("m_material", in);
}
void StrandsRenderer::PostCloneAction(const std::shared_ptr<IPrivateComponent>& target)
{
}
void StrandsRenderer::CollectAssetRef(std::vector<AssetRef>& list)
{
    list.push_back(m_strands);
    list.push_back(m_material);
}
void StrandsRenderer::OnDestroy()
{
    m_strands.Clear();
    m_material.Clear();

    m_material.Clear();
    m_castShadow = true;
}

