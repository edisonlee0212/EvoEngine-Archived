#include "EditorLayer.hpp"
#include "Particles.hpp"
#include "Gizmos.hpp"

using namespace EvoEngine;

void Particles::OnCreate()
{
    m_particleInfoList = std::make_shared<ParticleInfoList>();
    m_boundingBox = Bound();
    SetEnabled(true);
}

void Particles::RecalculateBoundingBox()
{
    const auto particleInfoList = m_particleInfoList.Get<ParticleInfoList>();
    if (!particleInfoList) return;
    if (particleInfoList->m_particleInfos.empty())
    {
        m_boundingBox.m_min = glm::vec3(0.0f);
        m_boundingBox.m_max = glm::vec3(0.0f);
        return;
    }
    auto minBound = glm::vec3(FLT_MAX);
    auto maxBound = glm::vec3(FLT_MAX);
    const auto meshBound = m_mesh.Get<Mesh>()->GetBound();
    for (const auto& i : particleInfoList->m_particleInfos)
    {
	    const glm::vec3 center = i.m_instanceMatrix.m_value * glm::vec4(meshBound.Center(), 1.0f);
	    const glm::vec3 size = glm::vec4(meshBound.Size(), 0) * i.m_instanceMatrix.m_value / 2.0f;
        minBound = glm::vec3(
            (glm::min)(minBound.x, center.x - size.x),
            (glm::min)(minBound.y, center.y - size.y),
            (glm::min)(minBound.z, center.z - size.z));

        maxBound = glm::vec3(
            (glm::max)(maxBound.x, center.x + size.x),
            (glm::max)(maxBound.y, center.y + size.y),
            (glm::max)(maxBound.z, center.z + size.z));
    }
    m_boundingBox.m_max = maxBound;
    m_boundingBox.m_min = minBound;
}

void Particles::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
    ImGui::Checkbox("Cast shadow##Particles", &m_castShadow);
    editorLayer->DragAndDropButton<Material>(m_material, "Material");
    editorLayer->DragAndDropButton<Mesh>(m_mesh, "Mesh");
    editorLayer->DragAndDropButton<ParticleInfoList>(m_particleInfoList, "ParticleInfoList");

    if (const auto particleInfoList = m_particleInfoList.Get<ParticleInfoList>()) {
        ImGui::Text(("Instance count##Particles" + std::to_string(particleInfoList->m_particleInfos.size())).c_str());
        if (ImGui::Button("Calculate bounds##Particles"))
        {
            RecalculateBoundingBox();
        }
        static bool displayBound;
        ImGui::Checkbox("Display bounds##Particles", &displayBound);
        if (displayBound)
        {
            static auto displayBoundColor = glm::vec4(0.0f, 1.0f, 0.0f, 0.2f);
            ImGui::ColorEdit4("Color:##Particles", (float*)(void*)&displayBoundColor);
            const auto transform = GetScene()->GetDataComponent<GlobalTransform>(GetOwner()).m_value;

            GizmoSettings gizmoSettings;
            gizmoSettings.m_drawSettings.m_cullMode = VK_CULL_MODE_NONE;
            gizmoSettings.m_drawSettings.m_blending = true;
            gizmoSettings.m_drawSettings.m_polygonMode = VK_POLYGON_MODE_LINE;
            gizmoSettings.m_drawSettings.m_lineWidth = 3.0f;

            Gizmos::DrawGizmoCube(displayBoundColor,
                transform * glm::translate(m_boundingBox.Center()) * glm::scale(m_boundingBox.Size()),
                1, gizmoSettings);
        }
    }
    
}

void Particles::Serialize(YAML::Emitter& out)
{
    out << YAML::Key << "m_castShadow" << m_castShadow;

    m_mesh.Save("m_mesh", out);
    m_material.Save("m_material", out);
    m_particleInfoList.Save("m_particleInfoList", out);
}

void Particles::Deserialize(const YAML::Node& in)
{
    m_castShadow = in["m_castShadow"].as<bool>();

    m_mesh.Load("m_mesh", in);
    m_material.Load("m_material", in);
    m_particleInfoList.Load("m_particleInfoList", in);
}
void Particles::PostCloneAction(const std::shared_ptr<IPrivateComponent>& target)
{
}

void Particles::CollectAssetRef(std::vector<AssetRef>& list)
{
    list.push_back(m_mesh);
    list.push_back(m_material);
    list.push_back(m_particleInfoList);
}
void Particles::OnDestroy()
{
    m_mesh.Clear();
    m_material.Clear();
    m_particleInfoList.Clear();

    m_material.Clear();
    m_castShadow = true;
}