#include "MeshRenderer.hpp"

#include "Material.hpp"
#include "Mesh.hpp"
#include "Transform.hpp"
#include "EditorLayer.hpp"
using namespace evo_engine;

void MeshRenderer::RenderBound(const std::shared_ptr<EditorLayer>& editorLayer, const glm::vec4& color)
{
	const auto scene = GetScene();
	const auto transform = scene->GetDataComponent<GlobalTransform>(GetOwner()).value;
	glm::vec3 size = m_mesh.Get<Mesh>()->GetBound().Size() * 2.0f;
	if (size.x < 0.01f)
		size.x = 0.01f;
	if (size.z < 0.01f)
		size.z = 0.01f;
	if (size.y < 0.01f)
		size.y = 0.01f;
	GizmoSettings gizmoSettings;
	gizmoSettings.draw_settings.m_cullMode = VK_CULL_MODE_NONE;
	gizmoSettings.draw_settings.m_blending = true;
	gizmoSettings.draw_settings.m_polygonMode = VK_POLYGON_MODE_LINE;
	gizmoSettings.draw_settings.m_lineWidth = 5.0f;
	editorLayer->DrawGizmoMesh(
		Resources::GetResource<Mesh>("PRIMITIVE_CUBE"),
		color,
		transform * (glm::translate(m_mesh.Get<Mesh>()->GetBound().Center()) * glm::scale(size)),
		1, gizmoSettings);

}

bool MeshRenderer::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	bool changed = false;
	if (ImGui::Checkbox("Cast shadow##MeshRenderer", &m_castShadow)) changed = true;
	if (editorLayer->DragAndDropButton<Material>(m_material, "Material")) changed = true;
	if (editorLayer->DragAndDropButton<Mesh>(m_mesh, "Mesh")) changed = true;
	if (m_mesh.Get<Mesh>())
	{
		if (ImGui::TreeNodeEx("Mesh##MeshRenderer", ImGuiTreeNodeFlags_DefaultOpen))
		{
			static bool displayBound = true;
			ImGui::Checkbox("Display bounds##MeshRenderer", &displayBound);
			if (displayBound)
			{
				static auto displayBoundColor = glm::vec4(0.0f, 1.0f, 0.0f, 0.1f);
				ImGui::ColorEdit4("Color:##MeshRenderer", (float*)(void*)&displayBoundColor);
				RenderBound(editorLayer, displayBoundColor);
			}
			ImGui::TreePop();
		}
	}

	return changed;
}

void MeshRenderer::Serialize(YAML::Emitter& out) const
{
	out << YAML::Key << "m_castShadow" << m_castShadow;

	m_mesh.Save("m_mesh", out);
	m_material.Save("m_material", out);
}

void MeshRenderer::Deserialize(const YAML::Node& in)
{
	m_castShadow = in["m_castShadow"].as<bool>();

	m_mesh.Load("m_mesh", in);
	m_material.Load("m_material", in);
}
void MeshRenderer::PostCloneAction(const std::shared_ptr<IPrivateComponent>& target)
{
}
void MeshRenderer::CollectAssetRef(std::vector<AssetRef>& list)
{
	list.push_back(m_mesh);
	list.push_back(m_material);
}
void MeshRenderer::OnDestroy()
{
	m_mesh.Clear();
	m_material.Clear();

	m_castShadow = true;
}
