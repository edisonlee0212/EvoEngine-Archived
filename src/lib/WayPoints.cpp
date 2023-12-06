#include "WayPoints.hpp"

#include "EditorLayer.hpp"
using namespace EvoEngine;

void WayPoints::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	EntityRef tempEntityHolder;
	auto scene = GetScene();
	bool changed = false;
	if (editorLayer->DragAndDropButton(tempEntityHolder, "Drop new SoilLayerDescriptor here...")) {
		auto entity = tempEntityHolder.Get();
		if (scene->IsEntityValid(entity)) {
			m_entities.emplace_back(entity);
			changed = true;
		}
		tempEntityHolder.Clear();
	}
	for (int i = 0; i < m_entities.size(); i++)
	{
		auto entity = m_entities[i].Get();
		if (scene->IsEntityValid(entity))
		{
			if (ImGui::TreeNodeEx(("No." + std::to_string(i + 1)).c_str(), ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::Text(("Name: " + scene->GetEntityName(entity)).c_str());

				if (ImGui::Button("Remove"))
				{
					m_entities.erase(m_entities.begin() + i);
					changed = true;
					ImGui::TreePop();
					continue;
				}
				if (i < m_entities.size() - 1) {
					ImGui::SameLine();
					if (ImGui::Button("Move down"))
					{
						changed = true;
						std::swap(m_entities[i + 1], m_entities[i]);
					}
				}
				if (i > 0) {
					ImGui::SameLine();
					if (ImGui::Button("Move up"))
					{
						changed = true;
						std::swap(m_entities[i - 1], m_entities[i]);
					}
				}
				ImGui::TreePop();
			}
		}
		else
		{
			m_entities.erase(m_entities.begin() + i);
			i--;
		}
	}
}

void WayPoints::OnCreate()
{
	
}

void WayPoints::OnDestroy()
{
	m_entities.clear();
	m_speed = 1.0f;
}

void WayPoints::Update()
{
	
}
