#include "EditorLayer.hpp"
#include "LODGroup.hpp"
#include "MeshRenderer.hpp"
#include "SkinnedMeshRenderer.hpp"
#include "StrandsRenderer.hpp"
using namespace evo_engine;

bool Lod::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	bool changed = false;
	std::string tag = "##LOD" + std::to_string(m_index);
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.f, 0.3f, 0, 1));
	ImGui::Button("Drop to Add...");
	ImGui::PopStyleColor(1);
	EntityRef temp {};
	if (EditorLayer::Droppable(temp))
	{
		const auto scene = Application::GetActiveScene();
		const auto entity = temp.Get();
		if(scene->HasPrivateComponent<MeshRenderer>(entity))
		{
			const auto pc = scene->GetOrSetPrivateComponent<MeshRenderer>(entity).lock();
			bool duplicate = false;
			for(const auto& i : m_renderers)
			{
				if (i.GetHandle() == pc->GetHandle()) {
					duplicate = true;
					break;
				}
			}
			if (!duplicate) {
				m_renderers.emplace_back();
				m_renderers.back().Set(pc);
			}
		}
		if (scene->HasPrivateComponent<SkinnedMeshRenderer>(entity))
		{
			const auto pc = scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(entity).lock();
			bool duplicate = false;
			for (const auto& i : m_renderers)
			{
				if (i.GetHandle() == pc->GetHandle()) {
					duplicate = true;
					break;
				}
			}
			if (!duplicate) {
				m_renderers.emplace_back();
				m_renderers.back().Set(pc);
			}
		}
		if (scene->HasPrivateComponent<Particles>(entity))
		{
			const auto pc = scene->GetOrSetPrivateComponent<Particles>(entity).lock();
			bool duplicate = false;
			for (const auto& i : m_renderers)
			{
				if (i.GetHandle() == pc->GetHandle()) {
					duplicate = true;
					break;
				}
			}
			if (!duplicate) {
				m_renderers.emplace_back();
				m_renderers.back().Set(pc);
			}
		}
		if (scene->HasPrivateComponent<StrandsRenderer>(entity))
		{
			const auto pc = scene->GetOrSetPrivateComponent<StrandsRenderer>(entity).lock();
			bool duplicate = false;
			for (const auto& i : m_renderers)
			{
				if (i.GetHandle() == pc->GetHandle()) {
					duplicate = true;
					break;
				}
			}
			if (!duplicate) {
				m_renderers.emplace_back();
				m_renderers.back().Set(pc);
			}
		}
	}
	for (auto it = m_renderers.begin(); it != m_renderers.end(); ++it)
	{
		if (const auto ptr = it->Get<IPrivateComponent>())
		{
			const auto scene = Application::GetActiveScene();
			if (!scene->IsEntityValid(ptr->GetOwner()))
			{
				it->Clear();
				ImGui::Button("none");
				return true;
			}
			ImGui::Button((scene->GetEntityName(ptr->GetOwner()) + "(" + ptr->GetTypeName() + ")").c_str());
			EditorLayer::Draggable(*it);
			if(EditorLayer::Remove(*it))
			{
				if(!it->Get<IPrivateComponent>())
				{
					m_renderers.erase(it);
					break;
				}
			}
		}
	}
	return changed;
}

void LodGroup::Serialize(YAML::Emitter& out) const
{
	out << YAML::Key << "m_overrideLodFactor" << YAML::Value << m_overrideLodFactor;
	out << YAML::Key << "m_lodFactor" << YAML::Value << m_lodFactor;

	if (!m_lods.empty()) {
		out << YAML::Key << "m_lods" << YAML::BeginSeq;
		for(const auto& lod : m_lods)
		{
			out << YAML::BeginMap;
			{
				out << YAML::Key << "m_index" << YAML::Value << lod.m_index;
				out << YAML::Key << "m_lodOffset" << YAML::Value << lod.m_lodOffset;
				out << YAML::Key << "m_transitionWidth" << YAML::Value << lod.m_transitionWidth;
				if(!lod.m_renderers.empty())
				{
					out << YAML::Key << "m_renderers" << YAML::BeginSeq;
					for (const auto& renderer : lod.m_renderers)
					{
						out << YAML::BeginMap;
						{
							renderer.Serialize(out);
						}
						out << YAML::EndMap;
					}
					out << YAML::EndSeq;
				}
			}
			out << YAML::EndMap;
		}
		out << YAML::EndSeq;
	}
}

void LodGroup::Deserialize(const YAML::Node& in)
{
	const auto scene = GetScene();
	if (in["m_overrideLodFactor"]) m_overrideLodFactor = in["m_overrideLodFactor"].as<bool>();
	if (in["m_lodFactor"]) m_lodFactor = in["m_lodFactor"].as<float>();
	if(in["m_lods"])
	{
		const auto& inLods = in["m_lods"];
		m_lods.clear();
		for(const auto& inLod : inLods)
		{
			m_lods.emplace_back();
			auto& lod = m_lods.back();
			lod.m_renderers.clear();
			if (inLod["m_index"]) lod.m_index = inLod["m_index"].as<int>();
			if (inLod["m_lodOffset"]) lod.m_lodOffset = inLod["m_lodOffset"].as<float>();
			if (inLod["m_transitionWidth"]) lod.m_transitionWidth = inLod["m_transitionWidth"].as<float>();
			if(inLod["m_renderers"])
			{
				for(const auto& inRenderer : inLod["m_renderers"])
				{
					lod.m_renderers.emplace_back();
					auto& renderer = lod.m_renderers.back();
					renderer.Deserialize(inRenderer, scene);
				}
			}
		}
	}
}

bool LodGroup::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	bool changed = false;
	if (ImGui::Checkbox("Override LOD Factor", &m_overrideLodFactor)) changed = true;
	if(m_overrideLodFactor)
	{
		if (ImGui::SliderFloat("Current LOD Factor", &m_lodFactor,0.f, 1.f)) changed = true;
	}else
	{
		ImGui::Text((std::string("LOD Factor: ") + std::to_string(m_lodFactor)).c_str());
	}
	if(ImGui::TreeNodeEx("LODs", ImGuiTreeNodeFlags_DefaultOpen))
	{
		int lodIndex = 0;
		for (auto it = m_lods.begin(); it != m_lods.end(); ++it)
		{
			if(ImGui::TreeNode((std::string("LOD ") + std::to_string(lodIndex)).c_str()))
			{
				std::string tag = "##LODs" + std::to_string(lodIndex);
				float min = 0.f;
				float max = 1.f;
				if(it != m_lods.begin())
				{
					min = (it - 1)->m_lodOffset;
				}
				if(it + 1 != m_lods.end())
				{
					max = (it + 1)->m_lodOffset;
				}
				if (ImGui::SliderFloat("LOD Offset", &it->m_lodOffset, min, max)) changed = true;
				if (it->OnInspect(editorLayer)) changed = true;
				ImGui::TreePop();
			}
			lodIndex++;
		}
		if(!m_lods.empty() && m_lods.back().m_lodOffset != 1.f)
		{
			ImGui::Text((std::string("Culled: [") + std::to_string(m_lods.back().m_lodOffset) + " -> 1.0]").c_str());
		}
		if (ImGui::Button("Push LOD"))
		{
			m_lods.emplace_back();
			m_lods.back().m_lodOffset = 1.f;
			m_lods.back().m_index = m_lods.size() - 1;
			if (m_lods.size() > 1)
			{
				float prevVal = 0.f;
				if (m_lods.size() > 2)
				{
					prevVal = m_lods.at(m_lods.size() - 3).m_lodOffset;
				}
				m_lods.at(m_lods.size() - 2).m_lodOffset = (1.f + prevVal) * .5f;
			}
			changed = true;
		}
		if (!m_lods.empty())
		{
			ImGui::SameLine();
			if (ImGui::Button("Pop LOD")) {
				const float lastOffset = m_lods.back().m_lodOffset;
				m_lods.pop_back();
				if (!m_lods.empty()) m_lods.back().m_lodOffset = lastOffset;
				changed = true;
			}
		}
		ImGui::TreePop();
	}

	return changed;
}

void LodGroup::Relink(const std::unordered_map<Handle, Handle>& map, const std::shared_ptr<Scene>& scene)
{
	for(auto& lod : m_lods)
	{
		for(auto& i : lod.m_renderers)
		{
			i.Relink(map, scene);
		}
	}
}
