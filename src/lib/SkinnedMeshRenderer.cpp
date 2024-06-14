#include "SkinnedMeshRenderer.hpp"

#include "EditorLayer.hpp"
using namespace evo_engine;
void SkinnedMeshRenderer::RenderBound(const std::shared_ptr<EditorLayer>& editorLayer, glm::vec4& color)
{
	const auto scene = GetScene();
	const auto transform = scene->GetDataComponent<GlobalTransform>(GetOwner()).m_value;
	glm::vec3 size = m_skinnedMesh.Get<SkinnedMesh>()->m_bound.Size() * 2.0f;
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
	gizmoSettings.m_drawSettings.m_lineWidth = 5.0f;
	editorLayer->DrawGizmoMesh(
		Resources::GetResource<Mesh>("PRIMITIVE_CUBE"),
		color,
		transform * (glm::translate(m_skinnedMesh.Get<SkinnedMesh>()->m_bound.Center()) * glm::scale(size)),
		1, gizmoSettings);
}

void SkinnedMeshRenderer::UpdateBoneMatrices()
{
	const auto scene = GetScene();
	const auto animator = m_animator.Get<Animator>();
	if (!animator)
		return;
	const auto skinnedMesh = m_skinnedMesh.Get<SkinnedMesh>();
	if (!skinnedMesh)
		return;
	if (m_ragDoll)
	{
		if (m_ragDollFreeze)
			return;

		m_boneMatrices->m_value.resize(skinnedMesh->m_boneAnimatorIndices.size());
		for (int i = 0; i < m_boundEntities.size(); i++)
		{
			auto entity = m_boundEntities[i].Get();
			if (entity.GetIndex() != 0)
			{
				m_ragDollTransformChain[i] = scene->GetDataComponent<GlobalTransform>(entity).m_value * animator->m_offsetMatrices[i];
			}
		}
		for (int i = 0; i < skinnedMesh->m_boneAnimatorIndices.size(); i++)
		{
			m_boneMatrices->m_value[i] = m_ragDollTransformChain[skinnedMesh->m_boneAnimatorIndices[i]];
		}
	}
	else
	{
		auto skinnedMesh = m_skinnedMesh.Get<SkinnedMesh>();
		if (animator->m_boneSize == 0)
			return;
		m_boneMatrices->m_value.resize(skinnedMesh->m_boneAnimatorIndices.size());
		for (int i = 0; i < skinnedMesh->m_boneAnimatorIndices.size(); i++)
		{
			m_boneMatrices->m_value[i] = animator->m_transformChain[skinnedMesh->m_boneAnimatorIndices[i]];
		}
	}
}

bool SkinnedMeshRenderer::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	bool changed = false;
	if (editorLayer->DragAndDropButton<Animator>(m_animator, "Animator")) changed = true;
	if (ImGui::Checkbox("Cast shadow##SkinnedMeshRenderer", &m_castShadow)) changed = true;
	if (editorLayer->DragAndDropButton<Material>(m_material, "Material")) changed = true;
	if (editorLayer->DragAndDropButton<SkinnedMesh>(m_skinnedMesh, "Skinned Mesh")) changed = true;
	if (m_skinnedMesh.Get<SkinnedMesh>())
	{
		if (ImGui::TreeNode("Skinned Mesh:##SkinnedMeshRenderer"))
		{
			static bool displayBound = true;
			ImGui::Checkbox("Display bounds##SkinnedMeshRenderer", &displayBound);
			if (displayBound)
			{
				static auto displayBoundColor = glm::vec4(0.0f, 1.0f, 0.0f, 0.2f);
				ImGui::ColorEdit4("Color:##SkinnedMeshRenderer", static_cast<float*>(static_cast<void*>(&displayBoundColor)));
				RenderBound(editorLayer, displayBoundColor);
			}
			ImGui::TreePop();
		}
	}
	if (const auto animator = m_animator.Get<Animator>())
	{
		static bool debugRenderBones = true;
		static float debugRenderBonesSize = 0.01f;
		static glm::vec4 debugRenderBonesColor = glm::vec4(1, 0, 0, 0.5);
		ImGui::Checkbox("Display bones", &debugRenderBones);
		if (animator && debugRenderBones)
		{
			ImGui::DragFloat("Size", &debugRenderBonesSize, 0.01f, 0.01f, 3.0f);
			ImGui::ColorEdit4("Color", &debugRenderBonesColor.x);
			auto scene = GetScene();
			auto owner = GetOwner();
			const auto selfScale = scene->GetDataComponent<GlobalTransform>(owner).GetScale();
			std::vector<ParticleInfo> debugRenderingMatrices;
			GlobalTransform ltw;

			static std::shared_ptr<ParticleInfoList> particleInfoList;
			if (!particleInfoList) particleInfoList = ProjectManager::CreateTemporaryAsset<ParticleInfoList>();
			if (!m_ragDoll)
			{
				debugRenderingMatrices.resize(animator->m_transformChain.size());
				Jobs::RunParallelFor(animator->m_transformChain.size(), [&](unsigned i)
					{
						debugRenderingMatrices.at(i).m_instanceMatrix.m_value = animator->m_transformChain.at(i);
						debugRenderingMatrices.at(i).m_instanceColor = debugRenderBonesColor;
					});

				ltw = scene->GetDataComponent<GlobalTransform>(owner);
			}
			else {
				debugRenderingMatrices.resize(m_ragDollTransformChain.size());
				Jobs::RunParallelFor(m_ragDollTransformChain.size(), [&](unsigned i)
					{
						debugRenderingMatrices.at(i).m_instanceMatrix.m_value = m_ragDollTransformChain.at(i);
						debugRenderingMatrices.at(i).m_instanceColor = debugRenderBonesColor;
					});
			}
			for (int index = 0; index < debugRenderingMatrices.size(); index++)
			{
				debugRenderingMatrices[index].m_instanceMatrix.m_value =
					debugRenderingMatrices[index].m_instanceMatrix.m_value * glm::inverse(animator->m_offsetMatrices[index]) * glm::inverse(glm::scale(selfScale));
			}
			particleInfoList->SetParticleInfos(debugRenderingMatrices);
			editorLayer->DrawGizmoMeshInstancedColored(Resources::GetResource<Mesh>("PRIMITIVE_SPHERE"), particleInfoList, ltw.m_value, debugRenderBonesSize);
		}

		if (ImGui::Checkbox("RagDoll", &m_ragDoll)) {
			if (m_ragDoll) {
				SetRagDoll(m_ragDoll);
			}
			changed = true;
		}
		if (m_ragDoll)
		{
			ImGui::Checkbox("Freeze", &m_ragDollFreeze);

			if (ImGui::TreeNode("RagDoll"))
			{
				for (int i = 0; i < m_boundEntities.size(); i++)
				{
					if (editorLayer->DragAndDropButton(m_boundEntities[i], "Bone: " + animator->m_names[i]))
					{
						auto entity = m_boundEntities[i].Get();
						SetRagDollBoundEntity(i, entity);
						changed = true;
					}
				}
				ImGui::TreePop();
			}
		}
	}
	return changed;
}

void SkinnedMeshRenderer::Serialize(YAML::Emitter& out) const
{
	out << YAML::Key << "m_castShadow" << m_castShadow;

	m_animator.Save("m_animator", out);
	m_skinnedMesh.Save("m_skinnedMesh", out);
	m_material.Save("m_material", out);

	out << YAML::Key << "m_ragDoll" << YAML::Value << m_ragDoll;
	out << YAML::Key << "m_ragDollFreeze" << YAML::Value << m_ragDollFreeze;

	if (!m_boundEntities.empty())
	{
		out << YAML::Key << "m_boundEntities" << YAML::Value << YAML::BeginSeq;
		for (int i = 0; i < m_boundEntities.size(); i++)
		{
			out << YAML::BeginMap;
			m_boundEntities[i].Serialize(out);
			out << YAML::EndMap;
		}
		out << YAML::EndSeq;
	}

	if (!m_ragDollTransformChain.empty())
	{
		out << YAML::Key << "m_ragDollTransformChain" << YAML::Value
			<< YAML::Binary(
				reinterpret_cast<const unsigned char*>(m_ragDollTransformChain.data()),
				m_ragDollTransformChain.size() * sizeof(glm::mat4));
	}
}

void SkinnedMeshRenderer::Deserialize(const YAML::Node& in)
{
	m_castShadow = in["m_castShadow"].as<bool>();

	m_animator.Load("m_animator", in, GetScene());
	m_skinnedMesh.Load("m_skinnedMesh", in);
	m_material.Load("m_material", in);

	m_ragDoll = in["m_ragDoll"].as<bool>();
	m_ragDollFreeze = in["m_ragDollFreeze"].as<bool>();
	if (auto inBoundEntities = in["m_boundEntities"])
	{
		for (const auto& i : inBoundEntities)
		{
			EntityRef ref;
			ref.Deserialize(i);
			m_boundEntities.push_back(ref);
		}
	}

	if (in["m_ragDollTransformChain"])
	{
		const auto chains = in["m_ragDollTransformChain"].as<YAML::Binary>();
		m_ragDollTransformChain.resize(chains.size() / sizeof(glm::mat4));
		std::memcpy(m_ragDollTransformChain.data(), chains.data(), chains.size());
	}
}
void SkinnedMeshRenderer::OnCreate()
{
	m_boneMatrices = std::make_shared<BoneMatrices>();
	SetEnabled(true);
}
void SkinnedMeshRenderer::PostCloneAction(const std::shared_ptr<IPrivateComponent>& target)
{
}
void SkinnedMeshRenderer::Relink(const std::unordered_map<Handle, Handle>& map, const std::shared_ptr<Scene>& scene)
{
	m_animator.Relink(map, scene);
	for (auto& i : m_boundEntities)
	{
		i.Relink(map);
	}
}
void SkinnedMeshRenderer::CollectAssetRef(std::vector<AssetRef>& list)
{
	list.push_back(m_skinnedMesh);
	list.push_back(m_material);
}
bool SkinnedMeshRenderer::RagDoll() const
{
	return m_ragDoll;
}
void SkinnedMeshRenderer::SetRagDoll(bool value)
{
	const auto animator = m_animator.Get<Animator>();
	if (value && !animator)
	{
		EVOENGINE_ERROR("Failed! No animator!");
		return;
	}
	m_ragDoll = value;
	if (m_ragDoll)
	{
		const auto scene = GetScene();
		// Resize entities
		m_boundEntities.resize(animator->m_transformChain.size());
		// Copy current transform chain
		m_ragDollTransformChain = animator->m_transformChain;
		const auto ltw = scene->GetDataComponent<GlobalTransform>(GetOwner()).m_value;
		for (auto& i : m_ragDollTransformChain) {
			i = ltw * i;
		}
	}
}
void SkinnedMeshRenderer::SetRagDollBoundEntity(int index, const Entity& entity, bool resetTransform)
{
	if (!m_ragDoll) {
		EVOENGINE_ERROR("Not ragdoll!");
		return;
	}
	if (index >= m_boundEntities.size()) {
		EVOENGINE_ERROR("Index exceeds limit!");
		return;
	}
	if (const auto scene = GetScene(); scene->IsEntityValid(entity))
	{
		if (const auto animator = m_animator.Get<Animator>())
		{
			if (resetTransform)
			{
				GlobalTransform globalTransform;
				globalTransform.m_value = m_ragDollTransformChain[index] * glm::inverse(animator->m_offsetMatrices[index]);
				scene->SetDataComponent(entity, globalTransform);
			}
		}
		m_boundEntities[index] = entity;
	}
}
void SkinnedMeshRenderer::SetRagDollBoundEntities(const std::vector<Entity>& entities, bool resetTransform)
{
	if (!m_ragDoll) {
		EVOENGINE_ERROR("Not ragdoll!");
		return;
	}
	for (int i = 0; i < entities.size(); i++) {
		SetRagDollBoundEntity(i, entities[i], resetTransform);
	}
}
size_t SkinnedMeshRenderer::GetRagDollBoneSize() const
{
	if (!m_ragDoll) {
		EVOENGINE_ERROR("Not ragdoll!");
		return 0;
	}
	return m_boundEntities.size();
}
void SkinnedMeshRenderer::OnDestroy()
{
	m_ragDollTransformChain.clear();
	m_boundEntities.clear();
	m_animator.Clear();
	m_boneMatrices.reset();
	m_skinnedMesh.Clear();
	m_material.Clear();
	m_ragDoll = false;
	m_ragDollFreeze = false;
	m_castShadow = true;
}

