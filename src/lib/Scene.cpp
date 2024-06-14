#include "Scene.hpp"
#include "Application.hpp"

#include "Entities.hpp"
#include "EntityMetadata.hpp"
#include "ClassRegistry.hpp"
#include "Jobs.hpp"
#include "Lights.hpp"
#include "MeshRenderer.hpp"
#include "SkinnedMeshRenderer.hpp"
#include "UnknownPrivateComponent.hpp"
using namespace evo_engine;

void Scene::Purge()
{
	m_pressedKeys.clear();
	m_mainCamera.Clear();

	m_sceneDataStorage.m_entityPrivateComponentStorage = PrivateComponentStorage();
	m_sceneDataStorage.m_entityPrivateComponentStorage.m_scene = std::dynamic_pointer_cast<Scene>(GetSelf());
	m_sceneDataStorage.m_entities.clear();
	m_sceneDataStorage.m_entityMetadataList.clear();
	for (int index = 1; index < m_sceneDataStorage.m_dataComponentStorages.size(); index++)
	{
		auto& i = m_sceneDataStorage.m_dataComponentStorages[index];
		for (auto& chunk : i.m_chunkArray.m_chunks)
			free(chunk.m_data);
	}
	m_sceneDataStorage.m_dataComponentStorages.clear();

	m_sceneDataStorage.m_dataComponentStorages.emplace_back();
	m_sceneDataStorage.m_entities.emplace_back();
	m_sceneDataStorage.m_entityMetadataList.emplace_back();
}

Bound Scene::GetBound() const
{
	return m_worldBound;
}

void Scene::SetBound(const Bound& value)
{
	m_worldBound = value;
}

Scene::~Scene()
{
	Purge();
	for (const auto& i : m_systems)
	{
		i.second->OnDestroy();
	}
}

void Scene::Start() const
{
	const auto entities = m_sceneDataStorage.m_entities;
	for (const auto& entity : entities)
	{
		if (entity.m_version == 0)
			continue;
		const auto entityInfo = m_sceneDataStorage.m_entityMetadataList[entity.m_index];
		if (!entityInfo.m_enabled)
			continue;
		for (const auto& privateComponentElement : entityInfo.m_privateComponentElements)
		{
			if (!privateComponentElement.m_privateComponentData->m_enabled)
				continue;
			if (!privateComponentElement.m_privateComponentData->m_started)
			{
				privateComponentElement.m_privateComponentData->Start();
				if (entity.m_version != entityInfo.m_version)
					break;
				privateComponentElement.m_privateComponentData->m_started = true;
			}
			if (entity.m_version != entityInfo.m_version)
				break;
		}
	}
	for (auto& i : m_systems)
	{
		if (i.second->Enabled())
		{
			if (!i.second->m_started)
			{
				i.second->Start();
				i.second->m_started = true;
			}
		}
	}
}

void Scene::Update() const
{
	const auto entities = m_sceneDataStorage.m_entities;
	for (const auto& entity : entities)
	{
		if (entity.m_version == 0)
			continue;
		const auto entityInfo = m_sceneDataStorage.m_entityMetadataList[entity.m_index];
		if (!entityInfo.m_enabled)
			continue;
		for (const auto& privateComponentElement : entityInfo.m_privateComponentElements)
		{
			if (!privateComponentElement.m_privateComponentData->m_enabled ||
				!privateComponentElement.m_privateComponentData->m_started)
				continue;
			privateComponentElement.m_privateComponentData->Update();
			if (entity.m_version != entityInfo.m_version)
				break;
		}
	}

	for (auto& i : m_systems)
	{
		if (i.second->Enabled() && i.second->m_started)
		{
			i.second->Update();
		}
	}
}

void Scene::LateUpdate() const
{
	const auto entities = m_sceneDataStorage.m_entities;
	for (const auto& entity : entities)
	{
		if (entity.m_version == 0)
			continue;
		const auto entityInfo = m_sceneDataStorage.m_entityMetadataList[entity.m_index];
		if (!entityInfo.m_enabled)
			continue;
		for (const auto& privateComponentElement : entityInfo.m_privateComponentElements)
		{
			if (!privateComponentElement.m_privateComponentData->m_enabled ||
				!privateComponentElement.m_privateComponentData->m_started)
				continue;
			privateComponentElement.m_privateComponentData->LateUpdate();
			if (entity.m_version != entityInfo.m_version)
				break;
		}
	}

	for (auto& i : m_systems)
	{
		if (i.second->Enabled() && i.second->m_started)
		{
			i.second->LateUpdate();
		}
	}

}
void Scene::FixedUpdate() const
{
	const auto entities = m_sceneDataStorage.m_entities;
	for (const auto& entity : entities)
	{
		if (entity.m_version == 0)
			continue;
		const auto entityInfo = m_sceneDataStorage.m_entityMetadataList[entity.m_index];
		if (!entityInfo.m_enabled)
			continue;
		for (const auto& privateComponentElement : entityInfo.m_privateComponentElements)
		{
			if (!privateComponentElement.m_privateComponentData->m_enabled ||
				!privateComponentElement.m_privateComponentData->m_started)
				continue;
			privateComponentElement.m_privateComponentData->FixedUpdate();
			if (entity.m_version != entityInfo.m_version)
				break;
		}
	}

	for (const auto& i : m_systems)
	{
		if (i.second->Enabled() && i.second->m_started)
		{
			i.second->FixedUpdate();
		}
	}

}
static const char* EnvironmentTypes[]{ "Environmental Map", "Color" };
bool Scene::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	bool modified = false;
	if (this == Application::GetActiveScene().get())
		if (editorLayer->DragAndDropButton<Camera>(m_mainCamera, "Main Camera", true))
			modified = true;
	if (ImGui::TreeNodeEx("Environment Settings", ImGuiTreeNodeFlags_DefaultOpen))
	{
		static int type = static_cast<int>(m_environment.m_environmentType);
		if (ImGui::Combo("Environment type", &type, EnvironmentTypes, IM_ARRAYSIZE(EnvironmentTypes)))
		{
			m_environment.m_environmentType = static_cast<EnvironmentType>(type);
			modified = true;
		}
		switch (m_environment.m_environmentType)
		{
		case EnvironmentType::EnvironmentalMap: {
			if (editorLayer->DragAndDropButton<EnvironmentalMap>(
				m_environment.m_environmentalMap, "Environmental Map"))
				modified = true;
		}
											  break;
		case EnvironmentType::Color: {
			if (ImGui::ColorEdit3("Background Color", &m_environment.m_backgroundColor.x))
				modified = true;
		}
								   break;
		}
		if (ImGui::DragFloat(
			"Environmental light intensity", &m_environment.m_ambientLightIntensity, 0.01f, 0.0f, 10.0f))
			modified = true;
		if (ImGui::DragFloat("Environmental light gamma", &m_environment.m_environmentGamma, 0.01f, 0.0f, 10.0f))
		{
			modified = true;
		}
		ImGui::TreePop();
	}
	if (ImGui::TreeNodeEx("Systems"))
	{
		if (ImGui::BeginPopupContextWindow("SystemInspectorPopup"))
		{
			ImGui::Text("Add system: ");
			ImGui::Separator();
			static float rank = 0.0f;
			ImGui::DragFloat("Rank", &rank, 1.0f, 0.0f, 999.0f);
			for (auto& i : editorLayer->m_systemMenuList)
			{
				i.second(rank);
			}
			ImGui::Separator();
			ImGui::EndPopup();
		}
		for (auto& i : Application::GetActiveScene()->m_systems)
		{
			if (ImGui::CollapsingHeader(i.second->GetTypeName().c_str()))
			{
				bool enabled = i.second->Enabled();
				if (ImGui::Checkbox("Enabled", &enabled))
				{
					if (i.second->Enabled() != enabled)
					{
						if (enabled)
						{
							i.second->Enable();
							modified = true;
						}
						else
						{
							i.second->Disable();
							modified = true;
						}
					}
				}
				i.second->OnInspect(editorLayer);
			}
		}
		ImGui::TreePop();
	}
	return modified;
}

std::shared_ptr<ISystem> Scene::GetOrCreateSystem(const std::string& systemName, float order)
{
	size_t typeId;
	auto ptr = Serialization::ProduceSerializable(systemName, typeId);
	auto system = std::dynamic_pointer_cast<ISystem>(ptr);
	system->m_handle = Handle();
	system->m_rank = order;
	m_systems.insert({ order, system });
	m_indexedSystems[typeId] = system;
	m_mappedSystems[system->m_handle] = system;
	system->m_started = false;
	system->OnCreate();
	SetUnsaved();
	return std::dynamic_pointer_cast<ISystem>(ptr);
}

void Scene::Serialize(YAML::Emitter& out) const
{
	out << YAML::Key << "m_environment" << YAML::Value << YAML::BeginMap;
	m_environment.Serialize(out);
	out << YAML::EndMap;
	m_mainCamera.Save("m_mainCamera", out);
	std::unordered_map<Handle, std::shared_ptr<IAsset>> assetMap;
	std::vector<AssetRef> list;
	list.push_back(m_environment.m_environmentalMap);
	auto& sceneDataStorage = m_sceneDataStorage;
#pragma region EntityInfo
	out << YAML::Key << "m_entityMetadataList" << YAML::Value << YAML::BeginSeq;
	for (int i = 1; i < sceneDataStorage.m_entityMetadataList.size(); i++)
	{
		auto& entityMetadata = sceneDataStorage.m_entityMetadataList[i];
		if (entityMetadata.m_handle == 0)
			continue;
		for (const auto& element : entityMetadata.m_privateComponentElements)
		{
			element.m_privateComponentData->CollectAssetRef(list);
		}
		entityMetadata.Serialize(out, std::dynamic_pointer_cast<Scene>(GetSelf()));
	}
	out << YAML::EndSeq;
#pragma endregion

#pragma region Systems
	out << YAML::Key << "m_systems" << YAML::Value << YAML::BeginSeq;
	for (const auto& i : m_systems)
	{
		SerializeSystem(i.second, out);
		i.second->CollectAssetRef(list);
	}
	out << YAML::EndSeq;
#pragma endregion

#pragma region Assets
	for (auto& i : list)
	{
		const auto asset = i.Get<IAsset>();

		if (asset && !Resources::IsResource(asset->GetHandle()))
		{
			if (asset->IsTemporary())
			{
				assetMap[asset->GetHandle()] = asset;
			}
			else if (!asset->Saved())
			{
				asset->Save();
			}
		}
	}
	bool listCheck = true;
	while (listCheck)
	{
		size_t currentSize = assetMap.size();
		list.clear();
		for (auto& i : assetMap)
		{
			i.second->CollectAssetRef(list);
		}
		for (auto& i : list)
		{
			auto asset = i.Get<IAsset>();
			if (asset && !Resources::IsResource(asset->GetHandle()))
			{
				if (asset->IsTemporary())
				{
					assetMap[asset->GetHandle()] = asset;
				}
				else if (!asset->Saved())
				{
					asset->Save();
				}
			}
		}
		if (assetMap.size() == currentSize)
			listCheck = false;
	}
	if (!assetMap.empty())
	{
		out << YAML::Key << "LocalAssets" << YAML::Value << YAML::BeginSeq;
		for (auto& i : assetMap)
		{
			out << YAML::BeginMap;
			out << YAML::Key << "m_typeName" << YAML::Value << i.second->GetTypeName();
			out << YAML::Key << "m_handle" << YAML::Value << i.second->GetHandle();
			i.second->Serialize(out);
			out << YAML::EndMap;
		}
		out << YAML::EndSeq;
	}
#pragma endregion

#pragma region DataComponentStorage
	out << YAML::Key << "m_dataComponentStorages" << YAML::Value << YAML::BeginSeq;
	for (int i = 1; i < sceneDataStorage.m_dataComponentStorages.size(); i++)
	{
		SerializeDataComponentStorage(sceneDataStorage.m_dataComponentStorages[i], out);
	}
	out << YAML::EndSeq;
#pragma endregion
	out << YAML::EndMap;
}
void Scene::Deserialize(const YAML::Node& in)
{
	Purge();
	auto scene = std::dynamic_pointer_cast<Scene>(GetSelf());
	m_sceneDataStorage.m_entities.clear();
	m_sceneDataStorage.m_entityMetadataList.clear();
	m_sceneDataStorage.m_dataComponentStorages.clear();
	m_sceneDataStorage.m_entities.emplace_back();
	m_sceneDataStorage.m_entityMetadataList.emplace_back();
	m_sceneDataStorage.m_dataComponentStorages.emplace_back();

#pragma region EntityMetadata
	auto inEntityMetadataList = in["m_entityMetadataList"];
	int currentIndex = 1;
	for (const auto& inEntityMetadata : inEntityMetadataList)
	{
		m_sceneDataStorage.m_entityMetadataList.emplace_back();
		auto& newInfo = m_sceneDataStorage.m_entityMetadataList.back();
		newInfo.Deserialize(inEntityMetadata, scene);
		Entity entity;
		entity.m_version = 1;
		entity.m_index = currentIndex;
		m_sceneDataStorage.m_entityMap[newInfo.m_handle] = entity;
		m_sceneDataStorage.m_entities.push_back(entity);
		currentIndex++;
	}
	currentIndex = 1;
	for (const auto& inEntityMetadata : inEntityMetadataList)
	{
		auto& metadata = m_sceneDataStorage.m_entityMetadataList[currentIndex];
		if (inEntityMetadata["Parent.Handle"])
		{
			metadata.m_parent =
				m_sceneDataStorage.m_entityMap[Handle(inEntityMetadata["Parent.Handle"].as<uint64_t>())];
			auto& parentMetadata = m_sceneDataStorage.m_entityMetadataList[metadata.m_parent.m_index];
			Entity entity;
			entity.m_version = 1;
			entity.m_index = currentIndex;
			parentMetadata.m_children.push_back(entity);
		}
		if (inEntityMetadata["Root.Handle"])
			metadata.m_root = m_sceneDataStorage.m_entityMap[Handle(inEntityMetadata["Root.Handle"].as<uint64_t>())];
		currentIndex++;
	}
#pragma endregion

#pragma region DataComponentStorage
	auto inDataComponentStorages = in["m_dataComponentStorages"];
	int storageIndex = 1;
	for (const auto& inDataComponentStorage : inDataComponentStorages)
	{
		m_sceneDataStorage.m_dataComponentStorages.emplace_back();
		auto& dataComponentStorage = m_sceneDataStorage.m_dataComponentStorages.back();
		dataComponentStorage.m_entitySize = inDataComponentStorage["m_entitySize"].as<size_t>();
		dataComponentStorage.m_chunkCapacity = inDataComponentStorage["m_chunkCapacity"].as<size_t>();
		dataComponentStorage.m_entityAliveCount = dataComponentStorage.m_entityCount =
			inDataComponentStorage["m_entityAliveCount"].as<size_t>();
		dataComponentStorage.m_chunkArray.m_entities.resize(dataComponentStorage.m_entityAliveCount);
		const size_t chunkSize = dataComponentStorage.m_entityCount / dataComponentStorage.m_chunkCapacity + 1;
		while (dataComponentStorage.m_chunkArray.m_chunks.size() <= chunkSize)
		{
			// Allocate new chunk;
			ComponentDataChunk chunk = {};
			chunk.m_data = static_cast<void*>(calloc(1, Entities::GetArchetypeChunkSize()));
			dataComponentStorage.m_chunkArray.m_chunks.push_back(chunk);
		}
		auto inDataComponentTypes = inDataComponentStorage["m_dataComponentTypes"];
		for (const auto& inDataComponentType : inDataComponentTypes)
		{
			DataComponentType dataComponentType;
			dataComponentType.m_name = inDataComponentType["m_name"].as<std::string>();
			dataComponentType.m_size = inDataComponentType["m_size"].as<size_t>();
			dataComponentType.m_offset = inDataComponentType["m_offset"].as<size_t>();
			dataComponentType.m_typeId = Serialization::GetDataComponentTypeId(dataComponentType.m_name);
			dataComponentStorage.m_dataComponentTypes.push_back(dataComponentType);
		}
		auto inDataChunkArray = inDataComponentStorage["m_chunkArray"];
		int chunkArrayIndex = 0;
		for (const auto& entityDataComponent : inDataChunkArray)
		{
			Handle handle = entityDataComponent["m_handle"].as<uint64_t>();
			Entity entity = m_sceneDataStorage.m_entityMap[handle];
			dataComponentStorage.m_chunkArray.m_entities[chunkArrayIndex] = entity;
			auto& metadata = m_sceneDataStorage.m_entityMetadataList[entity.m_index];
			metadata.m_dataComponentStorageIndex = storageIndex;
			metadata.m_chunkArrayIndex = chunkArrayIndex;
			const auto chunkIndex = metadata.m_chunkArrayIndex / dataComponentStorage.m_chunkCapacity;
			const auto chunkPointer = metadata.m_chunkArrayIndex % dataComponentStorage.m_chunkCapacity;
			const auto chunk = dataComponentStorage.m_chunkArray.m_chunks[chunkIndex];

			int typeIndex = 0;
			for (const auto& inDataComponent : entityDataComponent["DataComponents"])
			{
				auto& type = dataComponentStorage.m_dataComponentTypes[typeIndex];
				auto data = inDataComponent["Data"].as<YAML::Binary>();
				std::memcpy(
					chunk.GetDataPointer(static_cast<size_t>(
						type.m_offset * dataComponentStorage.m_chunkCapacity + chunkPointer * type.m_size)),
					data.data(),
					data.size());
				typeIndex++;
			}

			chunkArrayIndex++;
		}
		storageIndex++;
	}
	auto self = std::dynamic_pointer_cast<Scene>(GetSelf());
#pragma endregion
	m_mainCamera.Load("m_mainCamera", in, self);
#pragma region Assets
	std::vector<std::pair<int, std::shared_ptr<IAsset>>> localAssets;
	if (const auto inLocalAssets = in["LocalAssets"])
	{
		int index = 0;
		for (const auto& i : inLocalAssets)
		{
			// First, find the asset in assetregistry
			if (const auto typeName = i["m_typeName"].as<std::string>(); Serialization::HasSerializableType(typeName)) {
				auto asset =
					ProjectManager::CreateTemporaryAsset(typeName, i["m_handle"].as<uint64_t>());
				localAssets.emplace_back(index, asset);
			}
			index++;
		}

		for (const auto& i : localAssets)
		{
			i.second->Deserialize(inLocalAssets[i.first]);
		}
	}

#pragma endregion
	if (in["m_environment"])
		m_environment.Deserialize(in["m_environment"]);
	int entityIndex = 1;
	for (const auto& inEntityInfo : inEntityMetadataList)
	{
		auto& entityMetadata = m_sceneDataStorage.m_entityMetadataList.at(entityIndex);
		auto entity = m_sceneDataStorage.m_entities[entityIndex];
		if (auto inPrivateComponents = inEntityInfo["m_privateComponentElements"])
		{
			for (const auto& inPrivateComponent : inPrivateComponents)
			{
				const auto name = inPrivateComponent["m_typeName"].as<std::string>();
				size_t hashCode;
				if (Serialization::HasSerializableType(name))
				{
					auto ptr =
						std::static_pointer_cast<IPrivateComponent>(Serialization::ProduceSerializable(name, hashCode));
					ptr->m_enabled = inPrivateComponent["m_enabled"].as<bool>();
					ptr->m_started = false;
					m_sceneDataStorage.m_entityPrivateComponentStorage.SetPrivateComponent(entity, hashCode);
					entityMetadata.m_privateComponentElements.emplace_back(hashCode, ptr, entity, self);
				}
				else
				{
					auto ptr = std::static_pointer_cast<IPrivateComponent>(
						Serialization::ProduceSerializable("UnknownPrivateComponent", hashCode));
					ptr->m_enabled = false;
					ptr->m_started = false;
					std::dynamic_pointer_cast<UnknownPrivateComponent>(ptr)->m_originalTypeName = name;
					m_sceneDataStorage.m_entityPrivateComponentStorage.SetPrivateComponent(entity, hashCode);
					entityMetadata.m_privateComponentElements.emplace_back(hashCode, ptr, entity, self);
				}
			}
		}
		entityIndex++;
	}

#pragma region Systems
	if (auto inSystems = in["m_systems"]) {
		std::vector<std::pair<int, std::shared_ptr<ISystem>>> systems;
		int index = 0;
		for (const auto& inSystem : inSystems)
		{
			if (const auto typeName = inSystem["m_typeName"].as<std::string>(); Serialization::HasSerializableType(typeName)) {
				size_t hashCode;
				if (const auto ptr = std::static_pointer_cast<ISystem>(Serialization::ProduceSerializable(typeName, hashCode)))
				{
					ptr->m_handle = Handle(inSystem["m_handle"].as<uint64_t>());
					ptr->m_enabled = inSystem["m_enabled"].as<bool>();
					ptr->m_rank = inSystem["m_rank"].as<float>();
					ptr->m_started = false;
					m_systems.insert({ ptr->m_rank, ptr });
					m_indexedSystems.insert({ hashCode, ptr });
					m_mappedSystems[ptr->m_handle] = ptr;
					systems.emplace_back(index, ptr);
					ptr->m_scene = self;
					ptr->OnCreate();
				}
			}
			index++;
		}
#pragma endregion

		entityIndex = 1;
		for (const auto& inEntityInfo : inEntityMetadataList)
		{
			auto& entityInfo = m_sceneDataStorage.m_entityMetadataList.at(entityIndex);
			auto inPrivateComponents = inEntityInfo["m_privateComponentElements"];
			int componentIndex = 0;
			if (inPrivateComponents)
			{
				for (const auto& inPrivateComponent : inPrivateComponents)
				{
					auto name = inPrivateComponent["m_typeName"].as<std::string>();
					auto ptr = entityInfo.m_privateComponentElements[componentIndex].m_privateComponentData;
					ptr->Deserialize(inPrivateComponent);
					componentIndex++;
				}
			}
			entityIndex++;
		}

		for (const auto& i : systems)
		{
			i.second->Deserialize(inSystems[i.first]);
		}
	}
}
void Scene::SerializeDataComponentStorage(const DataComponentStorage& storage, YAML::Emitter& out) const
{
	out << YAML::BeginMap;
	{
		out << YAML::Key << "m_entitySize" << YAML::Value << storage.m_entitySize;
		out << YAML::Key << "m_chunkCapacity" << YAML::Value << storage.m_chunkCapacity;
		out << YAML::Key << "m_entityAliveCount" << YAML::Value << storage.m_entityAliveCount;
		out << YAML::Key << "m_dataComponentTypes" << YAML::Value << YAML::BeginSeq;
		for (const auto& i : storage.m_dataComponentTypes)
		{
			out << YAML::BeginMap;
			out << YAML::Key << "m_name" << YAML::Value << i.m_name;
			out << YAML::Key << "m_size" << YAML::Value << i.m_size;
			out << YAML::Key << "m_offset" << YAML::Value << i.m_offset;
			out << YAML::EndMap;
		}
		out << YAML::EndSeq;

		out << YAML::Key << "m_chunkArray" << YAML::Value << YAML::BeginSeq;
		for (int i = 0; i < storage.m_entityAliveCount; i++)
		{
			auto entity = storage.m_chunkArray.m_entities[i];
			if (entity.m_version == 0)
				continue;

			out << YAML::BeginMap;
			auto& entityInfo = m_sceneDataStorage.m_entityMetadataList.at(entity.m_index);
			out << YAML::Key << "m_handle" << YAML::Value << entityInfo.m_handle;

			auto& dataComponentStorage =
				m_sceneDataStorage.m_dataComponentStorages[entityInfo.m_dataComponentStorageIndex];
			const auto chunkIndex = entityInfo.m_chunkArrayIndex / dataComponentStorage.m_chunkCapacity;
			const auto chunkPointer = entityInfo.m_chunkArrayIndex % dataComponentStorage.m_chunkCapacity;
			const auto chunk = dataComponentStorage.m_chunkArray.m_chunks[chunkIndex];

			out << YAML::Key << "DataComponents" << YAML::Value << YAML::BeginSeq;
			for (const auto& type : dataComponentStorage.m_dataComponentTypes)
			{
				out << YAML::BeginMap;
				out << YAML::Key << "Data" << YAML::Value
					<< YAML::Binary(
						(const unsigned char*)chunk.GetDataPointer(type.m_offset * dataComponentStorage.m_chunkCapacity + chunkPointer * type.m_size),
						type.m_size);
				out << YAML::EndMap;
			}
			out << YAML::EndSeq;

			out << YAML::EndMap;
		}
		out << YAML::EndSeq;
	}
	out << YAML::EndMap;
}
void Scene::SerializeSystem(const std::shared_ptr<ISystem>& system, YAML::Emitter& out)
{
	out << YAML::BeginMap;
	{
		out << YAML::Key << "m_typeName" << YAML::Value << system->GetTypeName();
		out << YAML::Key << "m_enabled" << YAML::Value << system->m_enabled;
		out << YAML::Key << "m_rank" << YAML::Value << system->m_rank;
		out << YAML::Key << "m_handle" << YAML::Value << system->GetHandle();
		system->Serialize(out);
	}
	out << YAML::EndMap;
}

void Scene::OnCreate()
{
	m_sceneDataStorage.m_entities.emplace_back();
	m_sceneDataStorage.m_entityMetadataList.emplace_back();
	m_sceneDataStorage.m_dataComponentStorages.emplace_back();
	m_sceneDataStorage.m_entityPrivateComponentStorage.m_scene = std::dynamic_pointer_cast<Scene>(GetSelf());
	
#pragma region Main Camera
	const auto mainCameraEntity = CreateEntity("Main Camera");
	Transform ltw;
	ltw.SetPosition(glm::vec3(0.0f, 5.0f, 10.0f));
	ltw.SetScale(glm::vec3(1, 1, 1));
	ltw.SetEulerRotation(glm::radians(glm::vec3(0, 0, 0)));
	SetDataComponent(mainCameraEntity, ltw);
	auto mainCameraComponent = GetOrSetPrivateComponent<Camera>(mainCameraEntity).lock();
	m_mainCamera = mainCameraComponent;
	mainCameraComponent->m_skybox = Resources::GetResource<Cubemap>("DEFAULT_SKYBOX");
#pragma endregion

#pragma region Directional Light
	const auto directionalLightEntity = CreateEntity("Directional Light");
	ltw.SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
	ltw.SetEulerRotation(glm::radians(glm::vec3(90, 0, 0)));
	SetDataComponent(directionalLightEntity, ltw);
	auto directionLight = GetOrSetPrivateComponent<DirectionalLight>(directionalLightEntity).lock();
#pragma endregion

#pragma region Ground
	const auto groundEntity = CreateEntity("Ground");
	ltw.SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
	ltw.SetScale(glm::vec3(10, 1, 10));
	ltw.SetEulerRotation(glm::radians(glm::vec3(0, 0, 0)));
	SetDataComponent(groundEntity, ltw);
	auto groundMeshRendererComponent = GetOrSetPrivateComponent<MeshRenderer>(groundEntity).lock();
	groundMeshRendererComponent->m_material = ProjectManager::CreateTemporaryAsset<Material>();
	groundMeshRendererComponent->m_mesh = Resources::GetResource<Mesh>("PRIMITIVE_QUAD");
#pragma endregion

}



bool Scene::LoadInternal(const std::filesystem::path& path)
{
	auto previousScene = Application::GetActiveScene();
	Application::Attach(std::shared_ptr<Scene>(this, [](Scene*) {}));
	std::ifstream stream(path.string());
	std::stringstream stringStream;
	stringStream << stream.rdbuf();
	YAML::Node in = YAML::Load(stringStream.str());
	Deserialize(in);
	Application::Attach(previousScene);
	return true;
}
void Scene::Clone(const std::shared_ptr<Scene>& source, const std::shared_ptr<Scene>& newScene)
{
	newScene->m_environment = source->m_environment;
	newScene->m_saved = source->m_saved;
	newScene->m_worldBound = source->m_worldBound;
	std::unordered_map<Handle, Handle> entityMap;

	newScene->m_sceneDataStorage.Clone(entityMap, source->m_sceneDataStorage, newScene);
	for (const auto& i : source->m_systems)
	{
		auto systemName = i.second->GetTypeName();
		size_t hashCode;
		auto system = std::dynamic_pointer_cast<ISystem>(
			Serialization::ProduceSerializable(systemName, hashCode, i.second->GetHandle()));
		newScene->m_systems.insert({ i.first, system });
		newScene->m_indexedSystems[hashCode] = system;
		newScene->m_mappedSystems[i.second->GetHandle()] = system;
		system->m_scene = newScene;
		system->OnCreate();
		Serialization::CloneSystem(system, i.second);
		system->m_scene = newScene;
	}
	newScene->m_mainCamera.m_entityHandle = source->m_mainCamera.m_entityHandle;
	newScene->m_mainCamera.m_privateComponentTypeName = source->m_mainCamera.m_privateComponentTypeName;
	newScene->m_mainCamera.Relink(entityMap, newScene);
}

std::shared_ptr<LightProbe> Environment::GetLightProbe(const glm::vec3& position)
{
	if (const auto environmentalMap = m_environmentalMap.Get<EnvironmentalMap>())
	{
		if (auto lightProbe = environmentalMap->m_lightProbe.Get<LightProbe>()) return lightProbe;
	}
	return nullptr;
}

std::shared_ptr<ReflectionProbe> Environment::GetReflectionProbe(const glm::vec3& position)
{
	if (const auto environmentalMap = m_environmentalMap.Get<EnvironmentalMap>())
	{
		if (auto reflectionProbe = environmentalMap->m_lightProbe.Get<ReflectionProbe>()) return reflectionProbe;
	}
	return nullptr;
}

void Environment::Serialize(YAML::Emitter& out) const
{
	out << YAML::Key << "m_backgroundColor" << YAML::Value << m_backgroundColor;
	out << YAML::Key << "m_environmentGamma" << YAML::Value << m_environmentGamma;
	out << YAML::Key << "m_ambientLightIntensity" << YAML::Value << m_ambientLightIntensity;
	out << YAML::Key << "m_environmentType" << YAML::Value << static_cast<unsigned>(m_environmentType);
	m_environmentalMap.Save("m_environment", out);
}
void Environment::Deserialize(const YAML::Node& in)
{
	if (in["m_backgroundColor"])
		m_backgroundColor = in["m_backgroundColor"].as<glm::vec3>();
	if (in["m_environmentGamma"])
		m_environmentGamma = in["m_environmentGamma"].as<float>();
	if (in["m_ambientLightIntensity"])
		m_ambientLightIntensity = in["m_ambientLightIntensity"].as<float>();
	if (in["m_environmentType"])
		m_environmentType = (EnvironmentType)in["m_environmentType"].as<unsigned>();
	m_environmentalMap.Load("m_environment", in);
}
void SceneDataStorage::Clone(
	std::unordered_map<Handle, Handle>& entityMap,
	const SceneDataStorage& source,
	const std::shared_ptr<Scene>& newScene)
{
	m_entities = source.m_entities;
	m_entityMetadataList.resize(source.m_entityMetadataList.size());

	for (const auto& i : source.m_entityMetadataList)
	{
		entityMap.insert({ i.m_handle, i.m_handle });
	}
	m_dataComponentStorages.resize(source.m_dataComponentStorages.size());
	for (int i = 0; i < m_dataComponentStorages.size(); i++)
		m_dataComponentStorages[i] = source.m_dataComponentStorages[i];
	for (int i = 0; i < m_entityMetadataList.size(); i++)
		m_entityMetadataList[i].Clone(entityMap, source.m_entityMetadataList[i], newScene);

	m_entityMap = source.m_entityMap;
	m_entityPrivateComponentStorage = source.m_entityPrivateComponentStorage;
	m_entityPrivateComponentStorage.m_scene = newScene;
}

KeyActionType Scene::GetKey(int key)
{
	const auto search = m_pressedKeys.find(key);
	if (search != m_pressedKeys.end()) return search->second;
	return KeyActionType::Release;
}

#pragma region Entity Management
void Scene::UnsafeForEachDataComponent(
	const Entity& entity, const std::function<void(const DataComponentType& type, void* data)>& func)
{
	assert(IsEntityValid(entity));
	EntityMetadata& entityInfo = m_sceneDataStorage.m_entityMetadataList.at(entity.m_index);
	auto& dataComponentStorage = m_sceneDataStorage.m_dataComponentStorages[entityInfo.m_dataComponentStorageIndex];
	const size_t chunkIndex = entityInfo.m_chunkArrayIndex / dataComponentStorage.m_chunkCapacity;
	const size_t chunkPointer = entityInfo.m_chunkArrayIndex % dataComponentStorage.m_chunkCapacity;
	const ComponentDataChunk& chunk = dataComponentStorage.m_chunkArray.m_chunks[chunkIndex];
	for (const auto& i : dataComponentStorage.m_dataComponentTypes)
	{
		func(
			i,
			static_cast<void*>(
				static_cast<char*>(chunk.m_data) + i.m_offset * dataComponentStorage.m_chunkCapacity +
				chunkPointer * i.m_size));
	}
}

void Scene::ForEachPrivateComponent(
	const Entity& entity, const std::function<void(PrivateComponentElement& data)>& func)
{
	assert(IsEntityValid(entity));
	auto elements = m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_privateComponentElements;
	for (auto& component : elements)
	{
		func(component);
	}
}

void Scene::UnsafeForEachEntityStorage(
	const std::function<void(int i, const std::string& name, const DataComponentStorage& storage)>& func)
{
	auto& archetypeInfos = Entities::GetInstance().m_entityArchetypeInfos;
	for (int i = 0; i < archetypeInfos.size(); i++)
	{
		auto dcs = GetDataComponentStorage(i);
		if (!dcs.has_value())
			continue;
		func(i, archetypeInfos[i].m_name, dcs->first.get());
	}
}

void Scene::DeleteEntityInternal(unsigned entityIndex)
{
	EntityMetadata& entityInfo = m_sceneDataStorage.m_entityMetadataList.at(entityIndex);
	auto& dataComponentStorage = m_sceneDataStorage.m_dataComponentStorages[entityInfo.m_dataComponentStorageIndex];
	Entity actualEntity = m_sceneDataStorage.m_entities.at(entityIndex);

	m_sceneDataStorage.m_entityPrivateComponentStorage.DeleteEntity(actualEntity);
	entityInfo.m_version = actualEntity.m_version + 1;
	entityInfo.m_enabled = true;
	entityInfo.m_static = false;
	entityInfo.m_ancestorSelected = false;
	m_sceneDataStorage.m_entityMap.erase(entityInfo.m_handle);
	entityInfo.m_handle = Handle(0);

	entityInfo.m_privateComponentElements.clear();
	// Set to version 0, marks it as deleted.
	actualEntity.m_version = 0;
	dataComponentStorage.m_chunkArray.m_entities[entityInfo.m_chunkArrayIndex] = actualEntity;
	const auto originalIndex = entityInfo.m_chunkArrayIndex;
	if (entityInfo.m_chunkArrayIndex != dataComponentStorage.m_entityAliveCount - 1)
	{
		const auto swappedIndex =
			SwapEntity(dataComponentStorage, entityInfo.m_chunkArrayIndex, dataComponentStorage.m_entityAliveCount - 1);
		entityInfo.m_chunkArrayIndex = dataComponentStorage.m_entityAliveCount - 1;
		m_sceneDataStorage.m_entityMetadataList.at(swappedIndex).m_chunkArrayIndex = originalIndex;
	}
	dataComponentStorage.m_entityAliveCount--;

	m_sceneDataStorage.m_entities.at(entityIndex) = actualEntity;
}

std::optional<std::pair<std::reference_wrapper<DataComponentStorage>, unsigned>> Scene::GetDataComponentStorage(
	unsigned entityArchetypeIndex)
{
	auto& archetypeInfo = Entities::GetInstance().m_entityArchetypeInfos.at(entityArchetypeIndex);
	int targetIndex = 0;
	for (auto& i : m_sceneDataStorage.m_dataComponentStorages)
	{
		if (i.m_dataComponentTypes.size() != archetypeInfo.m_dataComponentTypes.size())
		{
			targetIndex++;
			continue;
		}
		bool check = true;
		for (int j = 0; j < i.m_dataComponentTypes.size(); j++)
		{
			if (i.m_dataComponentTypes[j].m_name != archetypeInfo.m_dataComponentTypes[j].m_name)
			{
				check = false;
				break;
			}
		}
		if (check)
		{
			return { {std::ref(i), targetIndex} };
		}
		targetIndex++;
	}
	// If we didn't find the target storage, then we need to create a new one.
	m_sceneDataStorage.m_dataComponentStorages.emplace_back(archetypeInfo);
	return {
		{std::ref(m_sceneDataStorage.m_dataComponentStorages.back()),
		 m_sceneDataStorage.m_dataComponentStorages.size() - 1} };
}

std::optional<std::pair<std::reference_wrapper<DataComponentStorage>, unsigned>> Scene::GetDataComponentStorage(
	const EntityArchetype& entityArchetype)
{
	return GetDataComponentStorage(entityArchetype.m_index);
}

std::vector<std::reference_wrapper<DataComponentStorage>> Scene::QueryDataComponentStorages(
	const EntityQuery& entityQuery)
{
	return QueryDataComponentStorages(entityQuery.m_index);
}

void Scene::GetEntityStorage(const DataComponentStorage& storage, std::vector<Entity>& container, const bool checkEnable) const
{
	const size_t amount = storage.m_entityAliveCount;
	if (amount == 0)
		return;
	if (checkEnable)
	{
		const auto threadSize = Jobs::GetWorkerSize();
		std::vector<std::vector<Entity>> tempStorage;
		tempStorage.resize(threadSize);
		const auto& chunkArray = storage.m_chunkArray;
		const auto& entities = &chunkArray.m_entities;
		Jobs::RunParallelFor(amount, [=, &entities, &tempStorage](const int i, const unsigned workerIndex) {
			const auto entity = entities->at(i);
			if (!m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_enabled)
				return;
			tempStorage[workerIndex].push_back(entity);
			},
			threadSize
		);
		for (auto& i : tempStorage)
		{
			container.insert(container.end(), i.begin(), i.end());
		}
	}
	else
	{
		container.resize(container.size() + amount);
		const size_t capacity = storage.m_chunkCapacity;
		memcpy(
			&container.at(container.size() - amount), storage.m_chunkArray.m_entities.data(), amount * sizeof(Entity));
	}
}

auto Scene::SwapEntity(DataComponentStorage& storage, const size_t index1, const size_t index2) -> size_t
{
	if (index1 == index2)
		return -1;
	const size_t retVal = storage.m_chunkArray.m_entities[index2].m_index;
	const auto other = storage.m_chunkArray.m_entities[index2];
	storage.m_chunkArray.m_entities[index2] = storage.m_chunkArray.m_entities[index1];
	storage.m_chunkArray.m_entities[index1] = other;
	const auto capacity = storage.m_chunkCapacity;
	const auto chunkIndex1 = index1 / capacity;
	const auto chunkIndex2 = index2 / capacity;
	const auto chunkPointer1 = index1 % capacity;
	const auto chunkPointer2 = index2 % capacity;
	for (const auto& i : storage.m_dataComponentTypes)
	{
		void* temp = static_cast<void*>(malloc(i.m_size));
		void* d1 = static_cast<void*>(
			static_cast<char*>(storage.m_chunkArray.m_chunks[chunkIndex1].m_data) + i.m_offset * capacity +
			i.m_size * chunkPointer1);

		void* d2 = static_cast<void*>(
			static_cast<char*>(storage.m_chunkArray.m_chunks[chunkIndex2].m_data) + i.m_offset * capacity +
			i.m_size * chunkPointer2);

		memcpy(temp, d1, i.m_size);
		memcpy(d1, d2, i.m_size);
		memcpy(d2, temp, i.m_size);
		free(temp);
	}
	return retVal;
}

void Scene::GetAllEntities(std::vector<Entity>& target)
{
	target.insert(target.end(), m_sceneDataStorage.m_entities.begin() + 1, m_sceneDataStorage.m_entities.end());
}

void Scene::ForEachDescendant(
	const Entity& target, const std::function<void(const Entity& entity)>& func, const bool& fromRoot)
{
	Entity realTarget = target;
	if (!IsEntityValid(realTarget))
		return;
	if (fromRoot)
		realTarget = GetRoot(realTarget);
	ForEachDescendantHelper(realTarget, func);
}

const std::vector<Entity>& Scene::UnsafeGetAllEntities()
{
	return m_sceneDataStorage.m_entities;
}

Entity Scene::CreateEntity(const std::string& name)
{
	return CreateEntity(Entities::GetInstance().m_basicArchetype, name);
}

Entity Scene::CreateEntity(const EntityArchetype& archetype, const std::string& name, const Handle& handle)
{
	assert(archetype.IsValid());

	Entity retVal;
	auto search = GetDataComponentStorage(archetype);
	DataComponentStorage& storage = search->first;
	if (storage.m_entityCount == storage.m_entityAliveCount)
	{
		const size_t chunkIndex = storage.m_entityCount / storage.m_chunkCapacity + 1;
		if (storage.m_chunkArray.m_chunks.size() <= chunkIndex)
		{
			// Allocate new chunk;
			ComponentDataChunk chunk;
			chunk.m_data = static_cast<void*>(calloc(1, Entities::GetArchetypeChunkSize()));
			storage.m_chunkArray.m_chunks.push_back(chunk);
		}
		retVal.m_index = m_sceneDataStorage.m_entities.size();
		// If the version is 0 in chunk means it's deleted.
		retVal.m_version = 1;
		EntityMetadata entityInfo;
		entityInfo.m_root = retVal;
		entityInfo.m_static = false;
		entityInfo.m_name = name;
		entityInfo.m_handle = handle;
		entityInfo.m_dataComponentStorageIndex = search->second;
		entityInfo.m_chunkArrayIndex = storage.m_entityCount;
		storage.m_chunkArray.m_entities.push_back(retVal);

		m_sceneDataStorage.m_entityMap[entityInfo.m_handle] = retVal;
		m_sceneDataStorage.m_entityMetadataList.push_back(std::move(entityInfo));
		m_sceneDataStorage.m_entities.push_back(retVal);
		storage.m_entityCount++;
		storage.m_entityAliveCount++;
	}
	else
	{
		retVal = storage.m_chunkArray.m_entities.at(storage.m_entityAliveCount);
		EntityMetadata& entityInfo = m_sceneDataStorage.m_entityMetadataList.at(retVal.m_index);
		entityInfo.m_root = retVal;
		entityInfo.m_static = false;
		entityInfo.m_handle = handle;
		entityInfo.m_enabled = true;
		entityInfo.m_name = name;
		retVal.m_version = entityInfo.m_version;

		m_sceneDataStorage.m_entityMap[entityInfo.m_handle] = retVal;
		storage.m_chunkArray.m_entities[entityInfo.m_chunkArrayIndex] = retVal;
		m_sceneDataStorage.m_entities.at(retVal.m_index) = retVal;
		storage.m_entityAliveCount++;
		// Reset all component data
		const auto chunkIndex = entityInfo.m_chunkArrayIndex / storage.m_chunkCapacity;
		const auto chunkPointer = entityInfo.m_chunkArrayIndex % storage.m_chunkCapacity;
		const auto chunk = storage.m_chunkArray.m_chunks[chunkIndex];
		for (const auto& i : storage.m_dataComponentTypes)
		{
			const auto offset = i.m_offset * storage.m_chunkCapacity + chunkPointer * i.m_size;
			chunk.ClearData(offset, i.m_size);
		}
	}
	SetDataComponent(retVal, Transform());
	SetDataComponent(retVal, GlobalTransform());
	SetDataComponent(retVal, TransformUpdateFlag());
	SetUnsaved();
	return retVal;
}

std::vector<Entity> Scene::CreateEntities(
	const EntityArchetype& archetype, const size_t& amount, const std::string& name)
{
	assert(archetype.IsValid());
	std::vector<Entity> retVal;
	auto search = GetDataComponentStorage(archetype);
	DataComponentStorage& storage = search->first;
	auto remainAmount = amount;
	const Transform transform;
	const GlobalTransform globalTransform;
	const TransformUpdateFlag transformStatus;
	while (remainAmount > 0 && storage.m_entityAliveCount != storage.m_entityCount)
	{
		remainAmount--;
		Entity entity = storage.m_chunkArray.m_entities.at(storage.m_entityAliveCount);
		EntityMetadata& entityInfo = m_sceneDataStorage.m_entityMetadataList.at(entity.m_index);
		entityInfo.m_root = entity;
		entityInfo.m_static = false;
		entityInfo.m_enabled = true;
		entityInfo.m_name = name;
		entity.m_version = entityInfo.m_version;
		entityInfo.m_handle = Handle();
		m_sceneDataStorage.m_entityMap[entityInfo.m_handle] = entity;
		storage.m_chunkArray.m_entities[entityInfo.m_chunkArrayIndex] = entity;
		m_sceneDataStorage.m_entities.at(entity.m_index) = entity;
		storage.m_entityAliveCount++;
		// Reset all component data
		const size_t chunkIndex = entityInfo.m_chunkArrayIndex / storage.m_chunkCapacity;
		const size_t chunkPointer = entityInfo.m_chunkArrayIndex % storage.m_chunkCapacity;
		const ComponentDataChunk& chunk = storage.m_chunkArray.m_chunks[chunkIndex];
		for (const auto& i : storage.m_dataComponentTypes)
		{
			const size_t offset = i.m_offset * storage.m_chunkCapacity + chunkPointer * i.m_size;
			chunk.ClearData(offset, i.m_size);
		}
		retVal.push_back(entity);
		SetDataComponent(entity, transform);
		SetDataComponent(entity, globalTransform);
		SetDataComponent(entity, TransformUpdateFlag());
	}
	if (remainAmount == 0)
		return retVal;
	storage.m_entityCount += remainAmount;
	storage.m_entityAliveCount += remainAmount;
	const size_t chunkIndex = storage.m_entityCount / storage.m_chunkCapacity + 1;
	while (storage.m_chunkArray.m_chunks.size() <= chunkIndex)
	{
		// Allocate new chunk;
		ComponentDataChunk chunk;
		chunk.m_data = static_cast<void*>(calloc(1, Entities::GetArchetypeChunkSize()));
		storage.m_chunkArray.m_chunks.push_back(chunk);
	}
	const size_t originalSize = m_sceneDataStorage.m_entities.size();
	m_sceneDataStorage.m_entities.resize(originalSize + remainAmount);
	m_sceneDataStorage.m_entityMetadataList.resize(originalSize + remainAmount);

	for (int i = 0; i < remainAmount; i++)
	{
		auto& entity = m_sceneDataStorage.m_entities.at(originalSize + i);
		entity.m_index = originalSize + i;
		entity.m_version = 1;

		auto& entityInfo = m_sceneDataStorage.m_entityMetadataList.at(originalSize + i);
		entityInfo = EntityMetadata();
		entityInfo.m_root = entity;
		entityInfo.m_static = false;
		entityInfo.m_name = name;
		entityInfo.m_dataComponentStorageIndex = search->second;
		entityInfo.m_chunkArrayIndex = storage.m_entityAliveCount - remainAmount + i;

		entityInfo.m_handle = Handle();

		m_sceneDataStorage.m_entityMap[entityInfo.m_handle] = entity;
	}

	storage.m_chunkArray.m_entities.insert(
		storage.m_chunkArray.m_entities.end(),
		m_sceneDataStorage.m_entities.begin() + originalSize,
		m_sceneDataStorage.m_entities.end());
	Jobs::RunParallelFor(remainAmount, [&, originalSize](unsigned i)
		{
			const auto& entity = m_sceneDataStorage.m_entities.at(originalSize + i);
			SetDataComponent(entity, transform);
			SetDataComponent(entity, globalTransform);
			SetDataComponent(entity, TransformUpdateFlag());
		}
	);

	retVal.insert(
		retVal.end(), m_sceneDataStorage.m_entities.begin() + originalSize, m_sceneDataStorage.m_entities.end());
	SetUnsaved();
	return retVal;
}

std::vector<Entity> Scene::CreateEntities(const size_t& amount, const std::string& name)
{
	return CreateEntities(Entities::GetInstance().m_basicArchetype, amount, name);
}

void Scene::DeleteEntity(const Entity& entity)
{
	if (!IsEntityValid(entity))
	{
		return;
	}
	const size_t entityIndex = entity.m_index;
	auto children = m_sceneDataStorage.m_entityMetadataList.at(entityIndex).m_children;
	for (const auto& child : children)
	{
		DeleteEntity(child);
	}
	if (m_sceneDataStorage.m_entityMetadataList.at(entityIndex).m_parent.m_index != 0)
		RemoveChild(entity, m_sceneDataStorage.m_entityMetadataList.at(entityIndex).m_parent);
	DeleteEntityInternal(entity.m_index);
	SetUnsaved();
}

std::string Scene::GetEntityName(const Entity& entity)
{
	assert(IsEntityValid(entity));
	const size_t index = entity.m_index;
	if (entity != m_sceneDataStorage.m_entities.at(index))
	{
		EVOENGINE_ERROR("Child already deleted!");
		return "";
	}
	return m_sceneDataStorage.m_entityMetadataList.at(index).m_name;
}

void Scene::SetEntityName(const Entity& entity, const std::string& name)
{
	assert(IsEntityValid(entity));
	const size_t index = entity.m_index;
	if (entity != m_sceneDataStorage.m_entities.at(index))
	{
		EVOENGINE_ERROR("Child already deleted!");
		return;
	}
	if (name.length() != 0)
	{
		m_sceneDataStorage.m_entityMetadataList.at(index).m_name = name;
		return;
	}
	m_sceneDataStorage.m_entityMetadataList.at(index).m_name = "Unnamed";
	SetUnsaved();
}
void Scene::SetEntityStatic(const Entity& entity, bool value)
{
	assert(IsEntityValid(entity));
	auto& entityInfo = m_sceneDataStorage.m_entityMetadataList.at(GetRoot(entity).m_index);
	entityInfo.m_static = value;
	SetUnsaved();
}
void Scene::SetParent(const Entity& child, const Entity& parent, const bool& recalculateTransform)
{
	assert(IsEntityValid(child) && IsEntityValid(parent));
	const size_t childIndex = child.m_index;
	const size_t parentIndex = parent.m_index;
	auto& parentEntityInfo = m_sceneDataStorage.m_entityMetadataList.at(parentIndex);
	for (const auto& i : parentEntityInfo.m_children)
	{
		if (i == child)
			return;
	}
	auto& childEntityInfo = m_sceneDataStorage.m_entityMetadataList.at(childIndex);
	if (childEntityInfo.m_parent.GetIndex() != 0)
	{
		RemoveChild(child, childEntityInfo.m_parent);
	}

	if (recalculateTransform)
	{
		const auto childGlobalTransform = GetDataComponent<GlobalTransform>(child);
		const auto parentGlobalTransform = GetDataComponent<GlobalTransform>(parent);
		Transform childTransform;
		childTransform.m_value = glm::inverse(parentGlobalTransform.m_value) * childGlobalTransform.m_value;
		SetDataComponent(child, childTransform);
	}
	childEntityInfo.m_parent = parent;
	if (parentEntityInfo.m_parent.GetIndex() == childIndex)
	{
		parentEntityInfo.m_parent = Entity();
		parentEntityInfo.m_root = parent;
		const size_t childrenCount = childEntityInfo.m_children.size();

		for (size_t i = 0; i < childrenCount; i++)
		{
			if (childEntityInfo.m_children[i].m_index == parent.GetIndex())
			{
				childEntityInfo.m_children[i] = childEntityInfo.m_children.back();
				childEntityInfo.m_children.pop_back();
				break;
			}
		}
	}
	childEntityInfo.m_root = parentEntityInfo.m_root;
	childEntityInfo.m_static = false;
	parentEntityInfo.m_children.push_back(child);
	if (parentEntityInfo.m_ancestorSelected)
	{
		const auto descendants = GetDescendants(child);
		for (const auto& i : descendants)
		{
			GetEntityMetadata(i).m_ancestorSelected = true;
		}
		childEntityInfo.m_ancestorSelected = true;
	}
	SetUnsaved();
}

Entity Scene::GetParent(const Entity& entity) const
{
	assert(IsEntityValid(entity));
	const size_t entityIndex = entity.m_index;
	return m_sceneDataStorage.m_entityMetadataList.at(entityIndex).m_parent;
}

std::vector<Entity> Scene::GetChildren(const Entity& entity)
{
	assert(IsEntityValid(entity));
	const size_t entityIndex = entity.m_index;
	return m_sceneDataStorage.m_entityMetadataList.at(entityIndex).m_children;
}

Entity Scene::GetChild(const Entity& entity, int index) const
{
	assert(IsEntityValid(entity));
	const size_t entityIndex = entity.m_index;
	auto& children = m_sceneDataStorage.m_entityMetadataList.at(entityIndex).m_children;
	if (children.size() > index)
		return children[index];
	return Entity();
}

size_t Scene::GetChildrenAmount(const Entity& entity) const
{
	assert(IsEntityValid(entity));
	const size_t entityIndex = entity.m_index;
	return m_sceneDataStorage.m_entityMetadataList.at(entityIndex).m_children.size();
}

void Scene::ForEachChild(const Entity& entity, const std::function<void(Entity child)>& func) const
{
	assert(IsEntityValid(entity));
	const auto children = m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_children;
	for (auto i : children)
	{
		if (IsEntityValid(i))
			func(i);
	}
}

void Scene::RemoveChild(const Entity& child, const Entity& parent)
{
	assert(IsEntityValid(child) && IsEntityValid(parent));
	const size_t childIndex = child.m_index;
	const size_t parentIndex = parent.m_index;
	auto& childEntityMetadata = m_sceneDataStorage.m_entityMetadataList.at(childIndex);
	auto& parentEntityMetadata = m_sceneDataStorage.m_entityMetadataList.at(parentIndex);
	if (childEntityMetadata.m_parent.m_index == 0)
	{
		EVOENGINE_ERROR("No child by the parent!");
	}
	childEntityMetadata.m_parent = Entity();
	childEntityMetadata.m_root = child;
	if (parentEntityMetadata.m_ancestorSelected)
	{
		const auto descendants = GetDescendants(child);
		for (const auto& i : descendants)
		{
			GetEntityMetadata(i).m_ancestorSelected = false;
		}
		childEntityMetadata.m_ancestorSelected = false;
	}
	const size_t childrenCount = parentEntityMetadata.m_children.size();

	for (size_t i = 0; i < childrenCount; i++)
	{
		if (parentEntityMetadata.m_children[i].m_index == childIndex)
		{
			parentEntityMetadata.m_children[i] = parentEntityMetadata.m_children.back();
			parentEntityMetadata.m_children.pop_back();
			break;
		}
	}
	const auto childGlobalTransform = GetDataComponent<GlobalTransform>(child);
	Transform childTransform;
	childTransform.m_value = childGlobalTransform.m_value;
	SetDataComponent(child, childTransform);
	SetUnsaved();
}

void Scene::RemoveDataComponent(const Entity& entity, const size_t& typeID)
{
	assert(IsEntityValid(entity));
	if (typeID == typeid(Transform).hash_code() || typeID == typeid(GlobalTransform).hash_code() ||
		typeID == typeid(TransformUpdateFlag).hash_code())
	{
		return;
	}
	EntityMetadata& entityInfo = m_sceneDataStorage.m_entityMetadataList.at(entity.m_index);
	auto& entityArchetypeInfos = Entities::GetInstance().m_entityArchetypeInfos;
	auto& dataComponentStorage = m_sceneDataStorage.m_dataComponentStorages[entityInfo.m_dataComponentStorageIndex];
	if (dataComponentStorage.m_dataComponentTypes.size() <= 3)
	{
		EVOENGINE_ERROR("Remove Component Data failed: Entity must have at least 1 data component besides 3 basic data "
			"components!");
		return;
	}
#pragma region Create new archetype
	EntityArchetypeInfo newArchetypeInfo;
	newArchetypeInfo.m_name = "New archetype";
	newArchetypeInfo.m_dataComponentTypes = dataComponentStorage.m_dataComponentTypes;
	bool found = false;
	for (int i = 0; i < newArchetypeInfo.m_dataComponentTypes.size(); i++)
	{
		if (newArchetypeInfo.m_dataComponentTypes[i].m_typeId == typeID)
		{
			newArchetypeInfo.m_dataComponentTypes.erase(newArchetypeInfo.m_dataComponentTypes.begin() + i);
			found = true;
			break;
		}
	}
	if (!found)
	{
		EVOENGINE_ERROR("Failed to remove component data: Component not found");
		return;
	}
	size_t offset = 0;
	DataComponentType prev = newArchetypeInfo.m_dataComponentTypes[0];
	for (auto& i : newArchetypeInfo.m_dataComponentTypes)
	{
		i.m_offset = offset;
		offset += i.m_size;
	}
	newArchetypeInfo.m_entitySize =
		newArchetypeInfo.m_dataComponentTypes.back().m_offset + newArchetypeInfo.m_dataComponentTypes.back().m_size;
	newArchetypeInfo.m_chunkCapacity = Entities::GetArchetypeChunkSize() / newArchetypeInfo.m_entitySize;
	auto archetype = Entities::CreateEntityArchetypeHelper(newArchetypeInfo);
#pragma endregion
#pragma region Create new Entity with new archetype
	const Entity newEntity = CreateEntity(archetype);
	// Transfer component data
	for (const auto& type : newArchetypeInfo.m_dataComponentTypes)
	{
		SetDataComponent(newEntity.m_index, type.m_typeId, type.m_size, GetDataComponentPointer(entity, type.m_typeId));
	}
	// 5. Swap entity.
	EntityMetadata& newEntityInfo = m_sceneDataStorage.m_entityMetadataList.at(newEntity.m_index);
	const auto tempArchetypeInfoIndex = newEntityInfo.m_dataComponentStorageIndex;
	const auto tempChunkArrayIndex = newEntityInfo.m_chunkArrayIndex;
	newEntityInfo.m_dataComponentStorageIndex = entityInfo.m_dataComponentStorageIndex;
	newEntityInfo.m_chunkArrayIndex = entityInfo.m_chunkArrayIndex;
	entityInfo.m_dataComponentStorageIndex = tempArchetypeInfoIndex;
	entityInfo.m_chunkArrayIndex = tempChunkArrayIndex;
	// Apply to chunk.
	m_sceneDataStorage.m_dataComponentStorages.at(entityInfo.m_dataComponentStorageIndex)
		.m_chunkArray.m_entities[entityInfo.m_chunkArrayIndex] = entity;
	m_sceneDataStorage.m_dataComponentStorages.at(newEntityInfo.m_dataComponentStorageIndex)
		.m_chunkArray.m_entities[newEntityInfo.m_chunkArrayIndex] = newEntity;
	DeleteEntity(newEntity);
#pragma endregion
	SetUnsaved();
}

void Scene::SetDataComponent(const unsigned& entityIndex, size_t id, size_t size, IDataComponent* data)
{
	auto& entityInfo = m_sceneDataStorage.m_entityMetadataList.at(entityIndex);
	auto& dataComponentStorage = m_sceneDataStorage.m_dataComponentStorages[entityInfo.m_dataComponentStorageIndex];
	const auto chunkIndex = entityInfo.m_chunkArrayIndex / dataComponentStorage.m_chunkCapacity;
	const auto chunkPointer = entityInfo.m_chunkArrayIndex % dataComponentStorage.m_chunkCapacity;
	const auto chunk = dataComponentStorage.m_chunkArray.m_chunks[chunkIndex];
	if (id == typeid(Transform).hash_code())
	{
		chunk.SetData(static_cast<size_t>(chunkPointer * sizeof(Transform)), sizeof(Transform), data);
		static_cast<TransformUpdateFlag*>(
			chunk.GetDataPointer(static_cast<size_t>(
				(sizeof(Transform) + sizeof(GlobalTransform)) * dataComponentStorage.m_chunkCapacity +
				chunkPointer * sizeof(TransformUpdateFlag))))
			->m_transformModified = true;
	}
	else if (id == typeid(GlobalTransform).hash_code())
	{
		chunk.SetData(
			static_cast<size_t>(
				sizeof(Transform) * dataComponentStorage.m_chunkCapacity + chunkPointer * sizeof(GlobalTransform)),
			sizeof(GlobalTransform),
			data);
		static_cast<TransformUpdateFlag*>(
			chunk.GetDataPointer(static_cast<size_t>(
				(sizeof(Transform) + sizeof(GlobalTransform)) * dataComponentStorage.m_chunkCapacity +
				chunkPointer * sizeof(TransformUpdateFlag))))
			->m_globalTransformModified = true;
	}
	else if (id == typeid(TransformUpdateFlag).hash_code())
	{
		chunk.SetData(
			static_cast<size_t>(
				(sizeof(Transform) + sizeof(GlobalTransform)) * dataComponentStorage.m_chunkCapacity +
				chunkPointer * sizeof(TransformUpdateFlag)),
			sizeof(TransformUpdateFlag),
			data);
	}
	else
	{
		for (const auto& type : dataComponentStorage.m_dataComponentTypes)
		{
			if (type.m_typeId == id)
			{
				chunk.SetData(
					static_cast<size_t>(
						type.m_offset * dataComponentStorage.m_chunkCapacity + chunkPointer * type.m_size),
					size,
					data);
				return;
			}
		}
		EVOENGINE_LOG("ComponentData doesn't exist");
	}
	SetUnsaved();
}
IDataComponent* Scene::GetDataComponentPointer(unsigned entityIndex, const size_t& id)
{
	EntityMetadata& entityInfo = m_sceneDataStorage.m_entityMetadataList.at(entityIndex);
	auto& dataComponentStorage = m_sceneDataStorage.m_dataComponentStorages[entityInfo.m_dataComponentStorageIndex];
	const auto chunkIndex = entityInfo.m_chunkArrayIndex / dataComponentStorage.m_chunkCapacity;
	const auto chunkPointer = entityInfo.m_chunkArrayIndex % dataComponentStorage.m_chunkCapacity;
	const auto chunk = dataComponentStorage.m_chunkArray.m_chunks[chunkIndex];
	if (id == typeid(Transform).hash_code())
	{
		return chunk.GetDataPointer(static_cast<size_t>(chunkPointer * sizeof(Transform)));
	}
	if (id == typeid(GlobalTransform).hash_code())
	{
		return chunk.GetDataPointer(static_cast<size_t>(
			sizeof(Transform) * dataComponentStorage.m_chunkCapacity + chunkPointer * sizeof(GlobalTransform)));
	}
	if (id == typeid(TransformUpdateFlag).hash_code())
	{
		return chunk.GetDataPointer(static_cast<size_t>(
			(sizeof(Transform) + sizeof(GlobalTransform)) * dataComponentStorage.m_chunkCapacity +
			chunkPointer * sizeof(TransformUpdateFlag)));
	}
	for (const auto& type : dataComponentStorage.m_dataComponentTypes)
	{
		if (type.m_typeId == id)
		{
			return chunk.GetDataPointer(
				static_cast<size_t>(type.m_offset * dataComponentStorage.m_chunkCapacity + chunkPointer * type.m_size));
		}
	}
	EVOENGINE_LOG("ComponentData doesn't exist");
	return nullptr;
}
IDataComponent* Scene::GetDataComponentPointer(const Entity& entity, const size_t& id)
{
	assert(IsEntityValid(entity));
	EntityMetadata& entityInfo = m_sceneDataStorage.m_entityMetadataList.at(entity.m_index);
	auto& dataComponentStorage = m_sceneDataStorage.m_dataComponentStorages[entityInfo.m_dataComponentStorageIndex];
	const auto chunkIndex = entityInfo.m_chunkArrayIndex / dataComponentStorage.m_chunkCapacity;
	const auto chunkPointer = entityInfo.m_chunkArrayIndex % dataComponentStorage.m_chunkCapacity;
	const auto chunk = dataComponentStorage.m_chunkArray.m_chunks[chunkIndex];
	if (id == typeid(Transform).hash_code())
	{
		return chunk.GetDataPointer(static_cast<size_t>(chunkPointer * sizeof(Transform)));
	}
	if (id == typeid(GlobalTransform).hash_code())
	{
		return chunk.GetDataPointer(static_cast<size_t>(
			sizeof(Transform) * dataComponentStorage.m_chunkCapacity + chunkPointer * sizeof(GlobalTransform)));
	}
	if (id == typeid(TransformUpdateFlag).hash_code())
	{
		return chunk.GetDataPointer(static_cast<size_t>(
			(sizeof(Transform) + sizeof(GlobalTransform)) * dataComponentStorage.m_chunkCapacity +
			chunkPointer * sizeof(TransformUpdateFlag)));
	}
	for (const auto& type : dataComponentStorage.m_dataComponentTypes)
	{
		if (type.m_typeId == id)
		{
			return chunk.GetDataPointer(
				static_cast<size_t>(type.m_offset * dataComponentStorage.m_chunkCapacity + chunkPointer * type.m_size));
		}
	}
	EVOENGINE_LOG("ComponentData doesn't exist");
	return nullptr;
}
Handle Scene::GetEntityHandle(const Entity& entity)
{
	return m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_handle;
}
void Scene::SetPrivateComponent(const Entity& entity, const std::shared_ptr<IPrivateComponent>& ptr)
{
	assert(ptr && IsEntityValid(entity));
	auto typeName = ptr->GetTypeName();
	auto& elements = m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_privateComponentElements;
	for (auto& element : elements)
	{
		if (typeName == element.m_privateComponentData->GetTypeName())
		{
			return;
		}
	}

	auto id = Serialization::GetSerializableTypeId(typeName);
	m_sceneDataStorage.m_entityPrivateComponentStorage.SetPrivateComponent(entity, id);
	elements.emplace_back(id, ptr, entity, std::dynamic_pointer_cast<Scene>(GetSelf()));
	SetUnsaved();
}

void Scene::ForEachDescendantHelper(const Entity& target, const std::function<void(const Entity& entity)>& func)
{
	func(target);
	ForEachChild(target, [&](Entity child) { ForEachDescendantHelper(child, func); });
}

Entity Scene::GetRoot(const Entity& entity)
{
	Entity retVal = entity;
	auto parent = GetParent(retVal);
	while (parent.GetIndex() != 0)
	{
		retVal = parent;
		parent = GetParent(retVal);
	}
	return retVal;
}

Entity Scene::GetEntity(const size_t& index)
{
	if (index > 0 && index < m_sceneDataStorage.m_entities.size())
		return m_sceneDataStorage.m_entities.at(index);
	return {};
}

void Scene::RemovePrivateComponent(const Entity& entity, size_t typeId)
{
	assert(IsEntityValid(entity));
	auto& privateComponentElements =
		m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_privateComponentElements;
	for (auto i = 0; i < privateComponentElements.size(); i++)
	{
		if (privateComponentElements[i].m_typeId == typeId)
		{
			m_sceneDataStorage.m_entityPrivateComponentStorage.RemovePrivateComponent(
				entity, typeId, privateComponentElements[i].m_privateComponentData);
			privateComponentElements.erase(privateComponentElements.begin() + i);
			SetUnsaved();
			break;
		}
	}
}

void Scene::SetEnable(const Entity& entity, const bool& value)
{
	assert(IsEntityValid(entity));
	if (m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_enabled != value)
	{
		for (auto& i : m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_privateComponentElements)
		{
			if (value)
			{
				i.m_privateComponentData->OnEntityEnable();
			}
			else
			{
				i.m_privateComponentData->OnEntityDisable();
			}
		}
	}
	m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_enabled = value;

	for (const auto& i : m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_children)
	{
		SetEnable(i, value);
	}
	SetUnsaved();
}

void Scene::SetEnableSingle(const Entity& entity, const bool& value)
{
	assert(IsEntityValid(entity));
	auto& entityMetadata = m_sceneDataStorage.m_entityMetadataList.at(entity.m_index);
	if (entityMetadata.m_enabled != value)
	{
		for (auto& i : entityMetadata.m_privateComponentElements)
		{
			if (value)
			{
				i.m_privateComponentData->OnEntityEnable();
			}
			else
			{
				i.m_privateComponentData->OnEntityDisable();
			}
		}
		entityMetadata.m_enabled = value;
	}
}
EntityMetadata& Scene::GetEntityMetadata(const Entity& entity)
{
	assert(IsEntityValid(entity));
	return m_sceneDataStorage.m_entityMetadataList.at(entity.m_index);
}
void Scene::ForAllEntities(const std::function<void(int i, Entity entity)>& func) const
{
	for (int index = 0; index < m_sceneDataStorage.m_entities.size(); index++)
	{
		if (m_sceneDataStorage.m_entities.at(index).m_version != 0)
		{
			func(index, m_sceneDataStorage.m_entities.at(index));
		}
	}
}

Bound Scene::GetEntityBoundingBox(const Entity& entity)
{
	auto descendants = GetDescendants(entity);
	descendants.emplace_back(entity);

	Bound retVal{};

	for (const auto& walker : descendants)
	{
		auto gt = GetDataComponent<GlobalTransform>(walker);
		if (HasPrivateComponent<MeshRenderer>(walker))
		{
			auto meshRenderer = GetOrSetPrivateComponent<MeshRenderer>(walker).lock();
			if (const auto mesh = meshRenderer->m_mesh.Get<Mesh>())
			{
				auto meshBound = mesh->GetBound();
				meshBound.ApplyTransform(gt.m_value);
				glm::vec3 center = meshBound.Center();

				glm::vec3 size = meshBound.Size();
				retVal.m_min = glm::vec3(
					(glm::min)(retVal.m_min.x, center.x - size.x),
					(glm::min)(retVal.m_min.y, center.y - size.y),
					(glm::min)(retVal.m_min.z, center.z - size.z));
				retVal.m_max = glm::vec3(
					(glm::max)(retVal.m_max.x, center.x + size.x),
					(glm::max)(retVal.m_max.y, center.y + size.y),
					(glm::max)(retVal.m_max.z, center.z + size.z));
			}
		}
		else if (HasPrivateComponent<SkinnedMeshRenderer>(walker))
		{
			auto meshRenderer = GetOrSetPrivateComponent<SkinnedMeshRenderer>(walker).lock();
			if (const auto mesh = meshRenderer->m_skinnedMesh.Get<SkinnedMesh>())
			{
				auto meshBound = mesh->GetBound();
				meshBound.ApplyTransform(gt.m_value);
				glm::vec3 center = meshBound.Center();

				glm::vec3 size = meshBound.Size();
				retVal.m_min = glm::vec3(
					(glm::min)(retVal.m_min.x, center.x - size.x),
					(glm::min)(retVal.m_min.y, center.y - size.y),
					(glm::min)(retVal.m_min.z, center.z - size.z));
				retVal.m_max = glm::vec3(
					(glm::max)(retVal.m_max.x, center.x + size.x),
					(glm::max)(retVal.m_max.y, center.y + size.y),
					(glm::max)(retVal.m_max.z, center.z + size.z));
			}
		}
	}

	return retVal;
}

void Scene::GetEntityArray(const EntityQuery& entityQuery, std::vector<Entity>& container, bool checkEnable)
{
	assert(entityQuery.IsValid());
	auto queriedStorages = QueryDataComponentStorages(entityQuery);
	for (const auto i : queriedStorages)
	{
		GetEntityStorage(i.get(), container, checkEnable);
	}
}

size_t Scene::GetEntityAmount(EntityQuery entityQuery, bool checkEnable)
{
	assert(entityQuery.IsValid());
	size_t retVal = 0;
	if (checkEnable)
	{
		auto queriedStorages = QueryDataComponentStorages(entityQuery);
		for (const auto i : queriedStorages)
		{
			for (int index = 0; index < i.get().m_entityAliveCount; index++)
			{
				if (IsEntityEnabled(i.get().m_chunkArray.m_entities[index]))
					retVal++;
			}
		}
	}
	else
	{
		auto queriedStorages = QueryDataComponentStorages(entityQuery);
		for (const auto i : queriedStorages)
		{
			retVal += i.get().m_entityAliveCount;
		}
	}
	return retVal;
}

std::vector<Entity> Scene::GetDescendants(const Entity& entity)
{
	std::vector<Entity> retVal;
	if(!IsEntityValid(entity)) return retVal;
	GetDescendantsHelper(entity, retVal);
	return retVal;
}
void Scene::GetDescendantsHelper(const Entity& target, std::vector<Entity>& results)
{
	auto& children = m_sceneDataStorage.m_entityMetadataList.at(target.m_index).m_children;
	if (!children.empty())
		results.insert(results.end(), children.begin(), children.end());
	for (const auto& i : children)
		GetDescendantsHelper(i, results);
}
template <typename T> std::vector<Entity> Scene::GetPrivateComponentOwnersList(const std::shared_ptr<Scene>& scene)
{
	return m_sceneDataStorage.m_entityPrivateComponentStorage.GetOwnersList<T>();
}

std::weak_ptr<IPrivateComponent> Scene::GetPrivateComponent(const Entity& entity, const std::string& typeName)
{
	size_t i = 0;
	auto& elements = m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_privateComponentElements;
	for (auto& element : elements)
	{
		if (typeName == element.m_privateComponentData->GetTypeName())
		{
			return element.m_privateComponentData;
		}
		i++;
	}
	throw 0;
}

Entity Scene::GetEntity(const Handle& handle)
{
	auto search = m_sceneDataStorage.m_entityMap.find(handle);
	if (search != m_sceneDataStorage.m_entityMap.end())
	{
		return search->second;
	}
	return {};
}
bool Scene::HasPrivateComponent(const Entity& entity, const std::string& typeName) const
{
	assert(IsEntityValid(entity));
	for (auto& element : m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_privateComponentElements)
	{
		if (element.m_privateComponentData->m_typeName == typeName)
		{
			return true;
		}
	}
	return false;
}
std::vector<std::reference_wrapper<DataComponentStorage>> Scene::QueryDataComponentStorages(
	unsigned int entityQueryIndex)
{
	const auto& queryInfos = Entities::GetInstance().m_entityQueryInfos.at(entityQueryIndex);
	auto& entityComponentStorage = m_sceneDataStorage.m_dataComponentStorages;
	std::vector<std::reference_wrapper<DataComponentStorage>> queriedStorage;
	// Select storage with every contained.
	if (!queryInfos.m_allDataComponentTypes.empty())
	{
		for (int i = 0; i < entityComponentStorage.size(); i++)
		{
			auto& dataStorage = entityComponentStorage.at(i);
			bool check = true;
			for (const auto& type : queryInfos.m_allDataComponentTypes)
			{
				if (!dataStorage.HasType(type.m_typeId))
					check = false;
			}
			if (check)
				queriedStorage.push_back(std::ref(dataStorage));
		}
	}
	else
	{
		for (int i = 0; i < entityComponentStorage.size(); i++)
		{
			auto& dataStorage = entityComponentStorage.at(i);
			queriedStorage.push_back(std::ref(dataStorage));
		}
	}
	// Erase with any
	if (!queryInfos.m_anyDataComponentTypes.empty())
	{
		for (int i = 0; i < queriedStorage.size(); i++)
		{
			bool contain = false;
			for (const auto& type : queryInfos.m_anyDataComponentTypes)
			{
				if (queriedStorage.at(i).get().HasType(type.m_typeId))
					contain = true;
				if (contain)
					break;
			}
			if (!contain)
			{
				queriedStorage.erase(queriedStorage.begin() + i);
				i--;
			}
		}
	}
	// Erase with none
	if (!queryInfos.m_noneDataComponentTypes.empty())
	{
		for (int i = 0; i < queriedStorage.size(); i++)
		{
			bool contain = false;
			for (const auto& type : queryInfos.m_noneDataComponentTypes)
			{
				if (queriedStorage.at(i).get().HasType(type.m_typeId))
					contain = true;
				if (contain)
					break;
			}
			if (contain)
			{
				queriedStorage.erase(queriedStorage.begin() + i);
				i--;
			}
		}
	}
	return queriedStorage;
}
bool Scene::IsEntityValid(const Entity& entity) const
{
	auto& storage = m_sceneDataStorage.m_entities;
	return entity.m_index != 0 && entity.m_version != 0 && entity.m_index < storage.size() &&
		storage.at(entity.m_index).m_version == entity.m_version;
}
bool Scene::IsEntityEnabled(const Entity& entity) const
{
	assert(IsEntityValid(entity));
	return m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_enabled;
}
bool Scene::IsEntityRoot(const Entity& entity) const
{
	assert(IsEntityValid(entity));
	return m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_root == entity;
}
bool Scene::IsEntityStatic(const Entity& entity)
{
	assert(IsEntityValid(entity));
	return m_sceneDataStorage.m_entityMetadataList.at(GetRoot(entity).m_index).m_static;
}

bool Scene::IsEntityAncestorSelected(const Entity& entity) const
{
	assert(IsEntityValid(entity));
	return m_sceneDataStorage.m_entityMetadataList.at(entity.m_index).m_ancestorSelected;
}

#pragma endregion
