#include "Scene.hpp"
#include "Application.hpp"

#include "ClassRegistry.hpp"
#include "Entities.hpp"
#include "EntityMetadata.hpp"
#include "Jobs.hpp"
#include "Lights.hpp"
#include "MeshRenderer.hpp"
#include "SkinnedMeshRenderer.hpp"
#include "UnknownPrivateComponent.hpp"
using namespace evo_engine;

void Scene::Purge() {
  pressed_keys_.clear();
  main_camera.Clear();

  scene_data_storage_.entity_private_component_storage = PrivateComponentStorage();
  scene_data_storage_.entity_private_component_storage.owner_scene = std::dynamic_pointer_cast<Scene>(GetSelf());
  scene_data_storage_.entities.clear();
  scene_data_storage_.entity_metadata_list.clear();
  for (int index = 1; index < scene_data_storage_.data_component_storage_list.size(); index++) {
    auto& i = scene_data_storage_.data_component_storage_list[index];
    for (auto& chunk : i.chunk_array.chunk_array)
      free(chunk.chunk_data);
  }
  scene_data_storage_.data_component_storage_list.clear();

  scene_data_storage_.data_component_storage_list.emplace_back();
  scene_data_storage_.entities.emplace_back();
  scene_data_storage_.entity_metadata_list.emplace_back();
}

Bound Scene::GetBound() const {
  return world_bound_;
}

void Scene::SetBound(const Bound& value) {
  world_bound_ = value;
}

Scene::~Scene() {
  Purge();
  for (const auto& i : systems_) {
    i.second->OnDestroy();
  }
}

void Scene::Start() const {
  const auto entities = scene_data_storage_.entities;
  for (const auto& entity : entities) {
    if (entity.version_ == 0)
      continue;
    const auto entityInfo = scene_data_storage_.entity_metadata_list[entity.index_];
    if (!entityInfo.entity_enabled)
      continue;
    for (const auto& privateComponentElement : entityInfo.private_component_elements) {
      if (!privateComponentElement.private_component_data->enabled_)
        continue;
      if (!privateComponentElement.private_component_data->started_) {
        privateComponentElement.private_component_data->Start();
        if (entity.version_ != entityInfo.entity_version)
          break;
        privateComponentElement.private_component_data->started_ = true;
      }
      if (entity.version_ != entityInfo.entity_version)
        break;
    }
  }
  for (auto& i : systems_) {
    if (i.second->Enabled()) {
      if (!i.second->started_) {
        i.second->Start();
        i.second->started_ = true;
      }
    }
  }
}

void Scene::Update() const {
  const auto entities = scene_data_storage_.entities;
  for (const auto& entity : entities) {
    if (entity.version_ == 0)
      continue;
    const auto entityInfo = scene_data_storage_.entity_metadata_list[entity.index_];
    if (!entityInfo.entity_enabled)
      continue;
    for (const auto& privateComponentElement : entityInfo.private_component_elements) {
      if (!privateComponentElement.private_component_data->enabled_ ||
          !privateComponentElement.private_component_data->started_)
        continue;
      privateComponentElement.private_component_data->Update();
      if (entity.version_ != entityInfo.entity_version)
        break;
    }
  }

  for (auto& i : systems_) {
    if (i.second->Enabled() && i.second->started_) {
      i.second->Update();
    }
  }
}

void Scene::LateUpdate() const {
  const auto entities = scene_data_storage_.entities;
  for (const auto& entity : entities) {
    if (entity.version_ == 0)
      continue;
    const auto entityInfo = scene_data_storage_.entity_metadata_list[entity.index_];
    if (!entityInfo.entity_enabled)
      continue;
    for (const auto& privateComponentElement : entityInfo.private_component_elements) {
      if (!privateComponentElement.private_component_data->enabled_ ||
          !privateComponentElement.private_component_data->started_)
        continue;
      privateComponentElement.private_component_data->LateUpdate();
      if (entity.version_ != entityInfo.entity_version)
        break;
    }
  }

  for (auto& i : systems_) {
    if (i.second->Enabled() && i.second->started_) {
      i.second->LateUpdate();
    }
  }
}
void Scene::FixedUpdate() const {
  const auto entities = scene_data_storage_.entities;
  for (const auto& entity : entities) {
    if (entity.version_ == 0)
      continue;
    const auto entityInfo = scene_data_storage_.entity_metadata_list[entity.index_];
    if (!entityInfo.entity_enabled)
      continue;
    for (const auto& privateComponentElement : entityInfo.private_component_elements) {
      if (!privateComponentElement.private_component_data->enabled_ ||
          !privateComponentElement.private_component_data->started_)
        continue;
      privateComponentElement.private_component_data->FixedUpdate();
      if (entity.version_ != entityInfo.entity_version)
        break;
    }
  }

  for (const auto& i : systems_) {
    if (i.second->Enabled() && i.second->started_) {
      i.second->FixedUpdate();
    }
  }
}
static const char* EnvironmentTypes[]{"Environmental Map", "Color"};
bool Scene::OnInspect(const std::shared_ptr<EditorLayer>& editor_layer) {
  bool modified = false;
  if (this == Application::GetActiveScene().get())
    if (editor_layer->DragAndDropButton<Camera>(main_camera, "Main Camera", true))
      modified = true;
  if (ImGui::TreeNodeEx("Environment Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
    static int type = static_cast<int>(environment.environment_type);
    if (ImGui::Combo("Environment type", &type, EnvironmentTypes, IM_ARRAYSIZE(EnvironmentTypes))) {
      environment.environment_type = static_cast<EnvironmentType>(type);
      modified = true;
    }
    switch (environment.environment_type) {
      case EnvironmentType::EnvironmentalMap: {
        if (editor_layer->DragAndDropButton<EnvironmentalMap>(environment.environmental_map, "Environmental Map"))
          modified = true;
      } break;
      case EnvironmentType::Color: {
        if (ImGui::ColorEdit3("Background Color", &environment.background_color.x))
          modified = true;
      } break;
    }
    if (ImGui::DragFloat("Environmental light intensity", &environment.ambient_light_intensity, 0.01f, 0.0f, 10.0f))
      modified = true;
    if (ImGui::DragFloat("Environmental light gamma", &environment.environment_gamma, 0.01f, 0.0f, 10.0f)) {
      modified = true;
    }
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Systems")) {
    if (ImGui::BeginPopupContextWindow("SystemInspectorPopup")) {
      ImGui::Text("Add system: ");
      ImGui::Separator();
      static float rank = 0.0f;
      ImGui::DragFloat("Rank", &rank, 1.0f, 0.0f, 999.0f);
      for (auto& i : editor_layer->system_menu_list_) {
        i.second(rank);
      }
      ImGui::Separator();
      ImGui::EndPopup();
    }
    for (auto& i : Application::GetActiveScene()->systems_) {
      if (ImGui::CollapsingHeader(i.second->GetTypeName().c_str())) {
        bool enabled = i.second->Enabled();
        if (ImGui::Checkbox("Enabled", &enabled)) {
          if (i.second->Enabled() != enabled) {
            if (enabled) {
              i.second->Enable();
              modified = true;
            } else {
              i.second->Disable();
              modified = true;
            }
          }
        }
        i.second->OnInspect(editor_layer);
      }
    }
    ImGui::TreePop();
  }
  return modified;
}

std::shared_ptr<ISystem> Scene::GetOrCreateSystem(const std::string& system_name, float order) {
  size_t typeId;
  auto ptr = Serialization::ProduceSerializable(system_name, typeId);
  auto system = std::dynamic_pointer_cast<ISystem>(ptr);
  system->handle_ = Handle();
  system->rank_ = order;
  systems_.insert({order, system});
  indexed_systems_[typeId] = system;
  mapped_systems_[system->handle_] = system;
  system->started_ = false;
  system->OnCreate();
  SetUnsaved();
  return std::dynamic_pointer_cast<ISystem>(ptr);
}

void Scene::Serialize(YAML::Emitter& out) const {
  out << YAML::Key << "environment" << YAML::Value << YAML::BeginMap;
  environment.Serialize(out);
  out << YAML::EndMap;
  main_camera.Save("main_camera", out);
  std::unordered_map<Handle, std::shared_ptr<IAsset>> assetMap;
  std::vector<AssetRef> list;
  list.push_back(environment.environmental_map);
  auto& sceneDataStorage = scene_data_storage_;
#pragma region EntityInfo
  out << YAML::Key << "entity_metadata_list" << YAML::Value << YAML::BeginSeq;
  for (int i = 1; i < sceneDataStorage.entity_metadata_list.size(); i++) {
    auto& entityMetadata = sceneDataStorage.entity_metadata_list[i];
    if (entityMetadata.entity_handle == 0)
      continue;
    for (const auto& element : entityMetadata.private_component_elements) {
      element.private_component_data->CollectAssetRef(list);
    }
    entityMetadata.Serialize(out, std::dynamic_pointer_cast<Scene>(GetSelf()));
  }
  out << YAML::EndSeq;
#pragma endregion

#pragma region Systems
  out << YAML::Key << "systems_" << YAML::Value << YAML::BeginSeq;
  for (const auto& i : systems_) {
    SerializeSystem(i.second, out);
    i.second->CollectAssetRef(list);
  }
  out << YAML::EndSeq;
#pragma endregion

#pragma region Assets
  for (auto& i : list) {
    const auto asset = i.Get<IAsset>();

    if (asset && !Resources::IsResource(asset->GetHandle())) {
      if (asset->IsTemporary()) {
        assetMap[asset->GetHandle()] = asset;
      } else if (!asset->Saved()) {
        asset->Save();
      }
    }
  }
  bool listCheck = true;
  while (listCheck) {
    size_t currentSize = assetMap.size();
    list.clear();
    for (auto& i : assetMap) {
      i.second->CollectAssetRef(list);
    }
    for (auto& i : list) {
      auto asset = i.Get<IAsset>();
      if (asset && !Resources::IsResource(asset->GetHandle())) {
        if (asset->IsTemporary()) {
          assetMap[asset->GetHandle()] = asset;
        } else if (!asset->Saved()) {
          asset->Save();
        }
      }
    }
    if (assetMap.size() == currentSize)
      listCheck = false;
  }
  if (!assetMap.empty()) {
    out << YAML::Key << "LocalAssets" << YAML::Value << YAML::BeginSeq;
    for (auto& i : assetMap) {
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
  out << YAML::Key << "data_component_storage_list" << YAML::Value << YAML::BeginSeq;
  for (int i = 1; i < sceneDataStorage.data_component_storage_list.size(); i++) {
    SerializeDataComponentStorage(sceneDataStorage.data_component_storage_list[i], out);
  }
  out << YAML::EndSeq;
#pragma endregion
  out << YAML::EndMap;
}
void Scene::Deserialize(const YAML::Node& in) {
  Purge();
  auto scene = std::dynamic_pointer_cast<Scene>(GetSelf());
  scene_data_storage_.entities.clear();
  scene_data_storage_.entity_metadata_list.clear();
  scene_data_storage_.data_component_storage_list.clear();
  scene_data_storage_.entities.emplace_back();
  scene_data_storage_.entity_metadata_list.emplace_back();
  scene_data_storage_.data_component_storage_list.emplace_back();

#pragma region EntityMetadata
  auto inEntityMetadataList = in["entity_metadata_list"];
  int currentIndex = 1;
  for (const auto& inEntityMetadata : inEntityMetadataList) {
    scene_data_storage_.entity_metadata_list.emplace_back();
    auto& newInfo = scene_data_storage_.entity_metadata_list.back();
    newInfo.Deserialize(inEntityMetadata, scene);
    Entity entity;
    entity.version_ = 1;
    entity.index_ = currentIndex;
    scene_data_storage_.entity_map[newInfo.entity_handle] = entity;
    scene_data_storage_.entities.push_back(entity);
    currentIndex++;
  }
  currentIndex = 1;
  for (const auto& inEntityMetadata : inEntityMetadataList) {
    auto& metadata = scene_data_storage_.entity_metadata_list[currentIndex];
    if (inEntityMetadata["Parent.Handle"]) {
      metadata.parent = scene_data_storage_.entity_map[Handle(inEntityMetadata["Parent.Handle"].as<uint64_t>())];
      auto& parentMetadata = scene_data_storage_.entity_metadata_list[metadata.parent.index_];
      Entity entity;
      entity.version_ = 1;
      entity.index_ = currentIndex;
      parentMetadata.children.push_back(entity);
    }
    if (inEntityMetadata["Root.Handle"])
      metadata.root = scene_data_storage_.entity_map[Handle(inEntityMetadata["Root.Handle"].as<uint64_t>())];
    currentIndex++;
  }
#pragma endregion

#pragma region DataComponentStorage
  auto inDataComponentStorages = in["data_component_storage_list"];
  int storageIndex = 1;
  for (const auto& inDataComponentStorage : inDataComponentStorages) {
    scene_data_storage_.data_component_storage_list.emplace_back();
    auto& dataComponentStorage = scene_data_storage_.data_component_storage_list.back();
    dataComponentStorage.entity_size = inDataComponentStorage["entity_size"].as<size_t>();
    dataComponentStorage.chunk_capacity = inDataComponentStorage["chunk_capacity"].as<size_t>();
    dataComponentStorage.entity_alive_count = dataComponentStorage.entity_count =
        inDataComponentStorage["entity_alive_count"].as<size_t>();
    dataComponentStorage.chunk_array.entity_array.resize(dataComponentStorage.entity_alive_count);
    const size_t chunkSize = dataComponentStorage.entity_count / dataComponentStorage.chunk_capacity + 1;
    while (dataComponentStorage.chunk_array.chunk_array.size() <= chunkSize) {
      // Allocate new chunk;
      ComponentDataChunk chunk = {};
      chunk.chunk_data = static_cast<void*>(calloc(1, Entities::GetArchetypeChunkSize()));
      dataComponentStorage.chunk_array.chunk_array.push_back(chunk);
    }
    auto inDataComponentTypes = inDataComponentStorage["data_component_types"];
    for (const auto& inDataComponentType : inDataComponentTypes) {
      DataComponentType dataComponentType;
      dataComponentType.type_name = inDataComponentType["type_name"].as<std::string>();
      dataComponentType.type_size = inDataComponentType["m_size"].as<size_t>();
      dataComponentType.type_offset = inDataComponentType["m_offset"].as<size_t>();
      dataComponentType.type_index = Serialization::GetDataComponentTypeId(dataComponentType.type_name);
      dataComponentStorage.data_component_types.push_back(dataComponentType);
    }
    auto inDataChunkArray = inDataComponentStorage["chunk_array"];
    int chunkArrayIndex = 0;
    for (const auto& entityDataComponent : inDataChunkArray) {
      Handle handle = entityDataComponent["m_handle"].as<uint64_t>();
      Entity entity = scene_data_storage_.entity_map[handle];
      dataComponentStorage.chunk_array.entity_array[chunkArrayIndex] = entity;
      auto& metadata = scene_data_storage_.entity_metadata_list[entity.index_];
      metadata.data_component_storage_index = storageIndex;
      metadata.chunk_array_index = chunkArrayIndex;
      const auto chunkIndex = metadata.chunk_array_index / dataComponentStorage.chunk_capacity;
      const auto chunkPointer = metadata.chunk_array_index % dataComponentStorage.chunk_capacity;
      const auto chunk = dataComponentStorage.chunk_array.chunk_array[chunkIndex];

      int typeIndex = 0;
      for (const auto& inDataComponent : entityDataComponent["DataComponents"]) {
        auto& type = dataComponentStorage.data_component_types[typeIndex];
        auto data = inDataComponent["Data"].as<YAML::Binary>();
        std::memcpy(chunk.GetDataPointer(static_cast<size_t>(type.type_offset * dataComponentStorage.chunk_capacity +
                                                             chunkPointer * type.type_size)),
                    data.data(), data.size());
        typeIndex++;
      }

      chunkArrayIndex++;
    }
    storageIndex++;
  }
  auto self = std::dynamic_pointer_cast<Scene>(GetSelf());
#pragma endregion
  main_camera.Load("main_camera", in, self);
#pragma region Assets
  std::vector<std::pair<int, std::shared_ptr<IAsset>>> localAssets;
  if (const auto inLocalAssets = in["LocalAssets"]) {
    int index = 0;
    for (const auto& i : inLocalAssets) {
      // First, find the asset in assetregistry
      if (const auto typeName = i["m_typeName"].as<std::string>(); Serialization::HasSerializableType(typeName)) {
        auto asset = ProjectManager::CreateTemporaryAsset(typeName, i["m_handle"].as<uint64_t>());
        localAssets.emplace_back(index, asset);
      }
      index++;
    }

    for (const auto& i : localAssets) {
      i.second->Deserialize(inLocalAssets[i.first]);
    }
  }

#pragma endregion
  if (in["environment"])
    environment.Deserialize(in["environment"]);
  int entityIndex = 1;
  for (const auto& inEntityInfo : inEntityMetadataList) {
    auto& entityMetadata = scene_data_storage_.entity_metadata_list.at(entityIndex);
    auto entity = scene_data_storage_.entities[entityIndex];
    if (auto inPrivateComponents = inEntityInfo["private_component_elements"]) {
      for (const auto& inPrivateComponent : inPrivateComponents) {
        const auto name = inPrivateComponent["m_typeName"].as<std::string>();
        size_t hashCode;
        if (Serialization::HasSerializableType(name)) {
          auto ptr = std::static_pointer_cast<IPrivateComponent>(Serialization::ProduceSerializable(name, hashCode));
          ptr->enabled_ = inPrivateComponent["enabled_"].as<bool>();
          ptr->started_ = false;
          scene_data_storage_.entity_private_component_storage.SetPrivateComponent(entity, hashCode);
          entityMetadata.private_component_elements.emplace_back(hashCode, ptr, entity, self);
        } else {
          auto ptr = std::static_pointer_cast<IPrivateComponent>(
              Serialization::ProduceSerializable("UnknownPrivateComponent", hashCode));
          ptr->enabled_ = false;
          ptr->started_ = false;
          std::dynamic_pointer_cast<UnknownPrivateComponent>(ptr)->m_originalTypeName = name;
          scene_data_storage_.entity_private_component_storage.SetPrivateComponent(entity, hashCode);
          entityMetadata.private_component_elements.emplace_back(hashCode, ptr, entity, self);
        }
      }
    }
    entityIndex++;
  }

#pragma region Systems
  if (auto inSystems = in["systems_"]) {
    std::vector<std::pair<int, std::shared_ptr<ISystem>>> systems;
    int index = 0;
    for (const auto& inSystem : inSystems) {
      if (const auto typeName = inSystem["m_typeName"].as<std::string>();
          Serialization::HasSerializableType(typeName)) {
        size_t hashCode;
        if (const auto ptr =
                std::static_pointer_cast<ISystem>(Serialization::ProduceSerializable(typeName, hashCode))) {
          ptr->handle_ = Handle(inSystem["m_handle"].as<uint64_t>());
          ptr->enabled_ = inSystem["enabled_"].as<bool>();
          ptr->rank_ = inSystem["m_rank"].as<float>();
          ptr->started_ = false;
          systems_.insert({ptr->rank_, ptr});
          indexed_systems_.insert({hashCode, ptr});
          mapped_systems_[ptr->handle_] = ptr;
          systems.emplace_back(index, ptr);
          ptr->scene_ = self;
          ptr->OnCreate();
        }
      }
      index++;
    }
#pragma endregion

    entityIndex = 1;
    for (const auto& inEntityInfo : inEntityMetadataList) {
      auto& entityInfo = scene_data_storage_.entity_metadata_list.at(entityIndex);
      auto inPrivateComponents = inEntityInfo["private_component_elements"];
      int componentIndex = 0;
      if (inPrivateComponents) {
        for (const auto& inPrivateComponent : inPrivateComponents) {
          auto name = inPrivateComponent["m_typeName"].as<std::string>();
          auto ptr = entityInfo.private_component_elements[componentIndex].private_component_data;
          ptr->Deserialize(inPrivateComponent);
          componentIndex++;
        }
      }
      entityIndex++;
    }

    for (const auto& i : systems) {
      i.second->Deserialize(inSystems[i.first]);
    }
  }
}
void Scene::SerializeDataComponentStorage(const DataComponentStorage& storage, YAML::Emitter& out) const {
  out << YAML::BeginMap;
  {
    out << YAML::Key << "entity_size" << YAML::Value << storage.entity_size;
    out << YAML::Key << "chunk_capacity" << YAML::Value << storage.chunk_capacity;
    out << YAML::Key << "entity_alive_count" << YAML::Value << storage.entity_alive_count;
    out << YAML::Key << "data_component_types" << YAML::Value << YAML::BeginSeq;
    for (const auto& i : storage.data_component_types) {
      out << YAML::BeginMap;
      out << YAML::Key << "type_name" << YAML::Value << i.type_name;
      out << YAML::Key << "m_size" << YAML::Value << i.type_size;
      out << YAML::Key << "m_offset" << YAML::Value << i.type_offset;
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;

    out << YAML::Key << "chunk_array" << YAML::Value << YAML::BeginSeq;
    for (int i = 0; i < storage.entity_alive_count; i++) {
      auto entity = storage.chunk_array.entity_array[i];
      if (entity.version_ == 0)
        continue;

      out << YAML::BeginMap;
      auto& entityInfo = scene_data_storage_.entity_metadata_list.at(entity.index_);
      out << YAML::Key << "m_handle" << YAML::Value << entityInfo.entity_handle;

      auto& dataComponentStorage =
          scene_data_storage_.data_component_storage_list[entityInfo.data_component_storage_index];
      const auto chunkIndex = entityInfo.chunk_array_index / dataComponentStorage.chunk_capacity;
      const auto chunkPointer = entityInfo.chunk_array_index % dataComponentStorage.chunk_capacity;
      const auto chunk = dataComponentStorage.chunk_array.chunk_array[chunkIndex];

      out << YAML::Key << "DataComponents" << YAML::Value << YAML::BeginSeq;
      for (const auto& type : dataComponentStorage.data_component_types) {
        out << YAML::BeginMap;
        out << YAML::Key << "Data" << YAML::Value
            << YAML::Binary((const unsigned char*)chunk.GetDataPointer(
                                type.type_offset * dataComponentStorage.chunk_capacity + chunkPointer * type.type_size),
                            type.type_size);
        out << YAML::EndMap;
      }
      out << YAML::EndSeq;

      out << YAML::EndMap;
    }
    out << YAML::EndSeq;
  }
  out << YAML::EndMap;
}
void Scene::SerializeSystem(const std::shared_ptr<ISystem>& system, YAML::Emitter& out) {
  out << YAML::BeginMap;
  {
    out << YAML::Key << "m_typeName" << YAML::Value << system->GetTypeName();
    out << YAML::Key << "enabled_" << YAML::Value << system->enabled_;
    out << YAML::Key << "m_rank" << YAML::Value << system->rank_;
    out << YAML::Key << "m_handle" << YAML::Value << system->GetHandle();
    system->Serialize(out);
  }
  out << YAML::EndMap;
}

void Scene::OnCreate() {
  scene_data_storage_.entities.emplace_back();
  scene_data_storage_.entity_metadata_list.emplace_back();
  scene_data_storage_.data_component_storage_list.emplace_back();
  scene_data_storage_.entity_private_component_storage.owner_scene = std::dynamic_pointer_cast<Scene>(GetSelf());

#pragma region Main Camera
  const auto mainCameraEntity = CreateEntity("Main Camera");
  Transform ltw;
  ltw.SetPosition(glm::vec3(0.0f, 5.0f, 10.0f));
  ltw.SetScale(glm::vec3(1, 1, 1));
  ltw.SetEulerRotation(glm::radians(glm::vec3(0, 0, 0)));
  SetDataComponent(mainCameraEntity, ltw);
  auto mainCameraComponent = GetOrSetPrivateComponent<Camera>(mainCameraEntity).lock();
  main_camera = mainCameraComponent;
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

bool Scene::LoadInternal(const std::filesystem::path& path) {
  auto previousScene = Application::GetActiveScene();
  Application::Attach(std::shared_ptr<Scene>(this, [](Scene*) {
  }));
  std::ifstream stream(path.string());
  std::stringstream stringStream;
  stringStream << stream.rdbuf();
  YAML::Node in = YAML::Load(stringStream.str());
  Deserialize(in);
  Application::Attach(previousScene);
  return true;
}
void Scene::Clone(const std::shared_ptr<Scene>& source, const std::shared_ptr<Scene>& new_scene) {
  new_scene->environment = source->environment;
  new_scene->saved_ = source->saved_;
  new_scene->world_bound_ = source->world_bound_;
  std::unordered_map<Handle, Handle> entityMap;

  new_scene->scene_data_storage_.Clone(entityMap, source->scene_data_storage_, new_scene);
  for (const auto& i : source->systems_) {
    auto systemName = i.second->GetTypeName();
    size_t hashCode;
    auto system = std::dynamic_pointer_cast<ISystem>(
        Serialization::ProduceSerializable(systemName, hashCode, i.second->GetHandle()));
    new_scene->systems_.insert({i.first, system});
    new_scene->indexed_systems_[hashCode] = system;
    new_scene->mapped_systems_[i.second->GetHandle()] = system;
    system->scene_ = new_scene;
    system->OnCreate();
    Serialization::CloneSystem(system, i.second);
    system->scene_ = new_scene;
  }
  new_scene->main_camera.entity_handle_ = source->main_camera.entity_handle_;
  new_scene->main_camera.private_component_type_name_ = source->main_camera.private_component_type_name_;
  new_scene->main_camera.Relink(entityMap, new_scene);
}

std::shared_ptr<LightProbe> Environment::GetLightProbe(const glm::vec3& position) {
  if (const auto environmentalMap = environmental_map.Get<EnvironmentalMap>()) {
    if (auto lightProbe = environmentalMap->m_lightProbe.Get<LightProbe>())
      return lightProbe;
  }
  return nullptr;
}

std::shared_ptr<ReflectionProbe> Environment::GetReflectionProbe(const glm::vec3& position) {
  if (const auto environmentalMap = environmental_map.Get<EnvironmentalMap>()) {
    if (auto reflectionProbe = environmentalMap->m_lightProbe.Get<ReflectionProbe>())
      return reflectionProbe;
  }
  return nullptr;
}

void Environment::Serialize(YAML::Emitter& out) const {
  out << YAML::Key << "background_color" << YAML::Value << background_color;
  out << YAML::Key << "environment_gamma" << YAML::Value << environment_gamma;
  out << YAML::Key << "ambient_light_intensity" << YAML::Value << ambient_light_intensity;
  out << YAML::Key << "environment_type" << YAML::Value << static_cast<unsigned>(environment_type);
  environmental_map.Save("environment", out);
}
void Environment::Deserialize(const YAML::Node& in) {
  if (in["background_color"])
    background_color = in["background_color"].as<glm::vec3>();
  if (in["environment_gamma"])
    environment_gamma = in["environment_gamma"].as<float>();
  if (in["ambient_light_intensity"])
    ambient_light_intensity = in["ambient_light_intensity"].as<float>();
  if (in["environment_type"])
    environment_type = (EnvironmentType)in["environment_type"].as<unsigned>();
  environmental_map.Load("environment", in);
}
void SceneDataStorage::Clone(std::unordered_map<Handle, Handle>& entity_links, const SceneDataStorage& source,
                             const std::shared_ptr<Scene>& new_scene) {
  entities = source.entities;
  entity_metadata_list.resize(source.entity_metadata_list.size());

  for (const auto& i : source.entity_metadata_list) {
    entity_links.insert({i.entity_handle, i.entity_handle});
  }
  data_component_storage_list.resize(source.data_component_storage_list.size());
  for (int i = 0; i < data_component_storage_list.size(); i++)
    data_component_storage_list[i] = source.data_component_storage_list[i];
  for (int i = 0; i < entity_metadata_list.size(); i++)
    entity_metadata_list[i].Clone(entity_links, source.entity_metadata_list[i], new_scene);

  entity_map = source.entity_map;
  entity_private_component_storage = source.entity_private_component_storage;
  entity_private_component_storage.owner_scene = new_scene;
}

KeyActionType Scene::GetKey(int key) {
  const auto search = pressed_keys_.find(key);
  if (search != pressed_keys_.end())
    return search->second;
  return KeyActionType::Release;
}

#pragma region Entity Management
void Scene::UnsafeForEachDataComponent(const Entity& entity,
                                       const std::function<void(const DataComponentType& type, void* data)>& func) {
  assert(IsEntityValid(entity));
  EntityMetadata& entityInfo = scene_data_storage_.entity_metadata_list.at(entity.index_);
  auto& dataComponentStorage = scene_data_storage_.data_component_storage_list[entityInfo.data_component_storage_index];
  const size_t chunkIndex = entityInfo.chunk_array_index / dataComponentStorage.chunk_capacity;
  const size_t chunkPointer = entityInfo.chunk_array_index % dataComponentStorage.chunk_capacity;
  const ComponentDataChunk& chunk = dataComponentStorage.chunk_array.chunk_array[chunkIndex];
  for (const auto& i : dataComponentStorage.data_component_types) {
    func(i, static_cast<void*>(static_cast<char*>(chunk.chunk_data) +
                               i.type_offset * dataComponentStorage.chunk_capacity + chunkPointer * i.type_size));
  }
}

void Scene::ForEachPrivateComponent(const Entity& entity,
                                    const std::function<void(PrivateComponentElement& data)>& func) {
  assert(IsEntityValid(entity));
  auto elements = scene_data_storage_.entity_metadata_list.at(entity.index_).private_component_elements;
  for (auto& component : elements) {
    func(component);
  }
}

void Scene::UnsafeForEachEntityStorage(
    const std::function<void(int i, const std::string& name, const DataComponentStorage& storage)>& func) {
  auto& archetypeInfos = Entities::GetInstance().entity_archetype_infos_;
  for (int i = 0; i < archetypeInfos.size(); i++) {
    auto dcs = GetDataComponentStorage(i);
    if (!dcs.has_value())
      continue;
    func(i, archetypeInfos[i].archetype_name, dcs->first.get());
  }
}

void Scene::DeleteEntityInternal(unsigned entity_index) {
  EntityMetadata& entityInfo = scene_data_storage_.entity_metadata_list.at(entity_index);
  auto& dataComponentStorage = scene_data_storage_.data_component_storage_list[entityInfo.data_component_storage_index];
  Entity actualEntity = scene_data_storage_.entities.at(entity_index);

  scene_data_storage_.entity_private_component_storage.DeleteEntity(actualEntity);
  entityInfo.entity_version = actualEntity.version_ + 1;
  entityInfo.entity_enabled = true;
  entityInfo.entity_static = false;
  entityInfo.ancestor_selected = false;
  scene_data_storage_.entity_map.erase(entityInfo.entity_handle);
  entityInfo.entity_handle = Handle(0);

  entityInfo.private_component_elements.clear();
  // Set to version 0, marks it as deleted.
  actualEntity.version_ = 0;
  dataComponentStorage.chunk_array.entity_array[entityInfo.chunk_array_index] = actualEntity;
  const auto originalIndex = entityInfo.chunk_array_index;
  if (entityInfo.chunk_array_index != dataComponentStorage.entity_alive_count - 1) {
    const auto swappedIndex =
        SwapEntity(dataComponentStorage, entityInfo.chunk_array_index, dataComponentStorage.entity_alive_count - 1);
    entityInfo.chunk_array_index = dataComponentStorage.entity_alive_count - 1;
    scene_data_storage_.entity_metadata_list.at(swappedIndex).chunk_array_index = originalIndex;
  }
  dataComponentStorage.entity_alive_count--;

  scene_data_storage_.entities.at(entity_index) = actualEntity;
}

std::optional<std::pair<std::reference_wrapper<DataComponentStorage>, unsigned>> Scene::GetDataComponentStorage(
    unsigned entity_archetype_index) {
  auto& archetypeInfo = Entities::GetInstance().entity_archetype_infos_.at(entity_archetype_index);
  int targetIndex = 0;
  for (auto& i : scene_data_storage_.data_component_storage_list) {
    if (i.data_component_types.size() != archetypeInfo.data_component_types.size()) {
      targetIndex++;
      continue;
    }
    bool check = true;
    for (int j = 0; j < i.data_component_types.size(); j++) {
      if (i.data_component_types[j].type_name != archetypeInfo.data_component_types[j].type_name) {
        check = false;
        break;
      }
    }
    if (check) {
      return {{std::ref(i), targetIndex}};
    }
    targetIndex++;
  }
  // If we didn't find the target storage, then we need to create a new one.
  scene_data_storage_.data_component_storage_list.emplace_back(archetypeInfo);
  return {{std::ref(scene_data_storage_.data_component_storage_list.back()),
           scene_data_storage_.data_component_storage_list.size() - 1}};
}

std::optional<std::pair<std::reference_wrapper<DataComponentStorage>, unsigned>> Scene::GetDataComponentStorage(
    const EntityArchetype& entity_archetype) {
  return GetDataComponentStorage(entity_archetype.index_);
}

std::vector<std::reference_wrapper<DataComponentStorage>> Scene::QueryDataComponentStorageList(
    const EntityQuery& entity_query) {
  return QueryDataComponentStorageList(entity_query.index_);
}

void Scene::GetEntityStorage(const DataComponentStorage& storage, std::vector<Entity>& container,
                             const bool check_enable) const {
  const size_t amount = storage.entity_alive_count;
  if (amount == 0)
    return;
  if (check_enable) {
    const auto threadSize = Jobs::GetWorkerSize();
    std::vector<std::vector<Entity>> tempStorage;
    tempStorage.resize(threadSize);
    const auto& chunkArray = storage.chunk_array;
    const auto& entities = &chunkArray.entity_array;
    Jobs::RunParallelFor(
        amount,
        [=, &entities, &tempStorage](const int i, const unsigned workerIndex) {
          const auto entity = entities->at(i);
          if (!scene_data_storage_.entity_metadata_list.at(entity.index_).entity_enabled)
            return;
          tempStorage[workerIndex].push_back(entity);
        },
        threadSize);
    for (auto& i : tempStorage) {
      container.insert(container.end(), i.begin(), i.end());
    }
  } else {
    container.resize(container.size() + amount);
    const size_t capacity = storage.chunk_capacity;
    memcpy(&container.at(container.size() - amount), storage.chunk_array.entity_array.data(), amount * sizeof(Entity));
  }
}

auto Scene::SwapEntity(DataComponentStorage& storage, const size_t index1, const size_t index2) -> size_t {
  if (index1 == index2)
    return -1;
  const size_t retVal = storage.chunk_array.entity_array[index2].index_;
  const auto other = storage.chunk_array.entity_array[index2];
  storage.chunk_array.entity_array[index2] = storage.chunk_array.entity_array[index1];
  storage.chunk_array.entity_array[index1] = other;
  const auto capacity = storage.chunk_capacity;
  const auto chunkIndex1 = index1 / capacity;
  const auto chunkIndex2 = index2 / capacity;
  const auto chunkPointer1 = index1 % capacity;
  const auto chunkPointer2 = index2 % capacity;
  for (const auto& i : storage.data_component_types) {
    void* temp = static_cast<void*>(malloc(i.type_size));
    void* d1 = static_cast<void*>(static_cast<char*>(storage.chunk_array.chunk_array[chunkIndex1].chunk_data) +
                                  i.type_offset * capacity + i.type_size * chunkPointer1);

    void* d2 = static_cast<void*>(static_cast<char*>(storage.chunk_array.chunk_array[chunkIndex2].chunk_data) +
                                  i.type_offset * capacity + i.type_size * chunkPointer2);

    memcpy(temp, d1, i.type_size);
    memcpy(d1, d2, i.type_size);
    memcpy(d2, temp, i.type_size);
    free(temp);
  }
  return retVal;
}

void Scene::GetAllEntities(std::vector<Entity>& target) {
  target.insert(target.end(), scene_data_storage_.entities.begin() + 1, scene_data_storage_.entities.end());
}

void Scene::ForEachDescendant(const Entity& target, const std::function<void(const Entity& entity)>& func,
                              const bool& from_root) {
  Entity realTarget = target;
  if (!IsEntityValid(realTarget))
    return;
  if (from_root)
    realTarget = GetRoot(realTarget);
  ForEachDescendantHelper(realTarget, func);
}

const std::vector<Entity>& Scene::UnsafeGetAllEntities() {
  return scene_data_storage_.entities;
}

Entity Scene::CreateEntity(const std::string& name) {
  return CreateEntity(Entities::GetInstance().basic_archetype_, name);
}

Entity Scene::CreateEntity(const EntityArchetype& archetype, const std::string& name, const Handle& handle) {
  assert(archetype.IsValid());

  Entity retVal;
  auto search = GetDataComponentStorage(archetype);
  DataComponentStorage& storage = search->first;
  if (storage.entity_count == storage.entity_alive_count) {
    const size_t chunkIndex = storage.entity_count / storage.chunk_capacity + 1;
    if (storage.chunk_array.chunk_array.size() <= chunkIndex) {
      // Allocate new chunk;
      ComponentDataChunk chunk;
      chunk.chunk_data = static_cast<void*>(calloc(1, Entities::GetArchetypeChunkSize()));
      storage.chunk_array.chunk_array.push_back(chunk);
    }
    retVal.index_ = scene_data_storage_.entities.size();
    // If the version is 0 in chunk means it's deleted.
    retVal.version_ = 1;
    EntityMetadata entityInfo;
    entityInfo.root = retVal;
    entityInfo.entity_static = false;
    entityInfo.entity_name = name;
    entityInfo.entity_handle = handle;
    entityInfo.data_component_storage_index = search->second;
    entityInfo.chunk_array_index = storage.entity_count;
    storage.chunk_array.entity_array.push_back(retVal);

    scene_data_storage_.entity_map[entityInfo.entity_handle] = retVal;
    scene_data_storage_.entity_metadata_list.push_back(std::move(entityInfo));
    scene_data_storage_.entities.push_back(retVal);
    storage.entity_count++;
    storage.entity_alive_count++;
  } else {
    retVal = storage.chunk_array.entity_array.at(storage.entity_alive_count);
    EntityMetadata& entityInfo = scene_data_storage_.entity_metadata_list.at(retVal.index_);
    entityInfo.root = retVal;
    entityInfo.entity_static = false;
    entityInfo.entity_handle = handle;
    entityInfo.entity_enabled = true;
    entityInfo.entity_name = name;
    retVal.version_ = entityInfo.entity_version;

    scene_data_storage_.entity_map[entityInfo.entity_handle] = retVal;
    storage.chunk_array.entity_array[entityInfo.chunk_array_index] = retVal;
    scene_data_storage_.entities.at(retVal.index_) = retVal;
    storage.entity_alive_count++;
    // Reset all component data
    const auto chunkIndex = entityInfo.chunk_array_index / storage.chunk_capacity;
    const auto chunkPointer = entityInfo.chunk_array_index % storage.chunk_capacity;
    const auto chunk = storage.chunk_array.chunk_array[chunkIndex];
    for (const auto& i : storage.data_component_types) {
      const auto offset = i.type_offset * storage.chunk_capacity + chunkPointer * i.type_size;
      chunk.ClearData(offset, i.type_size);
    }
  }
  SetDataComponent(retVal, Transform());
  SetDataComponent(retVal, GlobalTransform());
  SetDataComponent(retVal, TransformUpdateFlag());
  SetUnsaved();
  return retVal;
}

std::vector<Entity> Scene::CreateEntities(const EntityArchetype& archetype, const size_t& amount,
                                          const std::string& name) {
  assert(archetype.IsValid());
  std::vector<Entity> retVal;
  auto search = GetDataComponentStorage(archetype);
  DataComponentStorage& storage = search->first;
  auto remainAmount = amount;
  const Transform transform;
  const GlobalTransform globalTransform;
  const TransformUpdateFlag transformStatus;
  while (remainAmount > 0 && storage.entity_alive_count != storage.entity_count) {
    remainAmount--;
    Entity entity = storage.chunk_array.entity_array.at(storage.entity_alive_count);
    EntityMetadata& entityInfo = scene_data_storage_.entity_metadata_list.at(entity.index_);
    entityInfo.root = entity;
    entityInfo.entity_static = false;
    entityInfo.entity_enabled = true;
    entityInfo.entity_name = name;
    entity.version_ = entityInfo.entity_version;
    entityInfo.entity_handle = Handle();
    scene_data_storage_.entity_map[entityInfo.entity_handle] = entity;
    storage.chunk_array.entity_array[entityInfo.chunk_array_index] = entity;
    scene_data_storage_.entities.at(entity.index_) = entity;
    storage.entity_alive_count++;
    // Reset all component data
    const size_t chunkIndex = entityInfo.chunk_array_index / storage.chunk_capacity;
    const size_t chunkPointer = entityInfo.chunk_array_index % storage.chunk_capacity;
    const ComponentDataChunk& chunk = storage.chunk_array.chunk_array[chunkIndex];
    for (const auto& i : storage.data_component_types) {
      const size_t offset = i.type_offset * storage.chunk_capacity + chunkPointer * i.type_size;
      chunk.ClearData(offset, i.type_size);
    }
    retVal.push_back(entity);
    SetDataComponent(entity, transform);
    SetDataComponent(entity, globalTransform);
    SetDataComponent(entity, TransformUpdateFlag());
  }
  if (remainAmount == 0)
    return retVal;
  storage.entity_count += remainAmount;
  storage.entity_alive_count += remainAmount;
  const size_t chunkIndex = storage.entity_count / storage.chunk_capacity + 1;
  while (storage.chunk_array.chunk_array.size() <= chunkIndex) {
    // Allocate new chunk;
    ComponentDataChunk chunk;
    chunk.chunk_data = static_cast<void*>(calloc(1, Entities::GetArchetypeChunkSize()));
    storage.chunk_array.chunk_array.push_back(chunk);
  }
  const size_t originalSize = scene_data_storage_.entities.size();
  scene_data_storage_.entities.resize(originalSize + remainAmount);
  scene_data_storage_.entity_metadata_list.resize(originalSize + remainAmount);

  for (int i = 0; i < remainAmount; i++) {
    auto& entity = scene_data_storage_.entities.at(originalSize + i);
    entity.index_ = originalSize + i;
    entity.version_ = 1;

    auto& entityInfo = scene_data_storage_.entity_metadata_list.at(originalSize + i);
    entityInfo = EntityMetadata();
    entityInfo.root = entity;
    entityInfo.entity_static = false;
    entityInfo.entity_name = name;
    entityInfo.data_component_storage_index = search->second;
    entityInfo.chunk_array_index = storage.entity_alive_count - remainAmount + i;

    entityInfo.entity_handle = Handle();

    scene_data_storage_.entity_map[entityInfo.entity_handle] = entity;
  }

  storage.chunk_array.entity_array.insert(storage.chunk_array.entity_array.end(),
                                          scene_data_storage_.entities.begin() + originalSize,
                                          scene_data_storage_.entities.end());
  Jobs::RunParallelFor(remainAmount, [&, originalSize](unsigned i) {
    const auto& entity = scene_data_storage_.entities.at(originalSize + i);
    SetDataComponent(entity, transform);
    SetDataComponent(entity, globalTransform);
    SetDataComponent(entity, TransformUpdateFlag());
  });

  retVal.insert(retVal.end(), scene_data_storage_.entities.begin() + originalSize, scene_data_storage_.entities.end());
  SetUnsaved();
  return retVal;
}

std::vector<Entity> Scene::CreateEntities(const size_t& amount, const std::string& name) {
  return CreateEntities(Entities::GetInstance().basic_archetype_, amount, name);
}

void Scene::DeleteEntity(const Entity& entity) {
  if (!IsEntityValid(entity)) {
    return;
  }
  const size_t entityIndex = entity.index_;
  auto children = scene_data_storage_.entity_metadata_list.at(entityIndex).children;
  for (const auto& child : children) {
    DeleteEntity(child);
  }
  if (scene_data_storage_.entity_metadata_list.at(entityIndex).parent.index_ != 0)
    RemoveChild(entity, scene_data_storage_.entity_metadata_list.at(entityIndex).parent);
  DeleteEntityInternal(entity.index_);
  SetUnsaved();
}

std::string Scene::GetEntityName(const Entity& entity) {
  assert(IsEntityValid(entity));
  const size_t index = entity.index_;
  if (entity != scene_data_storage_.entities.at(index)) {
    EVOENGINE_ERROR("Child already deleted!");
    return "";
  }
  return scene_data_storage_.entity_metadata_list.at(index).entity_name;
}

void Scene::SetEntityName(const Entity& entity, const std::string& name) {
  assert(IsEntityValid(entity));
  const size_t index = entity.index_;
  if (entity != scene_data_storage_.entities.at(index)) {
    EVOENGINE_ERROR("Child already deleted!");
    return;
  }
  if (name.length() != 0) {
    scene_data_storage_.entity_metadata_list.at(index).entity_name = name;
    return;
  }
  scene_data_storage_.entity_metadata_list.at(index).entity_name = "Unnamed";
  SetUnsaved();
}
void Scene::SetEntityStatic(const Entity& entity, bool value) {
  assert(IsEntityValid(entity));
  auto& entityInfo = scene_data_storage_.entity_metadata_list.at(GetRoot(entity).index_);
  entityInfo.entity_static = value;
  SetUnsaved();
}
void Scene::SetParent(const Entity& child, const Entity& parent, const bool& recalculate_transform) {
  assert(IsEntityValid(child) && IsEntityValid(parent));
  const size_t childIndex = child.index_;
  const size_t parentIndex = parent.index_;
  auto& parentEntityInfo = scene_data_storage_.entity_metadata_list.at(parentIndex);
  for (const auto& i : parentEntityInfo.children) {
    if (i == child)
      return;
  }
  auto& childEntityInfo = scene_data_storage_.entity_metadata_list.at(childIndex);
  if (childEntityInfo.parent.GetIndex() != 0) {
    RemoveChild(child, childEntityInfo.parent);
  }

  if (recalculate_transform) {
    const auto childGlobalTransform = GetDataComponent<GlobalTransform>(child);
    const auto parentGlobalTransform = GetDataComponent<GlobalTransform>(parent);
    Transform childTransform;
    childTransform.value = glm::inverse(parentGlobalTransform.value) * childGlobalTransform.value;
    SetDataComponent(child, childTransform);
  }
  childEntityInfo.parent = parent;
  if (parentEntityInfo.parent.GetIndex() == childIndex) {
    parentEntityInfo.parent = Entity();
    parentEntityInfo.root = parent;
    const size_t childrenCount = childEntityInfo.children.size();

    for (size_t i = 0; i < childrenCount; i++) {
      if (childEntityInfo.children[i].index_ == parent.GetIndex()) {
        childEntityInfo.children[i] = childEntityInfo.children.back();
        childEntityInfo.children.pop_back();
        break;
      }
    }
  }
  childEntityInfo.root = parentEntityInfo.root;
  childEntityInfo.entity_static = false;
  parentEntityInfo.children.push_back(child);
  if (parentEntityInfo.ancestor_selected) {
    const auto descendants = GetDescendants(child);
    for (const auto& i : descendants) {
      GetEntityMetadata(i).ancestor_selected = true;
    }
    childEntityInfo.ancestor_selected = true;
  }
  SetUnsaved();
}

Entity Scene::GetParent(const Entity& entity) const {
  assert(IsEntityValid(entity));
  const size_t entityIndex = entity.index_;
  return scene_data_storage_.entity_metadata_list.at(entityIndex).parent;
}

std::vector<Entity> Scene::GetChildren(const Entity& entity) {
  assert(IsEntityValid(entity));
  const size_t entityIndex = entity.index_;
  return scene_data_storage_.entity_metadata_list.at(entityIndex).children;
}

Entity Scene::GetChild(const Entity& entity, int index) const {
  assert(IsEntityValid(entity));
  const size_t entityIndex = entity.index_;
  auto& children = scene_data_storage_.entity_metadata_list.at(entityIndex).children;
  if (children.size() > index)
    return children[index];
  return Entity();
}

size_t Scene::GetChildrenAmount(const Entity& entity) const {
  assert(IsEntityValid(entity));
  const size_t entityIndex = entity.index_;
  return scene_data_storage_.entity_metadata_list.at(entityIndex).children.size();
}

void Scene::ForEachChild(const Entity& entity, const std::function<void(Entity child)>& func) const {
  assert(IsEntityValid(entity));
  const auto children = scene_data_storage_.entity_metadata_list.at(entity.index_).children;
  for (auto i : children) {
    if (IsEntityValid(i))
      func(i);
  }
}

void Scene::RemoveChild(const Entity& child, const Entity& parent) {
  assert(IsEntityValid(child) && IsEntityValid(parent));
  const size_t childIndex = child.index_;
  const size_t parentIndex = parent.index_;
  auto& childEntityMetadata = scene_data_storage_.entity_metadata_list.at(childIndex);
  auto& parentEntityMetadata = scene_data_storage_.entity_metadata_list.at(parentIndex);
  if (childEntityMetadata.parent.index_ == 0) {
    EVOENGINE_ERROR("No child by the parent!");
  }
  childEntityMetadata.parent = Entity();
  childEntityMetadata.root = child;
  if (parentEntityMetadata.ancestor_selected) {
    const auto descendants = GetDescendants(child);
    for (const auto& i : descendants) {
      GetEntityMetadata(i).ancestor_selected = false;
    }
    childEntityMetadata.ancestor_selected = false;
  }
  const size_t childrenCount = parentEntityMetadata.children.size();

  for (size_t i = 0; i < childrenCount; i++) {
    if (parentEntityMetadata.children[i].index_ == childIndex) {
      parentEntityMetadata.children[i] = parentEntityMetadata.children.back();
      parentEntityMetadata.children.pop_back();
      break;
    }
  }
  const auto childGlobalTransform = GetDataComponent<GlobalTransform>(child);
  Transform childTransform;
  childTransform.value = childGlobalTransform.value;
  SetDataComponent(child, childTransform);
  SetUnsaved();
}

void Scene::RemoveDataComponent(const Entity& entity, const size_t& type_index) {
  assert(IsEntityValid(entity));
  if (type_index == typeid(Transform).hash_code() || type_index == typeid(GlobalTransform).hash_code() ||
      type_index == typeid(TransformUpdateFlag).hash_code()) {
    return;
  }
  EntityMetadata& entityInfo = scene_data_storage_.entity_metadata_list.at(entity.index_);
  auto& entityArchetypeInfos = Entities::GetInstance().entity_archetype_infos_;
  auto& dataComponentStorage = scene_data_storage_.data_component_storage_list[entityInfo.data_component_storage_index];
  if (dataComponentStorage.data_component_types.size() <= 3) {
    EVOENGINE_ERROR(
        "Remove Component Data failed: Entity must have at least 1 data component besides 3 basic data "
        "components!");
    return;
  }
#pragma region Create new archetype
  EntityArchetypeInfo newArchetypeInfo;
  newArchetypeInfo.archetype_name = "New archetype";
  newArchetypeInfo.data_component_types = dataComponentStorage.data_component_types;
  bool found = false;
  for (int i = 0; i < newArchetypeInfo.data_component_types.size(); i++) {
    if (newArchetypeInfo.data_component_types[i].type_index == type_index) {
      newArchetypeInfo.data_component_types.erase(newArchetypeInfo.data_component_types.begin() + i);
      found = true;
      break;
    }
  }
  if (!found) {
    EVOENGINE_ERROR("Failed to remove component data: Component not found");
    return;
  }
  size_t offset = 0;
  DataComponentType prev = newArchetypeInfo.data_component_types[0];
  for (auto& i : newArchetypeInfo.data_component_types) {
    i.type_offset = offset;
    offset += i.type_size;
  }
  newArchetypeInfo.entity_size =
      newArchetypeInfo.data_component_types.back().type_offset + newArchetypeInfo.data_component_types.back().type_size;
  newArchetypeInfo.chunk_capacity = Entities::GetArchetypeChunkSize() / newArchetypeInfo.entity_size;
  auto archetype = Entities::CreateEntityArchetypeHelper(newArchetypeInfo);
#pragma endregion
#pragma region Create new Entity with new archetype
  const Entity newEntity = CreateEntity(archetype);
  // Transfer component data
  for (const auto& type : newArchetypeInfo.data_component_types) {
    SetDataComponent(newEntity.index_, type.type_index, type.type_size,
                     GetDataComponentPointer(entity, type.type_index));
  }
  // 5. Swap entity.
  EntityMetadata& newEntityInfo = scene_data_storage_.entity_metadata_list.at(newEntity.index_);
  const auto tempArchetypeInfoIndex = newEntityInfo.data_component_storage_index;
  const auto tempChunkArrayIndex = newEntityInfo.chunk_array_index;
  newEntityInfo.data_component_storage_index = entityInfo.data_component_storage_index;
  newEntityInfo.chunk_array_index = entityInfo.chunk_array_index;
  entityInfo.data_component_storage_index = tempArchetypeInfoIndex;
  entityInfo.chunk_array_index = tempChunkArrayIndex;
  // Apply to chunk.
  scene_data_storage_.data_component_storage_list.at(entityInfo.data_component_storage_index)
      .chunk_array.entity_array[entityInfo.chunk_array_index] = entity;
  scene_data_storage_.data_component_storage_list.at(newEntityInfo.data_component_storage_index)
      .chunk_array.entity_array[newEntityInfo.chunk_array_index] = newEntity;
  DeleteEntity(newEntity);
#pragma endregion
  SetUnsaved();
}

void Scene::SetDataComponent(const unsigned& entity_index, size_t id, size_t size, IDataComponent* data) {
  auto& entityInfo = scene_data_storage_.entity_metadata_list.at(entity_index);
  auto& dataComponentStorage = scene_data_storage_.data_component_storage_list[entityInfo.data_component_storage_index];
  const auto chunkIndex = entityInfo.chunk_array_index / dataComponentStorage.chunk_capacity;
  const auto chunkPointer = entityInfo.chunk_array_index % dataComponentStorage.chunk_capacity;
  const auto chunk = dataComponentStorage.chunk_array.chunk_array[chunkIndex];
  if (id == typeid(Transform).hash_code()) {
    chunk.SetData(static_cast<size_t>(chunkPointer * sizeof(Transform)), sizeof(Transform), data);
    static_cast<TransformUpdateFlag*>(
        chunk.GetDataPointer(
            static_cast<size_t>((sizeof(Transform) + sizeof(GlobalTransform)) * dataComponentStorage.chunk_capacity +
                                chunkPointer * sizeof(TransformUpdateFlag))))
        ->transform_modified = true;
  } else if (id == typeid(GlobalTransform).hash_code()) {
    chunk.SetData(static_cast<size_t>(sizeof(Transform) * dataComponentStorage.chunk_capacity +
                                      chunkPointer * sizeof(GlobalTransform)),
                  sizeof(GlobalTransform), data);
    static_cast<TransformUpdateFlag*>(
        chunk.GetDataPointer(
            static_cast<size_t>((sizeof(Transform) + sizeof(GlobalTransform)) * dataComponentStorage.chunk_capacity +
                                chunkPointer * sizeof(TransformUpdateFlag))))
        ->global_transform_modified = true;
  } else if (id == typeid(TransformUpdateFlag).hash_code()) {
    chunk.SetData(
        static_cast<size_t>((sizeof(Transform) + sizeof(GlobalTransform)) * dataComponentStorage.chunk_capacity +
                            chunkPointer * sizeof(TransformUpdateFlag)),
        sizeof(TransformUpdateFlag), data);
  } else {
    for (const auto& type : dataComponentStorage.data_component_types) {
      if (type.type_index == id) {
        chunk.SetData(
            static_cast<size_t>(type.type_offset * dataComponentStorage.chunk_capacity + chunkPointer * type.type_size),
            size, data);
        return;
      }
    }
    EVOENGINE_LOG("ComponentData doesn't exist");
  }
  SetUnsaved();
}
IDataComponent* Scene::GetDataComponentPointer(unsigned entity_index, const size_t& id) {
  EntityMetadata& entityInfo = scene_data_storage_.entity_metadata_list.at(entity_index);
  auto& dataComponentStorage = scene_data_storage_.data_component_storage_list[entityInfo.data_component_storage_index];
  const auto chunkIndex = entityInfo.chunk_array_index / dataComponentStorage.chunk_capacity;
  const auto chunkPointer = entityInfo.chunk_array_index % dataComponentStorage.chunk_capacity;
  const auto chunk = dataComponentStorage.chunk_array.chunk_array[chunkIndex];
  if (id == typeid(Transform).hash_code()) {
    return chunk.GetDataPointer(static_cast<size_t>(chunkPointer * sizeof(Transform)));
  }
  if (id == typeid(GlobalTransform).hash_code()) {
    return chunk.GetDataPointer(static_cast<size_t>(sizeof(Transform) * dataComponentStorage.chunk_capacity +
                                                    chunkPointer * sizeof(GlobalTransform)));
  }
  if (id == typeid(TransformUpdateFlag).hash_code()) {
    return chunk.GetDataPointer(
        static_cast<size_t>((sizeof(Transform) + sizeof(GlobalTransform)) * dataComponentStorage.chunk_capacity +
                            chunkPointer * sizeof(TransformUpdateFlag)));
  }
  for (const auto& type : dataComponentStorage.data_component_types) {
    if (type.type_index == id) {
      return chunk.GetDataPointer(
          static_cast<size_t>(type.type_offset * dataComponentStorage.chunk_capacity + chunkPointer * type.type_size));
    }
  }
  EVOENGINE_LOG("ComponentData doesn't exist");
  return nullptr;
}
IDataComponent* Scene::GetDataComponentPointer(const Entity& entity, const size_t& id) {
  assert(IsEntityValid(entity));
  EntityMetadata& entityInfo = scene_data_storage_.entity_metadata_list.at(entity.index_);
  auto& dataComponentStorage = scene_data_storage_.data_component_storage_list[entityInfo.data_component_storage_index];
  const auto chunkIndex = entityInfo.chunk_array_index / dataComponentStorage.chunk_capacity;
  const auto chunkPointer = entityInfo.chunk_array_index % dataComponentStorage.chunk_capacity;
  const auto chunk = dataComponentStorage.chunk_array.chunk_array[chunkIndex];
  if (id == typeid(Transform).hash_code()) {
    return chunk.GetDataPointer(static_cast<size_t>(chunkPointer * sizeof(Transform)));
  }
  if (id == typeid(GlobalTransform).hash_code()) {
    return chunk.GetDataPointer(static_cast<size_t>(sizeof(Transform) * dataComponentStorage.chunk_capacity +
                                                    chunkPointer * sizeof(GlobalTransform)));
  }
  if (id == typeid(TransformUpdateFlag).hash_code()) {
    return chunk.GetDataPointer(
        static_cast<size_t>((sizeof(Transform) + sizeof(GlobalTransform)) * dataComponentStorage.chunk_capacity +
                            chunkPointer * sizeof(TransformUpdateFlag)));
  }
  for (const auto& type : dataComponentStorage.data_component_types) {
    if (type.type_index == id) {
      return chunk.GetDataPointer(
          static_cast<size_t>(type.type_offset * dataComponentStorage.chunk_capacity + chunkPointer * type.type_size));
    }
  }
  EVOENGINE_LOG("ComponentData doesn't exist");
  return nullptr;
}
Handle Scene::GetEntityHandle(const Entity& entity) {
  return scene_data_storage_.entity_metadata_list.at(entity.index_).entity_handle;
}
void Scene::SetPrivateComponent(const Entity& entity, const std::shared_ptr<IPrivateComponent>& ptr) {
  assert(ptr && IsEntityValid(entity));
  auto typeName = ptr->GetTypeName();
  auto& elements = scene_data_storage_.entity_metadata_list.at(entity.index_).private_component_elements;
  for (auto& element : elements) {
    if (typeName == element.private_component_data->GetTypeName()) {
      return;
    }
  }

  auto id = Serialization::GetSerializableTypeId(typeName);
  scene_data_storage_.entity_private_component_storage.SetPrivateComponent(entity, id);
  elements.emplace_back(id, ptr, entity, std::dynamic_pointer_cast<Scene>(GetSelf()));
  SetUnsaved();
}

void Scene::ForEachDescendantHelper(const Entity& target, const std::function<void(const Entity& entity)>& func) {
  func(target);
  ForEachChild(target, [&](Entity child) {
    ForEachDescendantHelper(child, func);
  });
}

Entity Scene::GetRoot(const Entity& entity) {
  Entity retVal = entity;
  auto parent = GetParent(retVal);
  while (parent.GetIndex() != 0) {
    retVal = parent;
    parent = GetParent(retVal);
  }
  return retVal;
}

Entity Scene::GetEntity(const size_t& index) {
  if (index > 0 && index < scene_data_storage_.entities.size())
    return scene_data_storage_.entities.at(index);
  return {};
}

void Scene::RemovePrivateComponent(const Entity& entity, size_t type_id) {
  assert(IsEntityValid(entity));
  auto& privateComponentElements =
      scene_data_storage_.entity_metadata_list.at(entity.index_).private_component_elements;
  for (auto i = 0; i < privateComponentElements.size(); i++) {
    if (privateComponentElements[i].type_index == type_id) {
      scene_data_storage_.entity_private_component_storage.RemovePrivateComponent(
          entity, type_id, privateComponentElements[i].private_component_data);
      privateComponentElements.erase(privateComponentElements.begin() + i);
      SetUnsaved();
      break;
    }
  }
}

void Scene::SetEnable(const Entity& entity, const bool& value) {
  assert(IsEntityValid(entity));
  if (scene_data_storage_.entity_metadata_list.at(entity.index_).entity_enabled != value) {
    for (auto& i : scene_data_storage_.entity_metadata_list.at(entity.index_).private_component_elements) {
      if (value) {
        i.private_component_data->OnEntityEnable();
      } else {
        i.private_component_data->OnEntityDisable();
      }
    }
  }
  scene_data_storage_.entity_metadata_list.at(entity.index_).entity_enabled = value;

  for (const auto& i : scene_data_storage_.entity_metadata_list.at(entity.index_).children) {
    SetEnable(i, value);
  }
  SetUnsaved();
}

void Scene::SetEnableSingle(const Entity& entity, const bool& value) {
  assert(IsEntityValid(entity));
  auto& entityMetadata = scene_data_storage_.entity_metadata_list.at(entity.index_);
  if (entityMetadata.entity_enabled != value) {
    for (auto& i : entityMetadata.private_component_elements) {
      if (value) {
        i.private_component_data->OnEntityEnable();
      } else {
        i.private_component_data->OnEntityDisable();
      }
    }
    entityMetadata.entity_enabled = value;
  }
}
EntityMetadata& Scene::GetEntityMetadata(const Entity& entity) {
  assert(IsEntityValid(entity));
  return scene_data_storage_.entity_metadata_list.at(entity.index_);
}
void Scene::ForAllEntities(const std::function<void(int i, Entity entity)>& func) const {
  for (int index = 0; index < scene_data_storage_.entities.size(); index++) {
    if (scene_data_storage_.entities.at(index).version_ != 0) {
      func(index, scene_data_storage_.entities.at(index));
    }
  }
}

Bound Scene::GetEntityBoundingBox(const Entity& entity) {
  auto descendants = GetDescendants(entity);
  descendants.emplace_back(entity);

  Bound retVal{};

  for (const auto& walker : descendants) {
    auto gt = GetDataComponent<GlobalTransform>(walker);
    if (HasPrivateComponent<MeshRenderer>(walker)) {
      auto meshRenderer = GetOrSetPrivateComponent<MeshRenderer>(walker).lock();
      if (const auto mesh = meshRenderer->m_mesh.Get<Mesh>()) {
        auto meshBound = mesh->GetBound();
        meshBound.ApplyTransform(gt.value);
        glm::vec3 center = meshBound.Center();

        glm::vec3 size = meshBound.Size();
        retVal.min = glm::vec3((glm::min)(retVal.min.x, center.x - size.x), (glm::min)(retVal.min.y, center.y - size.y),
                               (glm::min)(retVal.min.z, center.z - size.z));
        retVal.max = glm::vec3((glm::max)(retVal.max.x, center.x + size.x), (glm::max)(retVal.max.y, center.y + size.y),
                               (glm::max)(retVal.max.z, center.z + size.z));
      }
    } else if (HasPrivateComponent<SkinnedMeshRenderer>(walker)) {
      auto meshRenderer = GetOrSetPrivateComponent<SkinnedMeshRenderer>(walker).lock();
      if (const auto mesh = meshRenderer->m_skinnedMesh.Get<SkinnedMesh>()) {
        auto meshBound = mesh->GetBound();
        meshBound.ApplyTransform(gt.value);
        glm::vec3 center = meshBound.Center();

        glm::vec3 size = meshBound.Size();
        retVal.min = glm::vec3((glm::min)(retVal.min.x, center.x - size.x), (glm::min)(retVal.min.y, center.y - size.y),
                               (glm::min)(retVal.min.z, center.z - size.z));
        retVal.max = glm::vec3((glm::max)(retVal.max.x, center.x + size.x), (glm::max)(retVal.max.y, center.y + size.y),
                               (glm::max)(retVal.max.z, center.z + size.z));
      }
    }
  }

  return retVal;
}

void Scene::GetEntityArray(const EntityQuery& entity_query, std::vector<Entity>& container, bool check_enable) {
  assert(entity_query.IsValid());
  auto queriedStorages = QueryDataComponentStorageList(entity_query);
  for (const auto i : queriedStorages) {
    GetEntityStorage(i.get(), container, check_enable);
  }
}

size_t Scene::GetEntityAmount(EntityQuery entity_query, bool check_enable) {
  assert(entity_query.IsValid());
  size_t retVal = 0;
  if (check_enable) {
    auto queriedStorages = QueryDataComponentStorageList(entity_query);
    for (const auto i : queriedStorages) {
      for (int index = 0; index < i.get().entity_alive_count; index++) {
        if (IsEntityEnabled(i.get().chunk_array.entity_array[index]))
          retVal++;
      }
    }
  } else {
    auto queriedStorages = QueryDataComponentStorageList(entity_query);
    for (const auto i : queriedStorages) {
      retVal += i.get().entity_alive_count;
    }
  }
  return retVal;
}

std::vector<Entity> Scene::GetDescendants(const Entity& entity) {
  std::vector<Entity> retVal;
  if (!IsEntityValid(entity))
    return retVal;
  GetDescendantsHelper(entity, retVal);
  return retVal;
}
void Scene::GetDescendantsHelper(const Entity& target, std::vector<Entity>& results) {
  auto& children = scene_data_storage_.entity_metadata_list.at(target.index_).children;
  if (!children.empty())
    results.insert(results.end(), children.begin(), children.end());
  for (const auto& i : children)
    GetDescendantsHelper(i, results);
}
template <typename T>
std::vector<Entity> Scene::GetPrivateComponentOwnersList(const std::shared_ptr<Scene>& scene) {
  return scene_data_storage_.entity_private_component_storage.GetOwnersList<T>();
}

std::weak_ptr<IPrivateComponent> Scene::GetPrivateComponent(const Entity& entity, const std::string& type_name) {
  size_t i = 0;
  auto& elements = scene_data_storage_.entity_metadata_list.at(entity.index_).private_component_elements;
  for (auto& element : elements) {
    if (type_name == element.private_component_data->GetTypeName()) {
      return element.private_component_data;
    }
    i++;
  }
  throw 0;
}

Entity Scene::GetEntity(const Handle& handle) {
  auto search = scene_data_storage_.entity_map.find(handle);
  if (search != scene_data_storage_.entity_map.end()) {
    return search->second;
  }
  return {};
}
bool Scene::HasPrivateComponent(const Entity& entity, const std::string& type_name) const {
  assert(IsEntityValid(entity));
  for (auto& element : scene_data_storage_.entity_metadata_list.at(entity.index_).private_component_elements) {
    if (element.private_component_data->type_name_ == type_name) {
      return true;
    }
  }
  return false;
}
std::vector<std::reference_wrapper<DataComponentStorage>> Scene::QueryDataComponentStorageList(
    unsigned int entity_query_index) {
  const auto& queryInfos = Entities::GetInstance().entity_query_infos_.at(entity_query_index);
  auto& entityComponentStorage = scene_data_storage_.data_component_storage_list;
  std::vector<std::reference_wrapper<DataComponentStorage>> queriedStorage;
  // Select storage with every contained.
  if (!queryInfos.all_data_component_types.empty()) {
    for (int i = 0; i < entityComponentStorage.size(); i++) {
      auto& dataStorage = entityComponentStorage.at(i);
      bool check = true;
      for (const auto& type : queryInfos.all_data_component_types) {
        if (!dataStorage.HasType(type.type_index))
          check = false;
      }
      if (check)
        queriedStorage.push_back(std::ref(dataStorage));
    }
  } else {
    for (int i = 0; i < entityComponentStorage.size(); i++) {
      auto& dataStorage = entityComponentStorage.at(i);
      queriedStorage.push_back(std::ref(dataStorage));
    }
  }
  // Erase with any
  if (!queryInfos.any_data_component_types.empty()) {
    for (int i = 0; i < queriedStorage.size(); i++) {
      bool contain = false;
      for (const auto& type : queryInfos.any_data_component_types) {
        if (queriedStorage.at(i).get().HasType(type.type_index))
          contain = true;
        if (contain)
          break;
      }
      if (!contain) {
        queriedStorage.erase(queriedStorage.begin() + i);
        i--;
      }
    }
  }
  // Erase with none
  if (!queryInfos.none_data_component_types.empty()) {
    for (int i = 0; i < queriedStorage.size(); i++) {
      bool contain = false;
      for (const auto& type : queryInfos.none_data_component_types) {
        if (queriedStorage.at(i).get().HasType(type.type_index))
          contain = true;
        if (contain)
          break;
      }
      if (contain) {
        queriedStorage.erase(queriedStorage.begin() + i);
        i--;
      }
    }
  }
  return queriedStorage;
}
bool Scene::IsEntityValid(const Entity& entity) const {
  auto& storage = scene_data_storage_.entities;
  return entity.index_ != 0 && entity.version_ != 0 && entity.index_ < storage.size() &&
         storage.at(entity.index_).version_ == entity.version_;
}
bool Scene::IsEntityEnabled(const Entity& entity) const {
  assert(IsEntityValid(entity));
  return scene_data_storage_.entity_metadata_list.at(entity.index_).entity_enabled;
}
bool Scene::IsEntityRoot(const Entity& entity) const {
  assert(IsEntityValid(entity));
  return scene_data_storage_.entity_metadata_list.at(entity.index_).root == entity;
}
bool Scene::IsEntityStatic(const Entity& entity) {
  assert(IsEntityValid(entity));
  return scene_data_storage_.entity_metadata_list.at(GetRoot(entity).index_).entity_static;
}

bool Scene::IsEntityAncestorSelected(const Entity& entity) const {
  assert(IsEntityValid(entity));
  return scene_data_storage_.entity_metadata_list.at(entity.index_).ancestor_selected;
}

#pragma endregion
