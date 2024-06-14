// #include "ProfilerLayer.hpp"
// #include "RigidBody.hpp"
#include "TransformGraph.hpp"
#include "Application.hpp"
#include "ClassRegistry.hpp"
#include "Jobs.hpp"
#include "Scene.hpp"
using namespace evo_engine;

DataComponentRegistration<Transform> TransformRegistry("Transform");
DataComponentRegistration<GlobalTransform> GlobalTransformRegistry("GlobalTransform");
DataComponentRegistration<TransformUpdateFlag> TransformUpdateStatusRegistry("TransformUpdateFlag");

void TransformGraph::Initialize() {
  auto& transformGraph = GetInstance();
  transformGraph.transform_query_ = Entities::CreateEntityQuery();
  Entities::SetEntityQueryAllFilters(transformGraph.transform_query_, Transform(), GlobalTransform());
}

void TransformGraph::CalculateTransformGraph(const std::shared_ptr<Scene>& scene,
                                             const std::vector<EntityMetadata>& entity_infos,
                                             const GlobalTransform& parent_global_transform, const Entity& parent) {
  const auto& entityInfo = entity_infos.at(parent.GetIndex());
  for (const auto& entity : entityInfo.children) {
    auto* transformStatus = reinterpret_cast<TransformUpdateFlag*>(
        scene->GetDataComponentPointer(entity.GetIndex(), typeid(TransformUpdateFlag).hash_code()));
    GlobalTransform ltw;
    if (transformStatus->m_globalTransformModified) {
      ltw = scene->GetDataComponent<GlobalTransform>(entity.GetIndex());
      reinterpret_cast<Transform*>(scene->GetDataComponentPointer(entity.GetIndex(), typeid(Transform).hash_code()))
          ->m_value = glm::inverse(parent_global_transform.m_value) * ltw.m_value;
      transformStatus->m_globalTransformModified = false;
    } else {
      auto ltp = scene->GetDataComponent<Transform>(entity.GetIndex());
      ltw.m_value = parent_global_transform.m_value * ltp.m_value;
      *reinterpret_cast<GlobalTransform*>(
          scene->GetDataComponentPointer(entity.GetIndex(), typeid(GlobalTransform).hash_code())) = ltw;
    }
    transformStatus->m_transformModified = false;
    CalculateTransformGraph(scene, entity_infos, ltw, entity);
  }
}
void TransformGraph::CalculateTransformGraphs(const std::shared_ptr<Scene>& scene, const bool check_static) {
  if (!scene)
    return;
  auto& transformGraph = GetInstance();
  const auto& entityInfos = scene->scene_data_storage_.entity_metadata_list;
  // ProfilerLayer::StartEvent("TransformManager");
  Jobs::Wait(scene->ForEach<Transform, GlobalTransform, TransformUpdateFlag>(
      {}, transformGraph.transform_query_,
      [&](int i, Entity entity, Transform& transform, GlobalTransform& globalTransform,
          TransformUpdateFlag& transformStatus) {
        const EntityMetadata& entityInfo = scene->scene_data_storage_.entity_metadata_list.at(entity.GetIndex());
        if (entityInfo.parent.GetIndex() != 0) {
          transformStatus.m_transformModified = false;
          return;
        }
        if (check_static && entityInfo.entity_static) {
          transformStatus.m_transformModified = false;
          return;
        }
        if (transformStatus.m_globalTransformModified) {
          transform.m_value = globalTransform.m_value;
          transformStatus.m_globalTransformModified = false;
        } else {
          globalTransform.m_value = transform.m_value;
        }
        transformGraph.CalculateTransformGraph(scene, entityInfos, globalTransform, entity);
      },
      false));

  transformGraph.physics_system_override_ = false;
  // ProfilerLayer::EndEvent("TransformManager");
}
void TransformGraph::CalculateTransformGraphForDescendants(const std::shared_ptr<Scene>& scene, const Entity& entity) {
  if (!scene)
    return;
  auto& transformGraph = GetInstance();
  const auto& entityInfos = scene->scene_data_storage_.entity_metadata_list;
  transformGraph.CalculateTransformGraph(scene, entityInfos,
                                         scene->GetDataComponent<GlobalTransform>(entity.GetIndex()), entity);
}
