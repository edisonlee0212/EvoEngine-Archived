//#include "ProfilerLayer.hpp"
//#include "RigidBody.hpp"
#include "TransformGraph.hpp"
#include "Application.hpp"
#include "Scene.hpp"
#include "ClassRegistry.hpp"
#include "Jobs.hpp"
using namespace EvoEngine;

DataComponentRegistration<Transform> TransformRegistry("Transform");
DataComponentRegistration<GlobalTransform> GlobalTransformRegistry("GlobalTransform");
DataComponentRegistration<TransformUpdateStatus> TransformUpdateStatusRegistry("TransformUpdateStatus");


void TransformGraph::Initialize()
{
    auto& transformGraph = GetInstance();
    transformGraph.m_transformQuery = Entities::CreateEntityQuery();
    Entities::SetEntityQueryAllFilters(transformGraph.m_transformQuery, Transform(), GlobalTransform());
}

void TransformGraph::CalculateTransformGraph(
    const std::shared_ptr<Scene>& scene,
    const std::vector<EntityMetadata>& entityInfos,
    const GlobalTransform& parentGlobalTransform,
    const Entity &parent)
{
	const auto& entityInfo = entityInfos.at(parent.GetIndex());
    for (const auto& entity : entityInfo.m_children)
    {
        auto* transformStatus = reinterpret_cast<TransformUpdateStatus*>(
            scene->GetDataComponentPointer(entity.GetIndex(), typeid(TransformUpdateStatus).hash_code()));
        GlobalTransform ltw;
        if (transformStatus->m_globalTransformModified)
        {
            ltw = scene->GetDataComponent<GlobalTransform>(entity.GetIndex());
            reinterpret_cast<Transform*>(
                scene->GetDataComponentPointer(entity.GetIndex(), typeid(Transform).hash_code()))
                ->m_value = glm::inverse(parentGlobalTransform.m_value) * ltw.m_value;
            transformStatus->m_globalTransformModified = false;
        }
        else
        {
            auto ltp = scene->GetDataComponent<Transform>(entity.GetIndex());
            ltw.m_value = parentGlobalTransform.m_value * ltp.m_value;
            *reinterpret_cast<GlobalTransform*>(
                scene->GetDataComponentPointer(entity.GetIndex(), typeid(GlobalTransform).hash_code())) = ltw;
        }
        transformStatus->m_transformModified = false;
        CalculateTransformGraph(scene, entityInfos, ltw, entity);
    }
}
void TransformGraph::CalculateTransformGraphs(const std::shared_ptr<Scene>& scene, bool checkStatic)
{
    if (!scene)
        return;
    auto& transformGraph = GetInstance();
    const auto& entityInfos = scene->m_sceneDataStorage.m_entityMetadataList;
    //ProfilerLayer::StartEvent("TransformManager");
    scene->ForEach<Transform, GlobalTransform, TransformUpdateStatus>(
        Jobs::Workers(),
        transformGraph.m_transformQuery,
        [&](int i,
            Entity entity,
            Transform& transform,
            GlobalTransform& globalTransform,
            TransformUpdateStatus& transformStatus) {
	        const EntityMetadata& entityInfo = scene->m_sceneDataStorage.m_entityMetadataList.at(entity.GetIndex());
                if (entityInfo.m_parent.GetIndex() != 0)
                    return;
                if (checkStatic && entityInfo.m_static)
                    return;
                if (transformStatus.m_globalTransformModified)
                {
                    transform.m_value = globalTransform.m_value;
                    transformStatus.m_globalTransformModified = false;
                }
                else {
                    globalTransform.m_value = transform.m_value;
                }
                transformStatus.m_transformModified = false;
                transformGraph.CalculateTransformGraph(scene, entityInfos, globalTransform, entity);
        },
        false);
    transformGraph.m_physicsSystemOverride = false;
    //ProfilerLayer::EndEvent("TransformManager");
}
void TransformGraph::CalculateTransformGraphForDescendents(const std::shared_ptr<Scene>& scene, const Entity& entity)
{
    if (!scene)
        return;
    auto& transformGraph = GetInstance();
    const auto& entityInfos = scene->m_sceneDataStorage.m_entityMetadataList;
    transformGraph.CalculateTransformGraph(
        scene, entityInfos, scene->GetDataComponent<GlobalTransform>(entity.GetIndex()), entity);
}
