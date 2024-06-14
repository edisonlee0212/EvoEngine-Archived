#pragma once
#include "Entities.hpp"
#include "Transform.hpp"
#include "ILayer.hpp"
#include "EntityMetadata.hpp"
namespace evo_engine
{
    class TransformGraph : public ISingleton<TransformGraph>
    {
        friend class PhysicsSystem;
        friend class Application;
        EntityQuery m_transformQuery;
        bool m_physicsSystemOverride = false;
        static void CalculateTransformGraph(const std::shared_ptr<Scene>& scene, const std::vector<EntityMetadata>& entityInfos, const GlobalTransform& parentGlobalTransform, const Entity& parent);
        static void Initialize();
    public:
        static void CalculateTransformGraphForDescendants(const std::shared_ptr<Scene>& scene, const Entity& entity);
        static void CalculateTransformGraphs(const std::shared_ptr<Scene>& scene, bool checkStatic = true);
    };
} // namespace UniEngine