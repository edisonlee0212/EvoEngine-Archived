#pragma once

namespace EvoEngine
{
    class Scene;
    class ILayer
    {
        std::weak_ptr<Scene> m_scene;
        friend class Application;
        virtual void OnCreate()
        {
        }
        virtual void OnDestroy()
        {
        }
        virtual void PreUpdate()
        {
        }
        virtual void FixedUpdate()
        {
        }
        virtual void Update()
        {
        }
        virtual void LateUpdate()
        {
        }
        virtual void OnInspect()
        {
        }

    public:
        [[nodiscard]] std::shared_ptr<Scene> GetScene() const;
    };
} // namespace EvoEngine