#pragma once

namespace EvoEngine
{
	class Scene;
	class EditorLayer;
	class ILayer
	{
		std::weak_ptr<Scene> m_scene;
		friend class Application;
		friend class EditorLayer;
		virtual void OnCreate() {}
		virtual void OnDestroy() {}
		virtual void PreUpdate() {}
		virtual void FixedUpdate() {}
		virtual void Update() {}
		virtual void LateUpdate() {}
		virtual void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) {}
	public:
		[[nodiscard]] std::shared_ptr<Scene> GetScene() const;
	};
} // namespace EvoEngine