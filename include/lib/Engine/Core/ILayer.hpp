#pragma once
#include "Input.hpp"

namespace EvoEngine
{
	class Scene;
	class EditorLayer;
	class ILayer
	{
		std::weak_ptr<Scene> m_scene;
		std::weak_ptr<ILayer> m_subsequentLayer;
		friend class Application;
		friend class EditorLayer;
		friend class Input;
		virtual void OnCreate() {}
		virtual void OnDestroy() {}
		virtual void PreUpdate() {}
		virtual void FixedUpdate() {}
		virtual void Update() {}
		virtual void LateUpdate() {}
		virtual void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) {}
		virtual void OnInputEvent(const InputEvent& inputEvent);
	public:
		[[nodiscard]] std::shared_ptr<Scene> GetScene() const;
	};
} // namespace EvoEngine