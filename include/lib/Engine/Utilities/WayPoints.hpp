#pragma once
#include "IPrivateComponent.hpp"
namespace evo_engine
{
	class WayPoints : public IPrivateComponent
	{
	public:
		enum class Mode
		{
			FixedTime,
			FixedVelocity
		} m_mode = Mode::FixedTime;

		float m_speed = 1.0f;
		std::vector<EntityRef> m_entities;
		bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		void OnCreate() override;
		void OnDestroy() override;
		void Update() override;
	};
}