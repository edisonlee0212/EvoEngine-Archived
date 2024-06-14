//
// Created by Bosheng Li on 4/22/2022.
//
#include "ILayer.hpp"
#include "Scene.hpp"

using namespace evo_engine;

void ILayer::OnInputEvent(const InputEvent& inputEvent)
{
	if(!m_subsequentLayer.expired())
	{
		m_subsequentLayer.lock()->OnInputEvent(inputEvent);
	}
}

std::shared_ptr<Scene> ILayer::GetScene() const
{
	return m_scene.lock();
}
