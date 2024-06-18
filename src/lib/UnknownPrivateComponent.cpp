//
// Created by lllll on 8/23/2021.
//

#include "UnknownPrivateComponent.hpp"
using namespace EvoEngine;
bool UnknownPrivateComponent::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	ImGui::Text(std::string("Type: " + m_originalTypeName).c_str());

	return false;
}