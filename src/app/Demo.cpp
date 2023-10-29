#include "AnimationPlayer.hpp"
#include "Application.hpp"
#include "ClassRegistry.hpp"
#include "WindowLayer.hpp"
#include "RenderLayer.hpp"
#include "EditorLayer.hpp"
#include "MeshRenderer.hpp"
#include "PlayerController.hpp"
#include "Prefab.hpp"
#include "Times.hpp"


#include "PostProcessingStack.hpp"

using namespace EvoEngine;

int main() {
	
	Application::PushLayer<WindowLayer>();
	Application::PushLayer<EditorLayer>();
	Application::PushLayer<RenderLayer>();
	const ApplicationInfo applicationInfo {};
	
	Application::Initialize(applicationInfo);
	Application::Start();
	Application::Run();
	Application::Terminate();
	return 0;
}
