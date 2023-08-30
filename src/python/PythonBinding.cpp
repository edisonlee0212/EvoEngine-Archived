#include "pybind11/pybind11.h"
#include "AnimationPlayer.hpp"
#include "Application.hpp"
#include "ClassRegistry.hpp"
#include "WindowLayer.hpp"
#include "RenderLayer.hpp"
#include "EditorLayer.hpp"
#include "MeshRenderer.hpp"
#include "PlayerController.hpp"
#include "Prefab.hpp"
#include "Time.hpp"
#include "PhysicsLayer.hpp"
#include "PostProcessingStack.hpp"
using namespace EvoEngine;
namespace py = pybind11;

void StartEditor(const std::string& projectPath) {
	Application::PushLayer<WindowLayer>();
	Application::PushLayer<EditorLayer>();
	Application::PushLayer<RenderLayer>();

	ApplicationInfo applicationInfo;
	applicationInfo.m_projectPath = projectPath;
	Application::Initialize(applicationInfo);
	Application::Start();

	Application::Terminate();
}

void StartEmptyEditor() {
	Application::PushLayer<WindowLayer>();
	Application::PushLayer<EditorLayer>();
	Application::PushLayer<RenderLayer>();

	ApplicationInfo applicationInfo;
	Application::Initialize(applicationInfo);
	Application::Start();

	Application::Terminate();
}

void RenderScene(const std::string& projectPath, const int resolutionX, const int resolutionY, const std::string& outputPath)
{
	if (!std::filesystem::is_regular_file(projectPath))
	{
		EVOENGINE_ERROR("Project doesn't exist!");
		return;
	}
	if (resolutionX <= 0 || resolutionY <= 0)
	{
		EVOENGINE_ERROR("Resolution error!");
		return;
	}
	Application::PushLayer<RenderLayer>();

	ApplicationInfo applicationInfo;
	applicationInfo.m_projectPath = std::filesystem::path(projectPath);
	EVOENGINE_LOG("Loading project at " + projectPath);
	Application::Initialize(applicationInfo);
	EVOENGINE_LOG("Loaded project at " + projectPath);
	Application::Start();
	auto scene = Application::GetActiveScene();
	const auto mainCamera = scene->m_mainCamera.Get<Camera>();
	mainCamera->Resize({ resolutionX, resolutionY });
	Application::Loop();
	Graphics::WaitForDeviceIdle();
	
	
	mainCamera->GetRenderTexture()->StoreToPng(outputPath);
	EVOENGINE_LOG("Exported image to " + outputPath);
	Application::Terminate();
}

void RenderMesh(const std::string& meshPath, const int resolutionX, const int resolutionY, const std::string& outputPath)
{
	if (!std::filesystem::is_regular_file(meshPath))
	{
		EVOENGINE_ERROR("Project doesn't exist!");
		return;
	}
	if (resolutionX <= 0 || resolutionY <= 0)
	{
		EVOENGINE_ERROR("Resolution error!");
		return;
	}

}

PYBIND11_MODULE(pyevoengine, m) {
	m.doc() = "EvoEngine"; // optional module docstring
	m.def("start_editor", &StartEditor, "Start editor with target project");
	m.def("start_empty_editor", &StartEmptyEditor, "Start editor without project");
	m.def("render_scene", &RenderScene, "Render the default scene for given project");
}