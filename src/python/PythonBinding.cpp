#include "pybind11/pybind11.h"
#include "pybind11/stl/filesystem.h"
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
#include "ProjectManager.hpp"
#include "Scene.hpp"

using namespace evo_engine;

namespace py = pybind11;

#pragma region Helpers
void PushWindowLayer() {
	Application::PushLayer<WindowLayer>();
}
void PushEditorLayer() {
	Application::PushLayer<EditorLayer>();
}
void PushRenderLayer() {
	Application::PushLayer<RenderLayer>();
}

Entity CreateDynamicCube(
	const float& mass,
	const glm::vec3& color,
	const glm::vec3& position,
	const glm::vec3& rotation,
	const glm::vec3& scale,
	const std::string& name);

Entity CreateSolidCube(
	const float& mass,
	const glm::vec3& color,
	const glm::vec3& position,
	const glm::vec3& rotation,
	const glm::vec3& scale,
	const std::string& name);

Entity CreateCube(
	const glm::vec3& color,
	const glm::vec3& position,
	const glm::vec3& rotation,
	const glm::vec3& scale,
	const std::string& name);

Entity CreateDynamicSphere(
	const float& mass,
	const glm::vec3& color,
	const glm::vec3& position,
	const glm::vec3& rotation,
	const float& scale,
	const std::string& name);

Entity CreateSolidSphere(
	const float& mass,
	const glm::vec3& color,
	const glm::vec3& position,
	const glm::vec3& rotation,
	const float& scale,
	const std::string& name);

Entity CreateSphere(
	const glm::vec3& color,
	const glm::vec3& position,
	const glm::vec3& rotation,
	const float& scale,
	const std::string& name);

enum class DemoSetup
{
	Empty,
	Rendering,
	Galaxy,
	Planets
};
Entity LoadScene(const std::shared_ptr<Scene>& scene, const std::string& baseEntityName, bool addSpheres);
void SetupDemo(DemoSetup demoSetup);
void SetupDemoScene(DemoSetup demoSetup, ApplicationInfo& applicationInfo, bool enablePhysics);
#pragma endregion

void CreateRenderingDemo()
{
	DemoSetup demoSetup = DemoSetup::Rendering;
	SetupDemo(demoSetup);
}

void InitializePlanetsDemo()
{
	DemoSetup demoSetup = DemoSetup::Planets;
	SetupDemo(demoSetup);
}

void RegisterLayers(bool enableWindowLayer, bool enableEditorLayer)
{
	if (enableWindowLayer) Application::PushLayer<WindowLayer>();
	if (enableWindowLayer && enableEditorLayer) Application::PushLayer<EditorLayer>();
	Application::PushLayer<RenderLayer>();
}

void StartProjectWindowless(const std::filesystem::path& projectPath)
{
	if (std::filesystem::path(projectPath).extension().string() != ".eveproj") {
		EVOENGINE_ERROR("Project path doesn't point to a evo_engine project!");
		return;
	}
	RegisterLayers(false, false);
	ApplicationInfo applicationInfo{};
	applicationInfo.m_projectPath = projectPath;
	Application::Initialize(applicationInfo);
	Application::Start();
}

void StartProjectWithEditor(const std::filesystem::path& projectPath)
{
	if (!projectPath.empty()) {
		if (std::filesystem::path(projectPath).extension().string() != ".eveproj") {
			EVOENGINE_ERROR("Project path doesn't point to a evo_engine project!");
			return;
		}
	}
	RegisterLayers(false, false);
	ApplicationInfo applicationInfo{};
	applicationInfo.m_projectPath = projectPath;
	Application::Initialize(applicationInfo);
	Application::Start();
	Application::Run();
}

void CaptureActiveScene(const int resolutionX, const int resolutionY, const std::string& outputPath)
{
	if (resolutionX <= 0 || resolutionY <= 0)
	{
		EVOENGINE_ERROR("Resolution error!");
		return;
	}

	const auto scene = Application::GetActiveScene();
	if (!scene)
	{
		EVOENGINE_ERROR("No active scene!");
		return;
	}
	const auto mainCamera = scene->m_mainCamera.Get<Camera>();
	if (!mainCamera)
	{
		EVOENGINE_ERROR("No main camera in scene!");
		return;
	}
	mainCamera->Resize({ resolutionX, resolutionY });
	Application::Loop();
	mainCamera->GetRenderTexture()->StoreToPng(outputPath);
	EVOENGINE_LOG("Exported image to " + outputPath);
}


PYBIND11_MODULE(pyevoengine, m) {
	py::class_<Entity>(m, "Entity")
		.def("get_index", &Entity::GetIndex)
		.def("get_version", &Entity::GetVersion);

	
	py::class_<Scene>(m, "Scene")
		.def("create_entity", static_cast<Entity(Scene::*)(const std::string&)>(&Scene::CreateEntity))
		.def("delete_entity", &Scene::DeleteEntity)
		.def("save", &Scene::Save);
	py::class_<ApplicationInfo>(m, "ApplicationInfo")
		.def(py::init<>())
		.def_readwrite("m_projectPath", &ApplicationInfo::m_projectPath)
		.def_readwrite("m_applicationName", &ApplicationInfo::m_applicationName)
		.def_readwrite("m_enableDocking", &ApplicationInfo::m_enableDocking)
		.def_readwrite("m_enableViewport", &ApplicationInfo::m_enableViewport)
		.def_readwrite("m_fullScreen", &ApplicationInfo::m_fullScreen);

	py::class_<Application>(m, "Application")
		.def_static("initialize", &Application::Initialize)
		.def_static("start", &Application::Start)
		.def_static("run", &Application::Run)
		.def_static("loop", &Application::Loop)
		.def_static("terminate", &Application::Terminate)
		.def_static("get_active_scene", &Application::GetActiveScene);

	py::class_<ProjectManager>(m, "ProjectManager")
		.def_static("GetOrCreateProject", &ProjectManager::GetOrCreateProject)
		.def_static("SaveProject", &ProjectManager::SaveProject);
	m.doc() = "evo_engine"; // optional module docstring

	m.def("create_rendering_demo", &CreateRenderingDemo, "Create Rendering Demo");

	m.def("register_layers", &RegisterLayers, "RegisterLayers");
	m.def("start_project_windowless", &StartProjectWindowless, "StartProjectWindowless");
	m.def("start_project_with_editor", &StartProjectWithEditor, "StartProjectWithEditor");

	m.def("capture_active_scene", &CaptureActiveScene, "CaptureActiveScene");
}



void SetupDemo(DemoSetup demoSetup)
{
	Application::PushLayer<RenderLayer>();
	const ApplicationInfo applicationInfo{};
	Application::Initialize(applicationInfo);
	Application::Start();
}
