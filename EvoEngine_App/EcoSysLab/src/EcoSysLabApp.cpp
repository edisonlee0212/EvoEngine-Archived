// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>

#ifdef BUILD_WITH_RAYTRACER
#  include <CUDAModule.hpp>
#  include <RayTracerLayer.hpp>
#endif

#include "Times.hpp"

#include "ClassRegistry.hpp"
#include "Climate.hpp"
#include "EcoSysLabLayer.hpp"
#include "ForestDescriptor.hpp"
#include "HeightField.hpp"
#include "ObjectRotator.hpp"
#include "ParticlePhysics2DDemo.hpp"
#include "Physics2DDemo.hpp"
#include "ProjectManager.hpp"
#include "RadialBoundingVolume.hpp"
#include "Soil.hpp"
#include "SorghumLayer.hpp"
#include "Tree.hpp"
#include "TreeModel.hpp"
#include "TreePointCloudScanner.hpp"
#include "TreeStructor.hpp"
#include "WindowLayer.hpp"
using namespace eco_sys_lab;

void EngineSetup();

int main() {
  std::filesystem::path resourceFolderPath("../../../../../Resources");
  if (!std::filesystem::exists(resourceFolderPath)) {
    resourceFolderPath = "../../../../Resources";
  }
  if (!std::filesystem::exists(resourceFolderPath)) {
    resourceFolderPath = "../../../Resources";
  }
  if (!std::filesystem::exists(resourceFolderPath)) {
    resourceFolderPath = "../../Resources";
  }
  if (!std::filesystem::exists(resourceFolderPath)) {
    resourceFolderPath = "../Resources";
  }
  if (std::filesystem::exists(resourceFolderPath)) {
    for (auto i : std::filesystem::recursive_directory_iterator(resourceFolderPath)) {
      if (i.is_directory())
        continue;
      auto oldPath = i.path();
      auto newPath = i.path();
      bool remove = false;
      if (i.path().extension().string() == ".uescene") {
        newPath.replace_extension(".evescene");
        remove = true;
      }
      if (i.path().extension().string() == ".umeta") {
        newPath.replace_extension(".evefilemeta");
        remove = true;
      }
      if (i.path().extension().string() == ".ueproj") {
        newPath.replace_extension(".eveproj");
        remove = true;
      }
      if (i.path().extension().string() == ".ufmeta") {
        newPath.replace_extension(".evefoldermeta");
        remove = true;
      }
      if (remove) {
        std::filesystem::copy(oldPath, newPath);
        std::filesystem::remove(oldPath);
      }
    }
  }

  EngineSetup();

  Application::PushLayer<WindowLayer>();
  Application::PushLayer<EditorLayer>();
  Application::PushLayer<RenderLayer>();
#ifdef BUILD_WITH_RAYTRACER
  Application::PushLayer<RayTracerLayer>();
#endif

#ifdef BUILD_WITH_PHYSICS
  Application::PushLayer<PhysicsLayer>();
#endif
  Application::PushLayer<EcoSysLabLayer>();
  ClassRegistry::RegisterPrivateComponent<ObjectRotator>("ObjectRotator");
  ClassRegistry::RegisterPrivateComponent<Physics2DDemo>("Physics2DDemo");
  ClassRegistry::RegisterPrivateComponent<ParticlePhysics2DDemo>("ParticlePhysics2DDemo");

  ApplicationInfo application_configs;
  application_configs.application_name = "EcoSysLab";
  application_configs.project_path = std::filesystem::absolute(resourceFolderPath / "EcoSysLabProject" / "test.eveproj");
  Application::Initialize(application_configs);

#ifdef BUILD_WITH_RAYTRACER

  auto ray_tracer_layer = Application::GetLayer<RayTracerLayer>();
#endif
#ifdef BUILD_WITH_PHYSICS
  Application::GetActiveScene()->GetOrCreateSystem<PhysicsSystem>(1);
#endif
  // adjust default camera speed
  const auto editor_layer = Application::GetLayer<EditorLayer>();
  editor_layer->velocity = 2.f;
  editor_layer->default_scene_camera_position = glm::vec3(1.124, 0.218, 14.089);
  // override default scene camera position etc.
  editor_layer->show_camera_window = false;
  editor_layer->show_scene_window = true;
  editor_layer->show_entity_explorer_window = true;
  editor_layer->show_entity_inspector_window = true;
  const auto render_layer = Application::GetLayer<RenderLayer>();
#pragma region Engine Loop
  Application::Start();
  Application::Run();
#pragma endregion
  Application::Terminate();
}

void EngineSetup() {
  ProjectManager::SetActionAfterNewScene([=](const std::shared_ptr<Scene>& scene) {
#pragma region Engine Setup
    Transform transform;
    transform.SetEulerRotation(glm::radians(glm::vec3(150, 30, 0)));
#pragma region Preparations
    Times::SetTimeStep(0.016f);
    transform = Transform();
    transform.SetPosition(glm::vec3(0, 2, 35));
    transform.SetEulerRotation(glm::radians(glm::vec3(15, 0, 0)));
    if (const auto main_camera = Application::GetActiveScene()->main_camera.Get<Camera>()) {
      scene->SetDataComponent(main_camera->GetOwner(), transform);
      main_camera->use_clear_color = true;
      main_camera->clear_color = glm::vec3(0.5f);
    }
#pragma endregion
#pragma endregion
  });
}
