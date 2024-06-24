#include "AnimationPlayer.hpp"
#include "Application.hpp"
#include "ClassRegistry.hpp"
#include "EditorLayer.hpp"
#include "MeshRenderer.hpp"
#include "PlayerController.hpp"
#include "Prefab.hpp"
#include "RenderLayer.hpp"
#include "Times.hpp"
#include "TransformGraph.hpp"
#include "WindowLayer.hpp"
#include "TextureBaking.hpp"
#include "PostProcessingStack.hpp"

using namespace evo_engine;
#pragma region Helpers

enum class DemoSetup {
  Empty,
  Rendering,
};
Entity LoadScene(const std::shared_ptr<Scene>& scene, const std::string& base_entity_name, bool add_spheres);
void SetupDemoScene(DemoSetup demo_setup, ApplicationInfo& application_info);
#pragma endregion

int main() {
  constexpr DemoSetup demo_setup = DemoSetup::Rendering;
  Application::PushLayer<WindowLayer>();
  Application::PushLayer<EditorLayer>();
  Application::PushLayer<RenderLayer>();

  ApplicationInfo application_info;
  SetupDemoScene(demo_setup, application_info);

  Application::Initialize(application_info);

  Application::Start();
  Application::Run();
  Application::Terminate();
  return 0;
}
#pragma region Helpers
Entity LoadScene(const std::shared_ptr<Scene>& scene, const std::string& base_entity_name, bool add_spheres) {
  auto base_entity = scene->CreateEntity(base_entity_name);

  if (add_spheres) {
#pragma region Create 9 spheres in different PBR properties
    const int amount = 5;
    const auto collection = scene->CreateEntity("Spheres");
    const auto spheres = scene->CreateEntities(amount * amount * amount, "Instance");

    for (int i = 0; i < amount; i++) {
      for (int j = 0; j < amount; j++) {
        for (int k = 0; k < amount; k++) {
          const float scale_factor = 0.03f;
          auto& sphere = spheres[i * amount * amount + j * amount + k];
          Transform transform;
          glm::vec3 position = glm::vec3(i + 0.5f - amount / 2.0f, j + 0.5f - amount / 2.0f, k + 0.5f - amount / 2.0f);
          position += glm::linearRand(glm::vec3(-0.5f), glm::vec3(0.5f)) * scale_factor;
          transform.SetPosition(position * 5.f * scale_factor);
          transform.SetScale(glm::vec3(4.0f * scale_factor));
          scene->SetDataComponent(sphere, transform);
          const auto mesh_renderer = scene->GetOrSetPrivateComponent<MeshRenderer>(sphere).lock();
          mesh_renderer->mesh = Resources::GetResource<Mesh>("PRIMITIVE_SPHERE");
          const auto material = ProjectManager::CreateTemporaryAsset<Material>();
          mesh_renderer->material = material;
          material->material_properties.roughness = static_cast<float>(i) / (amount - 1);
          material->material_properties.metallic = static_cast<float>(j) / (amount - 1);
          scene->SetParent(sphere, collection);
        }
      }
    }
    scene->SetParent(collection, base_entity);
    Transform physics_demo_transform;
    physics_demo_transform.SetPosition(glm::vec3(0.0f, 0.0f, -3.5f));
    physics_demo_transform.SetScale(glm::vec3(3.0f));
    scene->SetDataComponent(collection, physics_demo_transform);
#pragma endregion
  }

#pragma region Create ground
  auto ground = scene->CreateEntity("Ground");
  auto ground_mesh_renderer = scene->GetOrSetPrivateComponent<MeshRenderer>(ground).lock();
  auto ground_mat = ProjectManager::CreateTemporaryAsset<Material>();

  ground_mesh_renderer->material = ground_mat;
  ground_mesh_renderer->mesh = Resources::GetResource<Mesh>("PRIMITIVE_CUBE");
  Transform ground_transform;
  ground_transform.SetValue(glm::vec3(0, -2.05, -0), glm::vec3(0), glm::vec3(30, 1, 60));
  scene->SetDataComponent(ground, ground_transform);
  scene->SetParent(ground, base_entity);
#pragma endregion
#pragma region Load models and display
  const auto sponza =
      std::dynamic_pointer_cast<Prefab>(ProjectManager::GetOrCreateAsset("Models/Sponza_FBX/Sponza.fbx"));
  const auto sponza_entity = sponza->ToEntity(scene);
  Transform sponza_transform;
  sponza_transform.SetValue(glm::vec3(0, -1.5, -6), glm::radians(glm::vec3(0, -90, 0)), glm::vec3(0.01));
  scene->SetDataComponent(sponza_entity, sponza_transform);
  scene->SetParent(sponza_entity, base_entity);

  auto title = std::dynamic_pointer_cast<Prefab>(ProjectManager::GetOrCreateAsset("Models/UniEngine.obj"));
  auto title_entity = title->ToEntity(scene);
  scene->SetEntityName(title_entity, "Title");
  Transform title_transform;
  title_transform.SetValue(glm::vec3(0.35, 7, -16), glm::radians(glm::vec3(0, 0, 0)), glm::vec3(0.005));
  scene->SetDataComponent(title_entity, title_transform);
  scene->SetParent(title_entity, base_entity);

  auto title_material =
      scene->GetOrSetPrivateComponent<MeshRenderer>(scene->GetChildren(scene->GetChildren(title_entity)[0])[0])
          .lock()
          ->material.Get<Material>();
  title_material->material_properties.emission = 4;
  title_material->material_properties.albedo_color = glm::vec3(1, 0.2, 0.5);

  auto dancing_storm_trooper = std::dynamic_pointer_cast<Prefab>(
      ProjectManager::GetOrCreateAsset("Models/dancing-stormtrooper/silly_dancing.fbx"));
  auto dancing_storm_trooper_entity = dancing_storm_trooper->ToEntity(scene);
  const auto dancing_storm_trooper_animation_player =
      scene->GetOrSetPrivateComponent<AnimationPlayer>(dancing_storm_trooper_entity).lock();
  dancing_storm_trooper_animation_player->auto_play = true;
  dancing_storm_trooper_animation_player->auto_play_speed = 30;
  scene->SetEntityName(dancing_storm_trooper_entity, "StormTrooper");
  Transform dancing_storm_trooper_transform;
  dancing_storm_trooper_transform.SetValue(glm::vec3(1.2, -1.5, 0), glm::vec3(0), glm::vec3(0.4));
  scene->SetDataComponent(dancing_storm_trooper_entity, dancing_storm_trooper_transform);
  scene->SetParent(dancing_storm_trooper_entity, base_entity);

  const auto capoeira = std::dynamic_pointer_cast<Prefab>(ProjectManager::GetOrCreateAsset("Models/Capoeira.fbx"));
  const auto capoeira_entity = capoeira->ToEntity(scene);
  const auto capoeira_animation_player = scene->GetOrSetPrivateComponent<AnimationPlayer>(capoeira_entity).lock();
  capoeira_animation_player->auto_play = true;
  capoeira_animation_player->auto_play_speed = 60;
  scene->SetEntityName(capoeira_entity, "Capoeira");
  Transform capoeira_transform;
  capoeira_transform.SetValue(glm::vec3(0.5, 2.7, -18), glm::vec3(0), glm::vec3(0.02));
  scene->SetDataComponent(capoeira_entity, capoeira_transform);
  auto capoeira_body_material =
      scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(scene->GetChildren(scene->GetChildren(capoeira_entity)[1])[0])
          .lock()
          ->material.Get<Material>();
  capoeira_body_material->material_properties.albedo_color = glm::vec3(0, 1, 1);
  capoeira_body_material->material_properties.metallic = 1;
  capoeira_body_material->material_properties.roughness = 0;
  auto capoeira_joints_material =
      scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(scene->GetChildren(scene->GetChildren(capoeira_entity)[0])[0])
          .lock()
          ->material.Get<Material>();
  capoeira_joints_material->material_properties.albedo_color = glm::vec3(0.3, 1.0, 0.5);
  capoeira_joints_material->material_properties.metallic = 1;
  capoeira_joints_material->material_properties.roughness = 0;
  capoeira_joints_material->material_properties.emission = 6;
  scene->SetParent(capoeira_entity, base_entity);

#pragma endregion

  return base_entity;
}

void SetupDemoScene(const DemoSetup demo_setup, ApplicationInfo& application_info) {

  std::filesystem::path resource_folder_path("../../../../../Resources");
  if (!std::filesystem::exists(resource_folder_path)) {
    resource_folder_path = "../../../../Resources";
  }
  if (!std::filesystem::exists(resource_folder_path)) {
    resource_folder_path = "../../../Resources";
  }
  if (!std::filesystem::exists(resource_folder_path)) {
    resource_folder_path = "../../Resources";
  }
  if (!std::filesystem::exists(resource_folder_path)) {
    resource_folder_path = "../../Resources";
  }
  if (!std::filesystem::exists(resource_folder_path)) {
    resource_folder_path = "../Resources";
  }
#pragma region Demo scene setup
  if (demo_setup != DemoSetup::Empty && std::filesystem::exists(resource_folder_path)) {
    for (const auto i : std::filesystem::recursive_directory_iterator(resource_folder_path)) {
      if (i.is_directory())
        continue;
      if (i.path().extension().string() == ".evescene" || i.path().extension().string() == ".evefilemeta" ||
          i.path().extension().string() == ".eveproj" || i.path().extension().string() == ".evefoldermeta") {
        std::filesystem::remove(i.path());
      }
    }
    for (const auto i : std::filesystem::recursive_directory_iterator(resource_folder_path)) {
      if (i.is_directory())
        continue;
      if (i.path().extension().string() == ".uescene" || i.path().extension().string() == ".umeta" ||
          i.path().extension().string() == ".ueproj" || i.path().extension().string() == ".ufmeta") {
        std::filesystem::remove(i.path());
      }
    }
  }
  switch (demo_setup) {
    case DemoSetup::Rendering: {
      application_info.application_name = "Rendering Demo";
      application_info.project_path = resource_folder_path / "Example Projects/Rendering/Rendering.eveproj";
      ProjectManager::SetActionAfterNewScene([&](const std::shared_ptr<Scene>& scene) {
        scene->environment.ambient_light_intensity = 0.1f;
#pragma region Set main camera to correct position and rotation
        const auto main_camera = scene->main_camera.Get<Camera>();
        main_camera->Resize({640, 480});
        main_camera->post_processing_stack = ProjectManager::CreateTemporaryAsset<PostProcessingStack>();
        const auto main_camera_entity = main_camera->GetOwner();
        auto main_camera_transform = scene->GetDataComponent<Transform>(main_camera_entity);
        main_camera_transform.SetPosition(glm::vec3(0, 0, 4));
        scene->SetDataComponent(main_camera_entity, main_camera_transform);
        auto camera = scene->GetOrSetPrivateComponent<Camera>(main_camera_entity).lock();
        scene->GetOrSetPrivateComponent<PlayerController>(main_camera_entity);
#pragma endregion

        LoadScene(scene, "Rendering Demo", true);

#pragma region Dynamic Lighting
        const auto dir_light_entity = scene->CreateEntity("Directional Light");
        const auto dir_light = scene->GetOrSetPrivateComponent<DirectionalLight>(dir_light_entity).lock();
        dir_light->diffuse_brightness = 1.0f;
        dir_light->light_size = 0.2f;
        const auto point_light_right_entity = scene->CreateEntity("Left Point Light");
        const auto point_light_right_renderer =
            scene->GetOrSetPrivateComponent<MeshRenderer>(point_light_right_entity).lock();
        const auto point_light_right_material = ProjectManager::CreateTemporaryAsset<Material>();
        point_light_right_renderer->material.Set<Material>(point_light_right_material);
        point_light_right_material->material_properties.albedo_color = glm::vec3(1.0, 0.8, 0.0);
        point_light_right_material->material_properties.emission = 100.0f;
        point_light_right_renderer->mesh = Resources::GetResource<Mesh>("PRIMITIVE_SPHERE");
        const auto point_light_right = scene->GetOrSetPrivateComponent<PointLight>(point_light_right_entity).lock();
        point_light_right->diffuse_brightness = 4;
        point_light_right->light_size = 0.02f;
        point_light_right->linear = 0.5f;
        point_light_right->quadratic = 0.1f;
        point_light_right->diffuse = glm::vec3(1.0, 0.8, 0.0);

        Transform dir_light_transform;
        dir_light_transform.SetEulerRotation(glm::radians(glm::vec3(105.0f, 0, 0.0f)));
        scene->SetDataComponent(dir_light_entity, dir_light_transform);
        Transform point_light_right_transform;
        point_light_right_transform.SetPosition(glm::vec3(4, 1.2, -5));
        point_light_right_transform.SetScale({1.0f, 1.0f, 1.0f});
        scene->SetDataComponent(point_light_right_entity, point_light_right_transform);

        Application::RegisterUpdateFunction([=]() {
          static bool last_frame_playing = false;
          if (!Application::IsPlaying()) {
            last_frame_playing = Application::IsPlaying();
            return;
          }
          const auto current_scene = Application::GetActiveScene();
          static float start_time;
          if (!last_frame_playing)
            start_time = Times::Now();
          const float current_time = Times::Now() - start_time;
          const float cos_time = glm::cos(current_time / 5.0f);
          Transform dir_light_transform;
          dir_light_transform.SetEulerRotation(glm::radians(glm::vec3(105.0f, current_time * 20, 0.0f)));
          current_scene->SetDataComponent(dir_light_entity, dir_light_transform);

          Transform point_light_right_transform;
          point_light_right_transform.SetPosition(glm::vec3(4, 1.2, cos_time * 5 - 5));
          point_light_right_transform.SetScale({1.0f, 1.0f, 1.0f});
          point_light_right_transform.SetPosition(glm::vec3(4, 1.2, cos_time * 5 - 5));
          current_scene->SetDataComponent(point_light_right_entity, point_light_right_transform);

          last_frame_playing = Application::IsPlaying();
        });
#pragma endregion
      });
    } break;
    default: {
    } break;
  }
#pragma endregion
}

#pragma endregion
