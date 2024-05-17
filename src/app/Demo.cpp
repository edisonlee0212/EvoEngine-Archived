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
#include "TransformGraph.hpp"


#include "PostProcessingStack.hpp"



using namespace EvoEngine;
#pragma region Helpers

enum class DemoSetup
{
	Empty,
	Rendering,
};
Entity LoadScene(const std::shared_ptr<Scene>& scene, const std::string& baseEntityName, bool addSpheres);
void SetupDemoScene(DemoSetup demoSetup, ApplicationInfo& applicationInfo);
#pragma endregion

int main() {
	DemoSetup demoSetup = DemoSetup::Empty;
	Application::PushLayer<WindowLayer>();
	Application::PushLayer<EditorLayer>();
	Application::PushLayer<RenderLayer>();

	ApplicationInfo applicationInfo;
	SetupDemoScene(demoSetup, applicationInfo);

	Application::Initialize(applicationInfo);

	Application::Start();
	Application::Run();
	Application::Terminate();
	return 0;
}
#pragma region Helpers
Entity LoadScene(const std::shared_ptr<Scene>& scene, const std::string& baseEntityName, bool addSpheres)
{
	auto baseEntity = scene->CreateEntity(baseEntityName);

	if (addSpheres)
	{
#pragma region Create 9 spheres in different PBR properties
		const int amount = 5;
		const float scaleFactor = 0.03f;
		const auto collection = scene->CreateEntity("Spheres");
		const auto spheres = scene->CreateEntities(amount * amount * amount, "Instance");

		for (int i = 0; i < amount; i++)
		{
			for (int j = 0; j < amount; j++)
			{
				for (int k = 0; k < amount; k++)
				{
					auto& sphere = spheres[i * amount * amount + j * amount + k];
					Transform transform;
					glm::vec3 position = glm::vec3(i + 0.5f - amount / 2.0f, j + 0.5f - amount / 2.0f, k + 0.5f - amount / 2.0f);
					position += glm::linearRand(glm::vec3(-0.5f), glm::vec3(0.5f)) * scaleFactor;
					transform.SetPosition(position * 5.f * scaleFactor);
					transform.SetScale(glm::vec3(4.0f * scaleFactor));
					scene->SetDataComponent(sphere, transform);
					const auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(sphere).lock();
					meshRenderer->m_mesh = Resources::GetResource<Mesh>("PRIMITIVE_SPHERE");
					const auto material = ProjectManager::CreateTemporaryAsset<Material>();
					meshRenderer->m_material = material;
					material->m_materialProperties.m_roughness = static_cast<float>(i) / (amount - 1);
					material->m_materialProperties.m_metallic = static_cast<float>(j) / (amount - 1);
					scene->SetParent(sphere, collection);
				}
			}
		}
		scene->SetParent(collection, baseEntity);
		Transform physicsDemoTransform;
		physicsDemoTransform.SetPosition(glm::vec3(0.0f, 0.0f, -3.5f));
		physicsDemoTransform.SetScale(glm::vec3(3.0f));
		scene->SetDataComponent(collection, physicsDemoTransform);
#pragma endregion
	}

#pragma region Create ground
	auto ground = scene->CreateEntity("Ground");
	auto groundMeshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(ground).lock();
	auto groundMat = ProjectManager::CreateTemporaryAsset<Material>();

	groundMeshRenderer->m_material = groundMat;
	groundMeshRenderer->m_mesh = Resources::GetResource<Mesh>("PRIMITIVE_CUBE");
	Transform groundTransform;
	groundTransform.SetValue(glm::vec3(0, -2.05, -0), glm::vec3(0), glm::vec3(30, 1, 60));
	scene->SetDataComponent(ground, groundTransform);
	scene->SetParent(ground, baseEntity);
#pragma endregion
#pragma region Load models and display
	const auto sponza = std::dynamic_pointer_cast<Prefab>(ProjectManager::GetOrCreateAsset("Models/Sponza_FBX/Sponza.fbx"));
	const auto sponzaEntity = sponza->ToEntity(scene);
	Transform sponzaTransform;
	sponzaTransform.SetValue(glm::vec3(0, -1.5, -6), glm::radians(glm::vec3(0, -90, 0)), glm::vec3(0.01));
	scene->SetDataComponent(sponzaEntity, sponzaTransform);
	scene->SetParent(sponzaEntity, baseEntity);

	auto title = std::dynamic_pointer_cast<Prefab>(ProjectManager::GetOrCreateAsset("Models/UniEngine.obj"));
	auto titleEntity = title->ToEntity(scene);
	scene->SetEntityName(titleEntity, "Title");
	Transform titleTransform;
	titleTransform.SetValue(glm::vec3(0.35, 7, -16), glm::radians(glm::vec3(0, 0, 0)), glm::vec3(0.005));
	scene->SetDataComponent(titleEntity, titleTransform);
	scene->SetParent(titleEntity, baseEntity);

	auto titleMaterial =
		scene->GetOrSetPrivateComponent<MeshRenderer>(scene->GetChildren(scene->GetChildren(titleEntity)[0])[0])
		.lock()
		->m_material.Get<Material>();
	titleMaterial->m_materialProperties.m_emission = 4;
	titleMaterial->m_materialProperties.m_albedoColor = glm::vec3(1, 0.2, 0.5);

	auto dancingStormTrooper = std::dynamic_pointer_cast<Prefab>(
		ProjectManager::GetOrCreateAsset("Models/dancing-stormtrooper/silly_dancing.fbx"));
	auto dancingStormTrooperEntity = dancingStormTrooper->ToEntity(scene);
	const auto dancingStormTrooperAnimationPlayer = scene->GetOrSetPrivateComponent<AnimationPlayer>(dancingStormTrooperEntity).lock();
	dancingStormTrooperAnimationPlayer->m_autoPlay = true;
	dancingStormTrooperAnimationPlayer->m_autoPlaySpeed = 30;
	scene->SetEntityName(dancingStormTrooperEntity, "StormTrooper");
	Transform dancingStormTrooperTransform;
	dancingStormTrooperTransform.SetValue(glm::vec3(1.2, -1.5, 0), glm::vec3(0), glm::vec3(0.4));
	scene->SetDataComponent(dancingStormTrooperEntity, dancingStormTrooperTransform);
	scene->SetParent(dancingStormTrooperEntity, baseEntity);


	const auto capoeira = std::dynamic_pointer_cast<Prefab>(ProjectManager::GetOrCreateAsset("Models/Capoeira.fbx"));
	const auto capoeiraEntity = capoeira->ToEntity(scene);
	const auto capoeiraAnimationPlayer = scene->GetOrSetPrivateComponent<AnimationPlayer>(capoeiraEntity).lock();
	capoeiraAnimationPlayer->m_autoPlay = true;
	capoeiraAnimationPlayer->m_autoPlaySpeed = 60;
	scene->SetEntityName(capoeiraEntity, "Capoeira");
	Transform capoeiraTransform;
	capoeiraTransform.SetValue(glm::vec3(0.5, 2.7, -18), glm::vec3(0), glm::vec3(0.02));
	scene->SetDataComponent(capoeiraEntity, capoeiraTransform);
	auto capoeiraBodyMaterial = scene
		->GetOrSetPrivateComponent<SkinnedMeshRenderer>(
			scene->GetChildren(scene->GetChildren(capoeiraEntity)[1])[0])
		.lock()
		->m_material.Get<Material>();
	capoeiraBodyMaterial->m_materialProperties.m_albedoColor = glm::vec3(0, 1, 1);
	capoeiraBodyMaterial->m_materialProperties.m_metallic = 1;
	capoeiraBodyMaterial->m_materialProperties.m_roughness = 0;
	auto capoeiraJointsMaterial = scene
		->GetOrSetPrivateComponent<SkinnedMeshRenderer>(
			scene->GetChildren(scene->GetChildren(capoeiraEntity)[0])[0])
		.lock()
		->m_material.Get<Material>();
	capoeiraJointsMaterial->m_materialProperties.m_albedoColor = glm::vec3(0.3, 1.0, 0.5);
	capoeiraJointsMaterial->m_materialProperties.m_metallic = 1;
	capoeiraJointsMaterial->m_materialProperties.m_roughness = 0;
	capoeiraJointsMaterial->m_materialProperties.m_emission = 6;
	scene->SetParent(capoeiraEntity, baseEntity);

#pragma endregion

	return baseEntity;
}

void SetupDemoScene(DemoSetup demoSetup, ApplicationInfo& applicationInfo)
{
	std::filesystem::path resourceFolderPath("../../../Resources");
	if (!std::filesystem::exists(resourceFolderPath)) {
		resourceFolderPath = "../../Resources";
	}
	if (!std::filesystem::exists(resourceFolderPath)) {
		resourceFolderPath = "../Resources";
	}
#pragma region Demo scene setup
	if (demoSetup != DemoSetup::Empty && std::filesystem::exists(resourceFolderPath)) {
		for (const auto i : std::filesystem::recursive_directory_iterator(resourceFolderPath))
		{
			if (i.is_directory()) continue;
			if (i.path().extension().string() == ".evescene" || i.path().extension().string() == ".evefilemeta" || i.path().extension().string() == ".eveproj" || i.path().extension().string() == ".evefoldermeta")
			{
				std::filesystem::remove(i.path());
			}
		}
		for (const auto i : std::filesystem::recursive_directory_iterator(resourceFolderPath))
		{
			if (i.is_directory()) continue;
			if (i.path().extension().string() == ".uescene" || i.path().extension().string() == ".umeta" || i.path().extension().string() == ".ueproj" || i.path().extension().string() == ".ufmeta")
			{
				std::filesystem::remove(i.path());
			}
		}
	}
	switch (demoSetup)
	{
	case DemoSetup::Rendering:
	{
		applicationInfo.m_applicationName = "Rendering Demo";
		applicationInfo.m_projectPath = resourceFolderPath / "Example Projects/Rendering/Rendering.eveproj";
		ProjectManager::SetActionAfterNewScene([&](const std::shared_ptr<Scene>& scene)
			{
				scene->m_environment.m_ambientLightIntensity = 0.1f;
#pragma region Set main camera to correct position and rotation
				const auto mainCamera = scene->m_mainCamera.Get<Camera>();
				mainCamera->Resize({ 640, 480 });
				mainCamera->m_postProcessingStack = ProjectManager::CreateTemporaryAsset<PostProcessingStack>();
				const auto mainCameraEntity = mainCamera->GetOwner();
				auto mainCameraTransform = scene->GetDataComponent<Transform>(mainCameraEntity);
				mainCameraTransform.SetPosition(glm::vec3(0, 0, 4));
				scene->SetDataComponent(mainCameraEntity, mainCameraTransform);
				auto camera = scene->GetOrSetPrivateComponent<Camera>(mainCameraEntity).lock();
				scene->GetOrSetPrivateComponent<PlayerController>(mainCameraEntity);
#pragma endregion

				LoadScene(scene, "Rendering Demo", true);

#pragma region Dynamic Lighting
				const auto dirLightEntity = scene->CreateEntity("Directional Light");
				const auto dirLight = scene->GetOrSetPrivateComponent<DirectionalLight>(dirLightEntity).lock();
				dirLight->m_diffuseBrightness = 0.0f;
				dirLight->m_lightSize = 0.2f;
				const auto pointLightRightEntity = scene->CreateEntity("Left Point Light");
				const auto pointLightRightRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(pointLightRightEntity).lock();
				const auto pointLightRightMaterial = ProjectManager::CreateTemporaryAsset<Material>();
				pointLightRightRenderer->m_material.Set<Material>(pointLightRightMaterial);
				pointLightRightMaterial->m_materialProperties.m_albedoColor = glm::vec3(1.0, 0.8, 0.0);
				pointLightRightMaterial->m_materialProperties.m_emission = 100.0f;
				pointLightRightRenderer->m_mesh = Resources::GetResource<Mesh>("PRIMITIVE_SPHERE");
				const auto pointLightRight = scene->GetOrSetPrivateComponent<PointLight>(pointLightRightEntity).lock();
				pointLightRight->m_diffuseBrightness = 4;
				pointLightRight->m_lightSize = 0.02f;
				pointLightRight->m_linear = 0.5f;
				pointLightRight->m_quadratic = 0.1f;
				pointLightRight->m_diffuse = glm::vec3(1.0, 0.8, 0.0);

				Transform dirLightTransform;
				dirLightTransform.SetEulerRotation(glm::radians(glm::vec3(105.0f, 0, 0.0f)));
				scene->SetDataComponent(dirLightEntity, dirLightTransform);
				Transform pointLightRightTransform;
				pointLightRightTransform.SetPosition(glm::vec3(4, 1.2, -5));
				pointLightRightTransform.SetScale({ 1.0f, 1.0f, 1.0f });
				scene->SetDataComponent(pointLightRightEntity, pointLightRightTransform);

				Application::RegisterUpdateFunction([=]() {
					static bool lastFramePlaying = false;
					if (!Application::IsPlaying()) {
						lastFramePlaying = Application::IsPlaying();
						return;
					}
					const auto currentScene = Application::GetActiveScene();
					static float startTime;
					if (!lastFramePlaying) startTime = Times::Now();
					const float currentTime = Times::Now() - startTime;
					const float cosTime = glm::cos(currentTime / 5.0f);
					Transform dirLightTransform;
					dirLightTransform.SetEulerRotation(glm::radians(glm::vec3(105.0f, currentTime * 20, 0.0f)));
					currentScene->SetDataComponent(dirLightEntity, dirLightTransform);

					Transform pointLightRightTransform;
					pointLightRightTransform.SetPosition(glm::vec3(4, 1.2, cosTime * 5 - 5));
					pointLightRightTransform.SetScale({ 1.0f, 1.0f, 1.0f });
					pointLightRightTransform.SetPosition(glm::vec3(4, 1.2, cosTime * 5 - 5));
					currentScene->SetDataComponent(pointLightRightEntity, pointLightRightTransform);

					lastFramePlaying = Application::IsPlaying();
					}
				);
#pragma endregion
			});
	}break;
	default:
	{

	}break;
	}
#pragma endregion
}

#pragma endregion
