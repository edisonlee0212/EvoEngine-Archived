#include "Application.hpp"
#include "WindowLayer.hpp"
#include "RenderLayer.hpp"
#include "EditorLayer.hpp"
#include "MeshRenderer.hpp"
#include "PlayerController.hpp"
#include "Prefab.hpp"
#include "Time.hpp"
using namespace EvoEngine;

void LoadScene();

int main() {
    const std::filesystem::path resourceFolderPath("../../../Resources");

    for (const auto i : std::filesystem::recursive_directory_iterator(resourceFolderPath))
    {
        if (i.is_directory()) continue;
        if (i.path().extension().string() == ".evescene" || i.path().extension().string() == ".evefilemeta" || i.path().extension().string() == ".eveproj" || i.path().extension().string() == ".evefoldermeta")
        {
            std::filesystem::remove(i.path());
        }
    }


    Application::PushLayer<WindowLayer>();
	Application::PushLayer<EditorLayer>();
    Application::PushLayer<RenderLayer>();

    ApplicationInfo applicationInfo;
    applicationInfo.m_projectPath = resourceFolderPath / "Example Projects/Rendering/Rendering.eveproj";

    ProjectManager::SetScenePostLoadActions([]() { LoadScene(); });
    Application::Initialize(applicationInfo);
    Application::Start();

    Application::Terminate();
    return 0;
}

void LoadScene()
{
    auto scene = Application::GetActiveScene();

    scene->m_environment.m_ambientLightIntensity = 0.1f;
#pragma region Set main camera to correct position and rotation
    const auto mainCamera = scene->m_mainCamera.Get<Camera>();
    auto mainCameraEntity = mainCamera->GetOwner();
    auto mainCameraTransform = scene->GetDataComponent<Transform>(mainCameraEntity);
    mainCameraTransform.SetPosition(glm::vec3(0, 0, 4));
    scene->SetDataComponent(mainCameraEntity, mainCameraTransform);
    auto camera = scene->GetOrSetPrivateComponent<Camera>(mainCameraEntity).lock();
    scene->GetOrSetPrivateComponent<PlayerController>(mainCameraEntity);
#pragma endregion

#pragma region Create 9 spheres in different PBR properties
    int amount = 6;
    auto collection = scene->CreateEntity("Spheres");
    auto spheres = scene->CreateEntities(amount * amount, "Instance");
    for (int i = 0; i < amount; i++)
    {
        for (int j = 0; j < amount; j++)
        {
            auto& sphere = spheres[i * amount + j];
            Transform transform;
            glm::vec3 position = glm::vec3(i - amount / 2.0f, j - amount / 2.0f, 0);
            transform.SetPosition(position * 0.2f);
            transform.SetScale(glm::vec3(0.2f));
            scene->SetDataComponent(sphere, transform);
            auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(sphere).lock();
            meshRenderer->m_mesh = Resources::GetResource<Mesh>("PRIMITIVE_SPHERE");
            auto material = ProjectManager::CreateTemporaryAsset<Material>();
            meshRenderer->m_material.Set<Material>(material);
            material->m_materialProperties.m_roughness = static_cast<float>(i) / (amount - 1);
            material->m_materialProperties.m_metallic = static_cast<float>(j) / (amount - 1);

            scene->SetParent(sphere, collection);
        }
    }
    Transform collectionTransform;
    collectionTransform.SetPosition({ 0.0, -0.8, 0.5 });
    collectionTransform.SetRotation(glm::radians(glm::vec3(-45, 0, 0)));
    scene->SetDataComponent(collection, collectionTransform);
#pragma endregion
#pragma region Load models and display

    auto sponza = std::dynamic_pointer_cast<Prefab>(ProjectManager::GetOrCreateAsset("Models/Sponza_FBX/Sponza.fbx"));
    auto sponzaEntity = sponza->ToEntity(scene);
    Transform sponzaTransform;
    sponzaTransform.SetValue(glm::vec3(0, -1.5, -6), glm::radians(glm::vec3(0, -90, 0)), glm::vec3(0.01));
    scene->SetDataComponent(sponzaEntity, sponzaTransform);

    auto title = std::dynamic_pointer_cast<Prefab>(ProjectManager::GetOrCreateAsset("Models/UniEngine.obj"));
    auto titleEntity = title->ToEntity(scene);
    scene->SetEntityName(titleEntity, "Title");
    Transform titleTransform;
    titleTransform.SetValue(glm::vec3(0.35, 7, -16), glm::radians(glm::vec3(0, 0, 0)), glm::vec3(0.005));
    scene->SetDataComponent(titleEntity, titleTransform);

    auto titleMaterial =
        scene->GetOrSetPrivateComponent<MeshRenderer>(scene->GetChildren(scene->GetChildren(titleEntity)[0])[0])
        .lock()
        ->m_material.Get<Material>();
    titleMaterial->m_materialProperties.m_emission = 4;
    titleMaterial->m_materialProperties.m_albedoColor = glm::vec3(1, 0.2, 0.5);

    auto dancingStormTrooper = std::dynamic_pointer_cast<Prefab>(
        ProjectManager::GetOrCreateAsset("Models/dancing-stormtrooper/silly_dancing.fbx"));
    auto dancingStormTrooperEntity = dancingStormTrooper->ToEntity(scene);
    const auto dancingStormTrooperAnimator = scene->GetOrSetPrivateComponent<Animator>(dancingStormTrooperEntity).lock();
    dancingStormTrooperAnimator->m_autoPlay = true;
    dancingStormTrooperAnimator->m_autoPlaySpeed = 30;
    scene->SetEntityName(dancingStormTrooperEntity, "StormTrooper");
    Transform dancingStormTrooperTransform;
    dancingStormTrooperTransform.SetValue(glm::vec3(1.2, -1.5, 0), glm::vec3(0), glm::vec3(0.4));
    scene->SetDataComponent(dancingStormTrooperEntity, dancingStormTrooperTransform);

    auto capoeira = std::dynamic_pointer_cast<Prefab>(ProjectManager::GetOrCreateAsset("Models/Capoeira.fbx"));
    auto capoeiraEntity = capoeira->ToEntity(scene);
    const auto capoeiraAnimator = scene->GetOrSetPrivateComponent<Animator>(capoeiraEntity).lock();
    capoeiraAnimator->m_autoPlay = true;
    capoeiraAnimator->m_autoPlaySpeed = 60;
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
#pragma endregion

#pragma region Create ground
    auto ground = scene->CreateEntity("Ground");
    auto groundMeshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(ground).lock();
    auto groundMat = ProjectManager::CreateTemporaryAsset<Material>();
    
    groundMeshRenderer->m_material = groundMat;
    groundMeshRenderer->m_mesh = Resources::GetResource<Mesh>("PRIMITIVE_CUBE");
    Transform groundTransform;
    groundTransform.SetValue(glm::vec3(0, -2.05, -0), glm::vec3(0), glm::vec3(30, 1, 60));
    scene->SetDataComponent(ground, groundTransform);
#pragma endregion

#pragma region Lighting
    auto dirLightEntity = scene->CreateEntity("Directional Light");
    auto dirLight = scene->GetOrSetPrivateComponent<DirectionalLight>(dirLightEntity).lock();
    dirLight->m_diffuseBrightness = 2.5f;
    dirLight->m_lightSize = 0.1f;
    /*
    auto pointLightLeftEntity = scene->CreateEntity("Right Point Light");
    auto pointLightLeftRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(pointLightLeftEntity).lock();
    auto groundMaterial = ProjectManager::CreateTemporaryAsset<Material>();
    pointLightLeftRenderer->m_material.Set<Material>(groundMaterial);
    groundMaterial->m_materialProperties.m_albedoColor = glm::vec3(0.0, 0.5, 1.0);
    groundMaterial->m_materialProperties.m_emission = 2.0f;
    pointLightLeftRenderer->m_mesh = Resources::GetResource<Mesh>("PRIMITIVE_SPHERE");
    auto pointLightLeft = scene->GetOrSetPrivateComponent<PointLight>(pointLightLeftEntity).lock();
    pointLightLeft->m_diffuseBrightness = 4;
    pointLightLeft->m_lightSize = 0.02f;
    pointLightLeft->m_linear = 0.02;
    pointLightLeft->m_quadratic = 0.0001;
    pointLightLeft->m_diffuse = glm::vec3(0.0, 0.5, 1.0);
    */
    auto pointLightRightEntity = scene->CreateEntity("Left Point Light");
    auto pointLightRightRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(pointLightRightEntity).lock();
    auto pointLightRightMaterial = ProjectManager::CreateTemporaryAsset<Material>();
    pointLightRightRenderer->m_material.Set<Material>(pointLightRightMaterial);
    pointLightRightMaterial->m_materialProperties.m_albedoColor = glm::vec3(1.0, 0.8, 0.0);
    pointLightRightMaterial->m_materialProperties.m_emission = 2.0f;
    pointLightRightRenderer->m_mesh = Resources::GetResource<Mesh>("PRIMITIVE_SPHERE");
    auto pointLightRight = scene->GetOrSetPrivateComponent<PointLight>(pointLightRightEntity).lock();
    pointLightRight->m_diffuseBrightness = 4;
    pointLightRight->m_lightSize = 0.01f;
    pointLightRight->m_linear = 0.02;
    pointLightRight->m_quadratic = 0.0001;
    pointLightRight->m_diffuse = glm::vec3(1.0, 0.8, 0.0);
    
#pragma endregion

    double startTime = 0;
    float startSinTime = glm::sin(startTime / 5.0f);
    float startCosTime = glm::cos(startTime / 5.0f);

    Transform dirLightTransform;
    dirLightTransform.SetEulerRotation(glm::radians(glm::vec3(100.0f, startTime * 10, 0.0f)));
    scene->SetDataComponent(dirLightEntity, dirLightTransform);
    /*
    Transform pointLightLeftTransform;
    pointLightLeftTransform.SetPosition(glm::vec3(-4, 1.2, startSinTime * 5 - 5));
    pointLightLeftTransform.SetScale({ 0.1f, 0.1f, 0.1f });
    scene->SetDataComponent(pointLightLeftEntity, pointLightLeftTransform);
    */
    Transform pointLightRightTransform;
    pointLightRightTransform.SetPosition(glm::vec3(4, 1.2, startCosTime * 5 - 5));
    pointLightRightTransform.SetScale({ 0.1f, 0.1f, 0.1f });
    scene->SetDataComponent(pointLightRightEntity, pointLightRightTransform);
    
    Application::RegisterUpdateFunction([=]() {
        //if (!Application::IsPlaying())
        //    return;
        auto currentScene = Application::GetActiveScene();
        const float currentTime = Time::CurrentTime();
        const float sinTime = glm::sin(currentTime / 5.0f);
        const float cosTime = glm::cos(currentTime / 5.0f);
        Transform dirLightTransform;
        dirLightTransform.SetEulerRotation(glm::radians(glm::vec3(10.0f, currentTime, 0.0f)));
        currentScene->SetDataComponent(dirLightEntity, dirLightTransform);
        dirLightTransform.SetEulerRotation(glm::radians(glm::vec3(100.0f, currentTime * 10, 0.0f)));
        currentScene->SetDataComponent(dirLightEntity, dirLightTransform);


        Transform pointLightRightTransform;
        pointLightRightTransform.SetPosition(glm::vec3(4, 1.2, cosTime * 5 - 5));
        pointLightRightTransform.SetScale({ 0.1f, 0.1f, 0.1f });
        pointLightRightTransform.SetPosition(glm::vec3(4, 1.2, cosTime * 5 - 5));
        currentScene->SetDataComponent(pointLightRightEntity, pointLightRightTransform);

        /*
        Transform pointLightLeftTransform;
        pointLightLeftTransform.SetPosition(glm::vec3(-4, 1.2, sinTime * 5 - 5));
        pointLightLeftTransform.SetScale({ 0.1f, 0.1f, 0.1f });
        pointLightLeftTransform.SetPosition(glm::vec3(-4, 1.2, sinTime * 5 - 5));
        scene->SetDataComponent(pointLightLeftEntity, pointLightLeftTransform);
        */
    });
     
}