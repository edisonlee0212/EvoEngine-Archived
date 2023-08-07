#include "AnimationPlayer.hpp"
#include "Application.hpp"
#include "WindowLayer.hpp"
#include "RenderLayer.hpp"
#include "EditorLayer.hpp"
#include "MeshRenderer.hpp"
#include "PlayerController.hpp"
#include "Prefab.hpp"
#include "Time.hpp"
#include "PhysicsLayer.hpp"
using namespace EvoEngine;
#pragma region Helpers
Entity CreateDynamicCube(
    const float& mass,
    const glm::vec3 color,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    const std::string& name);

Entity CreateSolidCube(
    const float& mass,
    const glm::vec3 color,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    const std::string& name);

Entity CreateCube(
    const glm::vec3 color,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    const std::string& name);

Entity CreateDynamicSphere(
    const float& mass,
    const glm::vec3 color,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const float& scale,
    const std::string& name);

Entity CreateSolidSphere(
    const float& mass,
    const glm::vec3 color,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const float& scale,
    const std::string& name);

Entity CreateSphere(
    const glm::vec3 color,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const float& scale,
    const std::string& name);
#pragma endregion
Entity LoadScene(const std::shared_ptr<Scene>& scene, const std::string& baseEntityName);
Entity LoadPhysicsScene(const std::shared_ptr<Scene>& scene, const std::string& baseEntityName);
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
    Application::PushLayer<PhysicsLayer>();
	Application::PushLayer<EditorLayer>();
    Application::PushLayer<RenderLayer>();
    
    ApplicationInfo applicationInfo;
    applicationInfo.m_projectPath = resourceFolderPath / "Example Projects/Rendering/Rendering.eveproj";
    ProjectManager::SetScenePostLoadActions([&](const std::shared_ptr<Scene>& scene)
    {
            scene->m_environment.m_ambientLightIntensity = 0.1f;
#pragma region Set main camera to correct position and rotation
            const auto mainCamera = scene->m_mainCamera.Get<Camera>();
            const auto mainCameraEntity = mainCamera->GetOwner();
            auto mainCameraTransform = scene->GetDataComponent<Transform>(mainCameraEntity);
            mainCameraTransform.SetPosition(glm::vec3(0, 0, 4));
            scene->SetDataComponent(mainCameraEntity, mainCameraTransform);
            auto camera = scene->GetOrSetPrivateComponent<Camera>(mainCameraEntity).lock();
            scene->GetOrSetPrivateComponent<PlayerController>(mainCameraEntity);
#pragma endregion
			LoadScene(scene, "Rendering Demo");
            const auto physicsDemo = LoadPhysicsScene(scene, "Physics Demo");
            Transform physicsDemoTransform;
            physicsDemoTransform.SetPosition(glm::vec3(-0.5f, -0.5f, -1.0f));
            scene->SetDataComponent(physicsDemo, physicsDemoTransform);
#pragma region Dynamic Lighting
            const auto dirLightEntity = scene->CreateEntity("Directional Light");
            const auto dirLight = scene->GetOrSetPrivateComponent<DirectionalLight>(dirLightEntity).lock();
            dirLight->m_diffuseBrightness = 2.5f;
            dirLight->m_lightSize = 0.1f;
            const auto pointLightRightEntity = scene->CreateEntity("Left Point Light");
            const auto pointLightRightRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(pointLightRightEntity).lock();
            const auto pointLightRightMaterial = ProjectManager::CreateTemporaryAsset<Material>();
            pointLightRightRenderer->m_material.Set<Material>(pointLightRightMaterial);
            pointLightRightMaterial->m_materialProperties.m_albedoColor = glm::vec3(1.0, 0.8, 0.0);
            pointLightRightMaterial->m_materialProperties.m_emission = 2.0f;
            pointLightRightRenderer->m_mesh = Resources::GetResource<Mesh>("PRIMITIVE_SPHERE");
            const auto pointLightRight = scene->GetOrSetPrivateComponent<PointLight>(pointLightRightEntity).lock();
            pointLightRight->m_diffuseBrightness = 4;
            pointLightRight->m_lightSize = 0.01f;
            pointLightRight->m_linear = 0.02f;
            pointLightRight->m_quadratic = 0.0001f;
            pointLightRight->m_diffuse = glm::vec3(1.0, 0.8, 0.0);

            Transform dirLightTransform;
            dirLightTransform.SetEulerRotation(glm::radians(glm::vec3(100.0f, 0, 0.0f)));
            scene->SetDataComponent(dirLightEntity, dirLightTransform);
            Transform pointLightRightTransform;
            pointLightRightTransform.SetPosition(glm::vec3(4, 1.2, -5));
            pointLightRightTransform.SetScale({ 0.1f, 0.1f, 0.1f });
            scene->SetDataComponent(pointLightRightEntity, pointLightRightTransform);

            Application::RegisterUpdateFunction([=]() {
                //if (!Application::IsPlaying())
                //    return;
                const auto currentScene = Application::GetActiveScene();
                const float currentTime = Time::CurrentTime();
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
                }
            );
#pragma endregion
    });
    Application::Initialize(applicationInfo);
    Application::Start();

    Application::Terminate();
    return 0;
}

Entity LoadScene(const std::shared_ptr<Scene>& scene, const std::string& baseEntityName)
{
    auto baseEntity = scene->CreateEntity(baseEntityName);
    /*
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
    scene->SetParent(collection, baseEntity);
#pragma endregion
	*/
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
    return baseEntity;
}

Entity LoadPhysicsScene(const std::shared_ptr<Scene>& scene, const std::string& baseEntityName)
{
	const auto baseEntity = scene->CreateEntity(baseEntityName);
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

                const auto rigidBody = scene->GetOrSetPrivateComponent<RigidBody>(sphere).lock();
                rigidBody->SetEnabled(true);
                rigidBody->SetDensityAndMassCenter(0.1f);
                auto sphereCollider = ProjectManager::CreateTemporaryAsset<Collider>();
                sphereCollider->SetShapeType(ShapeType::Sphere);
                sphereCollider->SetShapeParam(glm::vec3(2.0f * scaleFactor));
                rigidBody->AttachCollider(sphereCollider);
                scene->SetParent(sphere, collection);
            }
        }
    }
    scene->SetParent(collection, baseEntity);
#pragma endregion
#pragma region Create Boundaries
    {

        const auto ground =
            CreateSolidCube(1.0, glm::vec3(1.0f), glm::vec3(0, -35, 0) * scaleFactor, glm::vec3(0), glm::vec3(30, 1, 60) * scaleFactor, "Ground");

        const auto rightWall = CreateSolidCube(1.0, glm::vec3(1.0f), glm::vec3(30, -20, 0) * scaleFactor, glm::vec3(0), glm::vec3(1, 15, 60) * scaleFactor, "LeftWall");
        const auto leftWall = CreateSolidCube(1.0, glm::vec3(1.0f), glm::vec3(-30, -20, 0) * scaleFactor, glm::vec3(0), glm::vec3(1, 15, 60) * scaleFactor, "RightWall");
        const auto frontWall = CreateSolidCube(1.0, glm::vec3(1.0f), glm::vec3(0, -30, 60) * scaleFactor, glm::vec3(0), glm::vec3(30, 5, 1) * scaleFactor, "FrontWall");
        const auto backWall = CreateSolidCube(1.0, glm::vec3(1.0f), glm::vec3(0, -20, -60) * scaleFactor, glm::vec3(0), glm::vec3(30, 15, 1) * scaleFactor, "BackWall");
        scene->SetParent(rightWall, collection);
        scene->SetParent(ground, collection);
        scene->SetParent(backWall, collection);
        scene->SetParent(leftWall, collection);
        scene->SetParent(frontWall, collection);
    	/*
        const auto b1 = CreateDynamicCube(
            1.0, glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(-5, -7.5, 0) * scaleFactor, glm::vec3(0, 0, 45), glm::vec3(0.5) * scaleFactor, "Block 1");
        const auto b2 =
            CreateDynamicCube(1.0, glm::vec3(1.0f), glm::vec3(0, -10, 0) * scaleFactor, glm::vec3(0, 0, 45), glm::vec3(1) * scaleFactor, "Block 2");
        const auto b3 = CreateDynamicCube(
            1.0, glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(5, -7.5, 0) * scaleFactor, glm::vec3(0, 0, 45), glm::vec3(1) * scaleFactor, "Block 3");

        auto b1j = scene->GetOrSetPrivateComponent<Joint>(b1).lock();
        b1j->SetType(JointType::Fixed);
        b1j->Link(b2);
        auto b3j = scene->GetOrSetPrivateComponent<Joint>(b3).lock();
        b3j->SetType(JointType::Fixed);
        b3j->Link(b2);
        auto b2j = scene->GetOrSetPrivateComponent<Joint>(b2).lock();
        b2j->SetType(JointType::Fixed);
        b2j->Link(ground);

        const auto anchor = CreateDynamicCube(
            1.0, glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(-10, 0, 0) * scaleFactor, glm::vec3(0, 0, 45), glm::vec3(2.0f) * scaleFactor, "Anchor");
        scene->GetOrSetPrivateComponent<RigidBody>(anchor).lock()->SetKinematic(true);
        auto lastLink = anchor;
        for (int i = 1; i < 10; i++)
        {
            const auto link = CreateDynamicSphere(
                1.0, glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(-10 - i, 0, 0) * scaleFactor, glm::vec3(0, 0, 45), 2.0f * scaleFactor, "Link");
            auto joint = scene->GetOrSetPrivateComponent<Joint>(link).lock();
            joint->SetType(JointType::D6);
            joint->Link(lastLink);
            // joint->SetMotion(MotionAxis::SwingY, MotionType::Limited);
            scene->SetParent(link, anchor);
            lastLink = link;
        }

        const auto freeSphere = CreateDynamicCube(
            0.01, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(-20, 0, 0) * scaleFactor, glm::vec3(0, 0, 45), glm::vec3(5.0f) * scaleFactor, "Free Cube");
        auto joint = scene->GetOrSetPrivateComponent<Joint>(freeSphere).lock();
        joint->SetType(JointType::D6);
        joint->Link(lastLink);
        // joint->SetMotion(MotionAxis::TwistX, MotionType::Free);
        joint->SetMotion(MotionAxis::SwingY, MotionType::Free);
        joint->SetMotion(MotionAxis::SwingZ, MotionType::Free);
        */
    }
#pragma endregion
    return baseEntity;
}


Entity CreateSolidCube(
    const float& mass,
    const glm::vec3 color,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    const std::string& name)
{
    auto scene = Application::GetActiveScene();
    auto cube = CreateCube(color, position, rotation, scale, name);
    auto rigidBody = scene->GetOrSetPrivateComponent<RigidBody>(cube).lock();
    rigidBody->SetStatic(true);
    // The rigidbody can only apply mesh bound after it's attached to an entity with mesh renderer.
    rigidBody->SetEnabled(true);

    auto collider = ProjectManager::CreateTemporaryAsset<Collider>();
    collider->SetShapeType(ShapeType::Box);
    collider->SetShapeParam(scale);
    rigidBody->AttachCollider(collider);
    return cube;
}

Entity CreateDynamicCube(
    const float& mass,
    const glm::vec3 color,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    const std::string& name)
{
    auto scene = Application::GetActiveScene();
    auto cube = CreateCube(color, position, rotation, scale, name);
    auto rigidBody = scene->GetOrSetPrivateComponent<RigidBody>(cube).lock();
    rigidBody->SetStatic(false);
    rigidBody->SetDensityAndMassCenter(1);
    // The rigidbody can only apply mesh bound after it's attached to an entity with mesh renderer.
    rigidBody->SetEnabled(true);
    rigidBody->SetDensityAndMassCenter(mass / scale.x / scale.y / scale.z);

    auto collider = ProjectManager::CreateTemporaryAsset<Collider>();
    collider->SetShapeType(ShapeType::Box);
    collider->SetShapeParam(scale);
    rigidBody->AttachCollider(collider);
    return cube;
}

Entity CreateCube(
    const glm::vec3 color,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    const std::string& name)
{
    auto scene = Application::GetActiveScene();
    auto cube = scene->CreateEntity(name);
    auto groundMeshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(cube).lock();
    auto material = ProjectManager::CreateTemporaryAsset<Material>();
    groundMeshRenderer->m_material = material;
    material->m_materialProperties.m_albedoColor = color;
    groundMeshRenderer->m_mesh = Resources::GetResource<Mesh>("PRIMITIVE_CUBE");
    Transform groundTransform;
    groundTransform.SetValue(position, glm::radians(rotation), scale * 2.0f);
    scene->SetDataComponent(cube, groundTransform);
    /*
    GlobalTransform groundGlobalTransform;
    groundGlobalTransform.SetValue(position, glm::radians(rotation), scale);
    scene->SetDataComponent(cube, groundGlobalTransform);
    */
    return cube;
}

Entity CreateDynamicSphere(
    const float& mass,
    const glm::vec3 color,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const float& scale,
    const std::string& name)
{
    auto scene = Application::GetActiveScene();
    auto sphere = CreateSphere(color, position, rotation, scale, name);
    auto rigidBody = scene->GetOrSetPrivateComponent<RigidBody>(sphere).lock();
    rigidBody->SetStatic(false);
    rigidBody->SetDensityAndMassCenter(1);
    // The rigidbody can only apply mesh bound after it's attached to an entity with mesh renderer.
    rigidBody->SetEnabled(true);
    rigidBody->SetDensityAndMassCenter(mass / scale / scale / scale);

    auto collider = ProjectManager::CreateTemporaryAsset<Collider>();
    collider->SetShapeType(ShapeType::Sphere);
    collider->SetShapeParam(glm::vec3(scale));
    rigidBody->AttachCollider(collider);
    return sphere;
}

Entity CreateSolidSphere(
    const float& mass,
    const glm::vec3 color,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const float& scale,
    const std::string& name)
{
    auto scene = Application::GetActiveScene();
    auto sphere = CreateSphere(color, position, rotation, scale, name);
    auto rigidBody = scene->GetOrSetPrivateComponent<RigidBody>(sphere).lock();
    rigidBody->SetStatic(true);
    // The rigidbody can only apply mesh bound after it's attached to an entity with mesh renderer.
    rigidBody->SetEnabled(true);

    auto collider = ProjectManager::CreateTemporaryAsset<Collider>();
    collider->SetShapeType(ShapeType::Sphere);
    collider->SetShapeParam(glm::vec3(scale));
    rigidBody->AttachCollider(collider);
    return sphere;
}

Entity CreateSphere(
    const glm::vec3 color,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const float& scale,
    const std::string& name)
{
    auto scene = Application::GetActiveScene();
    auto sphere = scene->CreateEntity(name);
    auto groundMeshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(sphere).lock();
    auto material = ProjectManager::CreateTemporaryAsset<Material>();
    groundMeshRenderer->m_material = material;
    material->m_materialProperties.m_albedoColor = color;
    groundMeshRenderer->m_mesh = Resources::GetResource<Mesh>("PRIMITIVE_CUBE");
    Transform groundTransform;
    groundTransform.SetValue(position, glm::radians(rotation), glm::vec3(scale));
    // groundTransform.SetValue(glm::vec3(0, -15, 0), glm::vec3(0), glm::vec3(30, 1, 30));
    scene->SetDataComponent(sphere, groundTransform);

    GlobalTransform groundGlobalTransform;
    groundGlobalTransform.SetValue(position, glm::radians(rotation), glm::vec3(scale));
    scene->SetDataComponent(sphere, groundGlobalTransform);
    return sphere;
}