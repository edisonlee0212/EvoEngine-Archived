#include "EditorLayer.hpp"
#include "ILayer.hpp"
#include "Application.hpp"
#include "Console.hpp"
#include "Graphics.hpp"
#include "ProjectManager.hpp"
#include "WindowLayer.hpp"
#include "Scene.hpp"
using namespace EvoEngine;

void EditorLayer::OnCreate()
{
    m_basicEntityArchetype = Entities::CreateEntityArchetype("General", GlobalTransform(), Transform());
    RegisterComponentDataInspector<GlobalTransform>([](Entity entity, IDataComponent* data, bool isRoot) {
        auto* ltw = reinterpret_cast<GlobalTransform*>(data);
        glm::vec3 er;
        glm::vec3 t;
        glm::vec3 s;
        ltw->Decompose(t, er, s);
        er = glm::degrees(er);
        ImGui::DragFloat3("Position##Global", &t.x, 0.1f, 0, 0, "%.3f", ImGuiSliderFlags_ReadOnly);
        ImGui::DragFloat3("Rotation##Global", &er.x, 0.1f, 0, 0, "%.3f", ImGuiSliderFlags_ReadOnly);
        ImGui::DragFloat3("Scale##Global", &s.x, 0.1f, 0, 0, "%.3f", ImGuiSliderFlags_ReadOnly);
        return false;
        });
    RegisterComponentDataInspector<Transform>([&](Entity entity, IDataComponent* data, bool isRoot) {
        static Entity previousEntity;
        auto* ltp = static_cast<Transform*>(static_cast<void*>(data));
        bool edited = false;
        bool reload = previousEntity != entity;
        bool readOnly = false;
        auto scene = Application::GetActiveScene();
        /*
        if (Application::IsPlaying() && scene->HasPrivateComponent<RigidBody>(entity)) {
            auto rigidBody = scene->GetOrSetPrivateComponent<RigidBody>(entity).lock();
            if (!rigidBody->IsKinematic() && rigidBody->Registered()) {
                reload = true;
                readOnly = true;
            }
        }
        */
        if (reload) {
            previousEntity = entity;
            ltp->Decompose(m_previouslyStoredPosition, m_previouslyStoredRotation, m_previouslyStoredScale);
            m_previouslyStoredRotation = glm::degrees(m_previouslyStoredRotation);
            m_localPositionSelected = true;
            m_localRotationSelected = false;
            m_localScaleSelected = false;
        }
        if (ImGui::DragFloat3(
            "##LocalPosition",
            &m_previouslyStoredPosition.x,
            0.1f,
            0,
            0,
            "%.3f",
            readOnly ? ImGuiSliderFlags_ReadOnly : 0))
            edited = true;
        ImGui::SameLine();
        if (ImGui::Selectable("Position##Local", &m_localPositionSelected) && m_localPositionSelected) {
            m_localRotationSelected = false;
            m_localScaleSelected = false;
        }
        if (ImGui::DragFloat3(
            "##LocalRotation",
            &m_previouslyStoredRotation.x,
            1.0f,
            0,
            0,
            "%.3f",
            readOnly ? ImGuiSliderFlags_ReadOnly : 0))
            edited = true;
        ImGui::SameLine();
        if (ImGui::Selectable("Rotation##Local", &m_localRotationSelected) && m_localRotationSelected) {
            m_localPositionSelected = false;
            m_localScaleSelected = false;
        }
        if (ImGui::DragFloat3(
            "##LocalScale",
            &m_previouslyStoredScale.x,
            0.01f,
            0,
            0,
            "%.3f",
            readOnly ? ImGuiSliderFlags_ReadOnly : 0))
            edited = true;
        ImGui::SameLine();
        if (ImGui::Selectable("Scale##Local", &m_localScaleSelected) && m_localScaleSelected) {
            m_localRotationSelected = false;
            m_localPositionSelected = false;
        }
        if (edited) {
            ltp->m_value = glm::translate(m_previouslyStoredPosition) *
                glm::mat4_cast(glm::quat(glm::radians(m_previouslyStoredRotation))) *
                glm::scale(m_previouslyStoredScale);
        }
        return edited;
        });
    /*
    RegisterComponentDataInspector<Ray>([&](Entity entity, IDataComponent* data, bool isRoot) {
        auto* ray = static_cast<Ray*>(static_cast<void*>(data));
        bool changed = false;
        if (ImGui::InputFloat3("Start", &ray->m_start.x))
            changed = true;
        if (ImGui::InputFloat3("Direction", &ray->m_direction.x))
            changed = true;
        if (ImGui::InputFloat("Length", &ray->m_length))
            changed = true;
        return changed;
        });
        */

	const auto& windowLayer = Application::GetLayer<WindowLayer>();
	if (!windowLayer)
	{
		std::runtime_error("No WindowLayer present!");
		throw;
	}

	CreateRenderPass();
	UpdateFrameBuffers();
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleFonts;
    //io.ConfigFlags |= ImGuiConfigFlags_IsSRGB;
	ImGui::StyleColorsDark();

	//1: create descriptor pool for IMGUI
	// the size of the pool is very oversize, but it's copied from imgui demo itself.
	VkDescriptorPoolSize pool_sizes[] =
	{
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
	};

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.maxSets = 1000;
	pool_info.poolSizeCount = std::size(pool_sizes);
	pool_info.pPoolSizes = pool_sizes;

	m_imguiDescriptorPool.Create(pool_info);


	// 2: initialize imgui library

	//this initializes the core structures of imgui
	ImGui::CreateContext();

	//this initializes imgui for SDL
	ImGui_ImplGlfw_InitForVulkan(windowLayer->GetGlfwWindow(), true);

	//this initializes imgui for Vulkan
	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = Graphics::GetVkInstance();
	init_info.PhysicalDevice = Graphics::GetVkPhysicalDevice();
	init_info.Device = Graphics::GetVkDevice();
	init_info.QueueFamily = Graphics::GetQueueFamilyIndices().m_graphicsFamily.value();
	init_info.Queue = Graphics::GetGraphicsVkQueue();
	init_info.PipelineCache = VK_NULL_HANDLE;
	init_info.DescriptorPool = m_imguiDescriptorPool.GetVkDescriptorPool();
	init_info.MinImageCount = Graphics::GetSwapchain().GetVkImageViews().size();
	init_info.ImageCount = Graphics::GetSwapchain().GetVkImageViews().size();
	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    
	ImGui_ImplVulkan_LoadFunctions([](const char* function_name, void*) { return vkGetInstanceProcAddr(Graphics::GetVkInstance(), function_name); });
	ImGui_ImplVulkan_Init(&init_info, m_renderPass.GetVkRenderPass());
    ImGui::StyleColorsDark();
    //ImGui::GetStyle().ScaleAllSizes(2.0);
    
	//execute a gpu command to upload imgui font textures
	Graphics::ImmediateSubmit([&](VkCommandBuffer cmd) {
		ImGui_ImplVulkan_CreateFontsTexture(cmd);
		});

	//clear font textures from cpu data
	//ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void EditorLayer::OnDestroy()
{
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	m_imguiDescriptorPool.Destroy();
}

void EditorLayer::PreUpdate()
{
	UpdateFrameBuffers();

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGuizmo::BeginFrame();

#pragma region Dock
	static bool opt_fullscreen_persistant = true;
	bool opt_fullscreen = opt_fullscreen_persistant;
	static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

	// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
	// because it would be confusing to have two docking targets within each others.
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
	if (opt_fullscreen)
	{
		ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(viewport->WorkPos);
		ImGui::SetNextWindowSize(viewport->WorkSize);
		ImGui::SetNextWindowViewport(viewport->ID);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove;
		window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
	}

	// When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
	// and handle the pass-thru hole, so we ask Begin() to not render a background.
	if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
		window_flags |= ImGuiWindowFlags_NoBackground;

	// Important: note that we proceed even if Begin() returns false (aka window is collapsed).
	// This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
	// all active windows docked into it will lose their parent and become undocked.
	// We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
	// any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	static bool openDock = true;
	ImGui::Begin("Root DockSpace", &openDock, window_flags);
	ImGui::PopStyleVar();
	if (opt_fullscreen)
		ImGui::PopStyleVar(2);
	ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
	ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
	ImGui::End();
#pragma endregion

    m_mainCameraFocusOverride = false;
    m_sceneCameraFocusOverride = false;
    if (ImGui::BeginMainMenuBar()) {
        /*
        switch (Application::GetApplicationStatus()) {
        case ApplicationStatus::Stop: {
            if (ImGui::ImageButton(
                (ImTextureID)m_assetsIcons["PlayButton"]->UnsafeGetGLTexture()->Id(),
                { 15, 15 },
                { 0, 1 },
                { 1, 0 })) {
                Application::Play();
            }
            if (ImGui::ImageButton(
                (ImTextureID)m_assetsIcons["StepButton"]->UnsafeGetGLTexture()->Id(),
                { 15, 15 },
                { 0, 1 },
                { 1, 0 })) {
                Application::Step();
            }
            break;
        }
        case ApplicationStatus::Playing: {
            if (ImGui::ImageButton(
                (ImTextureID)m_assetsIcons["PauseButton"]->UnsafeGetGLTexture()->Id(),
                { 15, 15 },
                { 0, 1 },
                { 1, 0 })) {
                Application::Pause();
            }
            if (ImGui::ImageButton(
                (ImTextureID)m_assetsIcons["StopButton"]->UnsafeGetGLTexture()->Id(),
                { 15, 15 },
                { 0, 1 },
                { 1, 0 })) {
                Application::Stop();
            }
            break;
        }
        case ApplicationStatus::Pause: {
            if (ImGui::ImageButton(
                (ImTextureID)m_assetsIcons["PlayButton"]->UnsafeGetGLTexture()->Id(),
                { 15, 15 },
                { 0, 1 },
                { 1, 0 })) {
                Application::Play();
            }
            if (ImGui::ImageButton(
                (ImTextureID)m_assetsIcons["StepButton"]->UnsafeGetGLTexture()->Id(),
                { 15, 15 },
                { 0, 1 },
                { 1, 0 })) {
                Application::Step();
            }
            if (ImGui::ImageButton(
                (ImTextureID)m_assetsIcons["StopButton"]->UnsafeGetGLTexture()->Id(),
                { 15, 15 },
                { 0, 1 },
                { 1, 0 })) {
                Application::Stop();
            }
            break;
        }
        }
        */
        ImGui::Separator();
        if (ImGui::BeginMenu("Project")) {
            ImGui::EndMenu();
        }
        /*
        if (ImGui::BeginMenu("File"))
        {
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))
        {
            ImGui::EndMenu();
        }
        */
        if (ImGui::BeginMenu("View")) {
            ImGui::EndMenu();
        }
        /*
        if (ImGui::BeginMenu("Help"))
        {
            ImGui::EndMenu();
        }
        */
        ImGui::EndMainMenuBar();
    }
    m_mouseScreenPosition = glm::vec2(FLT_MAX, FLT_MIN);
}

void EditorLayer::Update()
{
	ImGui::ShowDemoWindow();
}
static const char* HierarchyDisplayMode[]{ "Archetype", "Hierarchy" };
void EditorLayer::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
    const auto& windowLayer = Application::GetLayer<WindowLayer>();
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("View"))
        {
            ImGui::Checkbox("Scene", &m_showSceneWindow);
            ImGui::Checkbox("Camera", &m_showCameraWindow);
            ImGui::Checkbox("Entity Explorer", &m_showEntityExplorerWindow);
            ImGui::Checkbox("Entity Inspector", &m_showEntityInspectorWindow);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    if (m_leftMouseButtonHold && !windowLayer->GetKey(GLFW_MOUSE_BUTTON_LEFT)) {
        m_leftMouseButtonHold = false;
    }
    if (m_rightMouseButtonHold && !windowLayer->GetKey(GLFW_MOUSE_BUTTON_RIGHT)) {
        m_rightMouseButtonHold = false;
        m_startMouse = false;
    }

    auto scene = GetScene();
    if (scene && m_showEntityExplorerWindow) {
        ImGui::Begin("Entity Explorer");
        if (ImGui::BeginPopupContextWindow("NewEntityPopup")) {
            if (ImGui::Button("Create new entity")) {
                auto newEntity = scene->CreateEntity(m_basicEntityArchetype);
            }
            ImGui::EndPopup();
        }
        ImGui::Combo(
            "Display mode", &m_selectedHierarchyDisplayMode, HierarchyDisplayMode,
            IM_ARRAYSIZE(HierarchyDisplayMode));
        std::string title = scene->GetTitle();
        if (ImGui::CollapsingHeader(title.c_str(),
            ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow)) {
            EditorLayer::DraggableAsset(scene);
            EditorLayer::RenameAsset(scene);
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                ProjectManager::GetInstance().m_inspectingAsset = scene;
            }
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Entity")) {
                    IM_ASSERT(payload->DataSize == sizeof(Handle));
                    auto payload_n = *static_cast<Handle*>(payload->Data);
                    auto newEntity = scene->GetEntity(payload_n);
                    auto parent = scene->GetParent(newEntity);
                    if (parent.GetIndex() != 0)
                        scene->RemoveChild(newEntity, parent);
                }
                ImGui::EndDragDropTarget();
            }
            if (m_selectedHierarchyDisplayMode == 0) {
                scene->UnsafeForEachEntityStorage(
                    [&](int i, const std::string& name, const DataComponentStorage& storage) {
                        if (i == 0)
                            return;
                        ImGui::Separator();
                        const std::string title = std::to_string(i) + ". " + name;
                        if (ImGui::TreeNode(title.c_str())) {
                            ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.2, 0.3, 0.2, 1.0));
                            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.2, 0.2, 0.2, 1.0));
                            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.2, 0.2, 0.3, 1.0));
                            for (int j = 0; j < storage.m_entityAliveCount; j++) {
                                Entity entity = storage.m_chunkArray.m_entities.at(j);
                                std::string title = std::to_string(entity.GetIndex()) + ": ";
                                title += scene->GetEntityName(entity);
                                const bool enabled = scene->IsEntityEnabled(entity);
                                if (enabled) {
                                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4({ 1, 1, 1, 1 }));
                                }
                                ImGui::TreeNodeEx(
                                    title.c_str(),
                                    ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_Leaf |
                                    ImGuiTreeNodeFlags_NoAutoOpenOnLog |
                                    (m_selectedEntity == entity ? ImGuiTreeNodeFlags_Framed
                                        : ImGuiTreeNodeFlags_FramePadding));
                                if (enabled) {
                                    ImGui::PopStyleColor();
                                }
                                DrawEntityMenu(enabled, entity);
                                if (!m_lockEntitySelection && ImGui::IsItemHovered() &&
                                    ImGui::IsMouseClicked(0)) {
                                    SetSelectedEntity(entity, false);
                                }
                            }
                            ImGui::PopStyleColor();
                            ImGui::PopStyleColor();
                            ImGui::PopStyleColor();
                            ImGui::TreePop();
                        }
                    });
            }
            else if (m_selectedHierarchyDisplayMode == 1) {
                ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.2, 0.3, 0.2, 1.0));
                ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.2, 0.2, 0.2, 1.0));
                ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.2, 0.2, 0.3, 1.0));
                scene->ForAllEntities([&](int i, Entity entity) {
                    if (scene->GetParent(entity).GetIndex() == 0)
                        DrawEntityNode(entity, 0);
                    });
                m_selectedEntityHierarchyList.clear();
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();
            }
        }
        ImGui::End();
    }
    if (scene && m_showEntityInspectorWindow) {
        ImGui::Begin("Entity Inspector");
        ImGui::Text("Selection:");
        ImGui::SameLine();
        ImGui::Checkbox("Lock", &m_lockEntitySelection);
        ImGui::SameLine();
        ImGui::Checkbox("Highlight", &m_highlightSelection);
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            SetSelectedEntity({});
        }
        ImGui::Separator();
        if (scene->IsEntityValid(m_selectedEntity)) {
            std::string title = std::to_string(m_selectedEntity.GetIndex()) + ": ";
            title += scene->GetEntityName(m_selectedEntity);
            bool enabled = scene->IsEntityEnabled(m_selectedEntity);
            if (ImGui::Checkbox((title + "##EnabledCheckbox").c_str(), &enabled)) {
                if (scene->IsEntityEnabled(m_selectedEntity) != enabled) {
                    scene->SetEnable(m_selectedEntity, enabled);
                }
            }
            ImGui::SameLine();
            bool isStatic = scene->IsEntityStatic(m_selectedEntity);
            if (ImGui::Checkbox("Static##StaticCheckbox", &isStatic)) {
                if (scene->IsEntityStatic(m_selectedEntity) != isStatic) {
                    scene->SetEntityStatic(m_selectedEntity, enabled);
                }
            }

            bool deleted = DrawEntityMenu(scene->IsEntityEnabled(m_selectedEntity), m_selectedEntity);
            if (!deleted) {
                if (ImGui::CollapsingHeader("Data components", ImGuiTreeNodeFlags_DefaultOpen)) {
                    if (ImGui::BeginPopupContextItem("DataComponentInspectorPopup")) {
                        ImGui::Text("Add data component: ");
                        ImGui::Separator();
                        for (auto& i : m_componentDataMenuList) {
                            i.second(m_selectedEntity);
                        }
                        ImGui::Separator();
                        ImGui::EndPopup();
                    }
                    bool skip = false;
                    int i = 0;
                    scene->UnsafeForEachDataComponent(m_selectedEntity, [&](DataComponentType type, void* data) {
                        if (skip)
                            return;
                        std::string info = type.m_name;
                        info += " Size: " + std::to_string(type.m_size);
                        ImGui::Text(info.c_str());
                        ImGui::PushID(i);
                        if (ImGui::BeginPopupContextItem(
                            ("DataComponentDeletePopup" + std::to_string(i)).c_str())) {
                            if (ImGui::Button("Remove")) {
                                skip = true;
                                scene->RemoveDataComponent(m_selectedEntity, type.m_typeId);
                            }
                            ImGui::EndPopup();
                        }
                        ImGui::PopID();
                        InspectComponentData(
                            m_selectedEntity,
                            static_cast<IDataComponent*>(data),
                            type,
                            scene->GetParent(m_selectedEntity).GetIndex() != 0);
                        ImGui::Separator();
                        i++;
                        });
                }

                if (ImGui::CollapsingHeader("Private components", ImGuiTreeNodeFlags_DefaultOpen)) {
                    if (ImGui::BeginPopupContextItem("PrivateComponentInspectorPopup")) {
                        ImGui::Text("Add private component: ");
                        ImGui::Separator();
                        for (auto& i : m_privateComponentMenuList) {
                            i.second(m_selectedEntity);
                        }
                        ImGui::Separator();
                        ImGui::EndPopup();
                    }

                    int i = 0;
                    bool skip = false;
                    scene->ForEachPrivateComponent(m_selectedEntity, [&](PrivateComponentElement& data) {
                        if (skip)
                            return;
                        ImGui::Checkbox(
                            data.m_privateComponentData->GetTypeName().c_str(),
                            &data.m_privateComponentData->m_enabled);
                        EditorLayer::DraggablePrivateComponent(data.m_privateComponentData);
                        const std::string tag = "##" + data.m_privateComponentData->GetTypeName() +
                            std::to_string(data.m_privateComponentData->GetHandle());
                        if (ImGui::BeginPopupContextItem(tag.c_str())) {
                            if (ImGui::Button(("Remove" + tag).c_str())) {
                                skip = true;
                                scene->RemovePrivateComponent(m_selectedEntity, data.m_typeId);
                            }
                            ImGui::EndPopup();
                        }
                        if (!skip) {
                            if (ImGui::TreeNodeEx(
                                ("Component Settings##" + std::to_string(i)).c_str(),
                                ImGuiTreeNodeFlags_DefaultOpen)) {
                                data.m_privateComponentData->OnInspect(editorLayer);
                                ImGui::TreePop();
                            }
                        }
                        ImGui::Separator();
                        i++;
                        });
                }
            }
        }
        else {
            m_selectedEntity = Entity();
        }
        ImGui::End();
    }

    if (scene && m_sceneCameraWindowFocused && windowLayer->GetKey(GLFW_KEY_DELETE))
    {
        if (scene->IsEntityValid(m_selectedEntity))
        {
            scene->DeleteEntity(m_selectedEntity);
        }
    }

    //if (m_showSceneWindow) SceneCameraWindow();
    //if (m_showCameraWindow) MainCameraWindow();
}

void EditorLayer::LateUpdate()
{
	const auto& layers = Application::GetLayers();
	for (const auto& layer : layers) layer->OnInspect(Application::GetLayer<EditorLayer>());


	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer)
		{
			const auto extent2D = Graphics::GetSwapchain().GetVkExtent2D();
			VkRenderPassBeginInfo renderPassBeginInfo{};
			renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassBeginInfo.renderPass = m_renderPass.GetVkRenderPass();
			renderPassBeginInfo.framebuffer = m_framebuffers[Graphics::GetNextImageIndex()].GetVkFrameBuffer();
			renderPassBeginInfo.renderArea.offset = { 0, 0 };
			renderPassBeginInfo.renderArea.extent = extent2D;

			VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
			renderPassBeginInfo.clearValueCount = 1;
			renderPassBeginInfo.pClearValues = &clearColor;

			vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			ImGui::Render();
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

			vkCmdEndRenderPass(commandBuffer);
		});
}

bool EditorLayer::DrawEntityMenu(const bool& enabled, const Entity& entity)
{
    bool deleted = false;
    if (ImGui::BeginPopupContextItem(std::to_string(entity.GetIndex()).c_str())) {
	    const auto scene = GetScene();
        ImGui::Text(("Handle: " + std::to_string(scene->GetEntityHandle(entity).GetValue())).c_str());
        if (ImGui::Button("Delete")) {
            scene->DeleteEntity(entity);
            deleted = true;
        }
        if (!deleted && ImGui::Button(enabled ? "Disable" : "Enable")) {
            if (enabled) {
                scene->SetEnable(entity, false);
            }
            else {
                scene->SetEnable(entity, true);
            }
        }
        const std::string tag = "##Entity" + std::to_string(scene->GetEntityHandle(entity));
        if (!deleted && ImGui::BeginMenu(("Rename" + tag).c_str())) {
            static char newName[256];
            ImGui::InputText("New name", newName, 256);
            if (ImGui::Button("Confirm")) {
                scene->SetEntityName(entity, std::string(newName));
                memset(newName, 0, 256);
            }
            ImGui::EndMenu();
        }
        ImGui::EndPopup();
    }
    return deleted;
}

void EditorLayer::DrawEntityNode(const Entity& entity, const unsigned& hierarchyLevel)
{
    auto scene = GetScene();
    std::string title = std::to_string(entity.GetIndex()) + ": ";
    title += scene->GetEntityName(entity);
    const bool enabled = scene->IsEntityEnabled(entity);
    if (enabled) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4({ 1, 1, 1, 1 }));
    }
    const int index = m_selectedEntityHierarchyList.size() - hierarchyLevel - 1;
    if (!m_selectedEntityHierarchyList.empty() && index >= 0 && index < m_selectedEntityHierarchyList.size() &&
        m_selectedEntityHierarchyList[index] == entity) {
        ImGui::SetNextItemOpen(true);
    }
    const bool opened = ImGui::TreeNodeEx(
        title.c_str(),
        ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_OpenOnArrow |
        ImGuiTreeNodeFlags_NoAutoOpenOnLog |
        (m_selectedEntity == entity ? ImGuiTreeNodeFlags_Framed : ImGuiTreeNodeFlags_FramePadding));
    if (ImGui::BeginDragDropSource()) {
        auto handle = scene->GetEntityHandle(entity);
        ImGui::SetDragDropPayload("Entity", &handle, sizeof(Handle));
        ImGui::TextColored(ImVec4(0, 0, 1, 1), title.c_str());
        ImGui::EndDragDropSource();
    }
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Entity")) {
            IM_ASSERT(payload->DataSize == sizeof(Handle));
            scene->SetParent(scene->GetEntity(*static_cast<Handle*>(payload->Data)), entity, true);
        }
        ImGui::EndDragDropTarget();
    }
    if (enabled) {
        ImGui::PopStyleColor();
    }
    if (!m_lockEntitySelection && ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
        SetSelectedEntity(entity, false);
    }
    const bool deleted = DrawEntityMenu(enabled, entity);
    if (opened && !deleted) {
        ImGui::TreePush();
        scene->ForEachChild(
            entity, [=](Entity child) {
                DrawEntityNode(child, hierarchyLevel + 1);
            });
        ImGui::TreePop();
    }
}

void EditorLayer::InspectComponentData(Entity entity, IDataComponent* data, DataComponentType type, bool isRoot)
{
    if (m_componentDataInspectorMap.find(type.m_typeId) !=
        m_componentDataInspectorMap.end()) {
        if (m_componentDataInspectorMap.at(type.m_typeId)(entity, data, isRoot)) {
            auto scene = GetScene();
            scene->SetUnsaved();
        }
    }
}

void EditorLayer::CreateRenderPass()
{
	VkAttachmentDescription colorAttachment{};
	colorAttachment.format = Graphics::GetSwapchain().GetVkFormat();
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorAttachmentRef{};
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	VkSubpassDependency dependency{};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = 1;
	renderPassInfo.pDependencies = &dependency;

	m_renderPass.Create(renderPassInfo);
}

bool EditorLayer::UpdateFrameBuffers()
{
	const auto currentSwapchainVersion = Graphics::GetSwapchainVersion();
	if (currentSwapchainVersion == m_storedSwapchainVersion) return false;

	m_storedSwapchainVersion = currentSwapchainVersion;
	const auto& swapChain = Graphics::GetSwapchain();
	const auto& swapChainImageViews = swapChain.GetVkImageViews();
	m_framebuffers.resize(swapChainImageViews.size());
	for (size_t i = 0; i < swapChainImageViews.size(); i++) {
		const VkImageView attachments[] = { swapChainImageViews[i] };
		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = m_renderPass.GetVkRenderPass();
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = attachments;
		framebufferInfo.width = swapChain.GetVkExtent2D().width;
		framebufferInfo.height = swapChain.GetVkExtent2D().height;
		framebufferInfo.layers = 1;
		m_framebuffers[i].Create(framebufferInfo);
	}

	return true;
}

void EditorLayer::SetSelectedEntity(const Entity& entity, bool openMenu)
{
    if (entity == m_selectedEntity)
        return;
    m_selectedEntityHierarchyList.clear();
    if (entity.GetIndex() == 0) {
        m_selectedEntity = Entity();
        m_lockEntitySelection = false;
        return;
    }
    const auto scene = GetScene();
    if (!scene->IsEntityValid(entity))
        return;
    m_selectedEntity = entity;
    if (!openMenu)
        return;
    auto walker = entity;
    while (walker.GetIndex() != 0) {
        m_selectedEntityHierarchyList.push_back(walker);
        walker = scene->GetParent(walker);
    }
}


bool EditorLayer::UnsafeDroppableAsset(AssetRef& target, const std::vector<std::string>& typeNames)
{
    bool statusChanged = false;
    if (ImGui::BeginDragDropTarget())
    {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Asset"))
        {
            const std::shared_ptr<IAsset> ptr = target.Get<IAsset>();
            IM_ASSERT(payload->DataSize == sizeof(Handle));
            Handle payload_n = *static_cast<Handle*>(payload->Data);
            if (!ptr || payload_n.GetValue() != target.GetAssetHandle().GetValue())
            {
                auto asset = ProjectManager::GetAsset(payload_n);
                for (const auto& typeName : typeNames)
                {
                    if (asset && asset->GetTypeName() == typeName)
                    {
                        target.Clear();
                        target.m_assetHandle = payload_n;
                        target.Update();
                        statusChanged = true;
                        break;
                    }
                }
            }
        }
        ImGui::EndDragDropTarget();
    }
    return statusChanged;
}

bool EditorLayer::UnsafeDroppablePrivateComponent(PrivateComponentRef& target, const std::vector<std::string>& typeNames)
{
    bool statusChanged = false;
    if (ImGui::BeginDragDropTarget())
    {
        const auto currentScene = Application::GetActiveScene();
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Entity"))
        {
            IM_ASSERT(payload->DataSize == sizeof(Handle));
            auto payload_n = *static_cast<Handle*>(payload->Data);
            auto entity = currentScene->GetEntity(payload_n);
            if (currentScene->IsEntityValid(entity))
            {
                for (const auto& typeName : typeNames)
                {
                    if (currentScene->HasPrivateComponent(entity, typeName))
                    {
                        const auto ptr = target.Get<IPrivateComponent>();
                        auto newPrivateComponent = currentScene->GetPrivateComponent(entity, typeName).lock();
                        target = newPrivateComponent;
                        statusChanged = true;
                        break;
                    }
                }
            }
        }
        else if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("PrivateComponent"))
        {

            IM_ASSERT(payload->DataSize == sizeof(Handle));
            auto payload_n = *static_cast<Handle*>(payload->Data);
            auto entity = currentScene->GetEntity(payload_n);
            for (const auto& typeName : typeNames)
            {
                if (currentScene->HasPrivateComponent(entity, typeName))
                {
                    target = currentScene->GetPrivateComponent(entity, typeName).lock();
                    statusChanged = true;
                    break;
                }
            }
        }
        ImGui::EndDragDropTarget();
    }
    return statusChanged;
}

std::map<std::string, std::shared_ptr<Texture2D>>& EditorLayer::AssetIcons()
{
    return m_assetsIcons;
}

bool EditorLayer::DragAndDropButton(EntityRef& entityRef, const std::string& name, bool modifiable)
{
    ImGui::Text(name.c_str());
    ImGui::SameLine();
    bool statusChanged = false;
    auto entity = entityRef.Get();
    if (entity.GetIndex() != 0)
    {
        auto scene = Application::GetActiveScene();
        ImGui::Button(scene->GetEntityName(entity).c_str());
        Draggable(entityRef);
        if (modifiable)
        {
            statusChanged = Rename(entityRef);
            statusChanged = Remove(entityRef) || statusChanged;
        }
    }
    else
    {
        ImGui::Button("none");
    }
    statusChanged = Droppable(entityRef) || statusChanged;
    return statusChanged;
}
bool EditorLayer::Droppable(EntityRef& entityRef)
{
    bool statusChanged = false;
    if (ImGui::BeginDragDropTarget())
    {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Entity"))
        {
            auto scene = Application::GetActiveScene();
            IM_ASSERT(payload->DataSize == sizeof(Handle));
            auto payload_n = *static_cast<Handle*>(payload->Data);
            auto newEntity = scene->GetEntity(payload_n);
            if (scene->IsEntityValid(newEntity))
            {
                entityRef = newEntity;
                statusChanged = true;
            }
        }
        ImGui::EndDragDropTarget();
    }
    return statusChanged;
}
void EditorLayer::Draggable(EntityRef& entityRef)
{
    auto entity = entityRef.Get();
    if (entity.GetIndex() != 0)
    {
        DraggableEntity(entity);
    }
}
void EditorLayer::DraggableEntity(const Entity& entity)
{
    if (ImGui::BeginDragDropSource())
    {
        auto scene = Application::GetActiveScene();
        auto handle = scene->GetEntityHandle(entity);
        ImGui::SetDragDropPayload("Entity", &handle, sizeof(Handle));
        ImGui::TextColored(ImVec4(0, 0, 1, 1), scene->GetEntityName(entity).c_str());
        ImGui::EndDragDropSource();
    }
}
bool EditorLayer::Rename(EntityRef& entityRef)
{
    auto entity = entityRef.Get();
    bool statusChanged = RenameEntity(entity);
    return statusChanged;
}
bool EditorLayer::Remove(EntityRef& entityRef)
{
    bool statusChanged = false;
    auto entity = entityRef.Get();
    auto scene = Application::GetActiveScene();
    if (scene->IsEntityValid(entity))
    {
        const std::string tag = "##Entity" + std::to_string(scene->GetEntityHandle(entity));
        if (ImGui::BeginPopupContextItem(tag.c_str()))
        {
            if (ImGui::Button(("Remove" + tag).c_str()))
            {
                entityRef.Clear();
                statusChanged = true;
            }
            ImGui::EndPopup();
        }
    }
    return statusChanged;
}
bool EditorLayer::RenameEntity(const Entity& entity) const
{
    bool statusChanged = false;
    auto scene = Application::GetActiveScene();
    if (scene->IsEntityValid(entity))
    {
        const std::string tag = "##Entity" + std::to_string(scene->GetEntityHandle(entity));
        if (ImGui::BeginPopupContextItem(tag.c_str()))
        {
            if (ImGui::BeginMenu(("Rename" + tag).c_str()))
            {
                static char newName[256];
                ImGui::InputText(("New name" + tag).c_str(), newName, 256);
                if (ImGui::Button(("Confirm" + tag).c_str()))
                {
                    scene->SetEntityName(entity, std::string(newName));
                    memset(newName, 0, 256);
                }
                ImGui::EndMenu();
            }
            ImGui::EndPopup();
        }
    }
    return statusChanged;
}

bool EditorLayer::DragAndDropButton(
    AssetRef& target, const std::string& name, const std::vector<std::string>& acceptableTypeNames, bool modifiable)
{
    ImGui::Text(name.c_str());
    ImGui::SameLine();
    auto ptr = target.Get<IAsset>();
    bool statusChanged = false;
    if (ptr)
    {
        const auto title = ptr->GetTitle();
        ImGui::Button(title.c_str());
        DraggableAsset(ptr);
        if (modifiable)
        {
            statusChanged = Rename(target);
            statusChanged = Remove(target) || statusChanged;
        }
        if (!statusChanged && ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
        {
            ProjectManager::GetInstance().m_inspectingAsset = ptr;
        }
    }
    else
    {
        ImGui::Button("none");
    }
    statusChanged = UnsafeDroppableAsset(target, acceptableTypeNames) || statusChanged;
    return statusChanged;
}
bool EditorLayer::DragAndDropButton(
    PrivateComponentRef& target,
    const std::string& name,
    const std::vector<std::string>& acceptableTypeNames,
    bool modifiable)
{
    ImGui::Text(name.c_str());
    ImGui::SameLine();
    bool statusChanged = false;
    auto ptr = target.Get<IPrivateComponent>();
    if (ptr)
    {
        auto scene = Application::GetActiveScene();
        ImGui::Button(scene->GetEntityName(ptr->GetOwner()).c_str());
        const std::string tag = "##" + ptr->GetTypeName() + std::to_string(ptr->GetHandle());
        DraggablePrivateComponent(ptr);
        if (modifiable)
        {
            statusChanged = Remove(target);
        }
    }
    else
    {
        ImGui::Button("none");
    }
    statusChanged = UnsafeDroppablePrivateComponent(target, acceptableTypeNames) || statusChanged;
    return statusChanged;
}