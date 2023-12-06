#include "EditorLayer.hpp"
#include "ILayer.hpp"
#include "Application.hpp"
#include "Prefab.hpp"
#include "Graphics.hpp"
#include "Material.hpp"
#include "Mesh.hpp"
#include "MeshRenderer.hpp"
#include "ProjectManager.hpp"
#include "WindowLayer.hpp"
#include "Scene.hpp"
#include "Times.hpp"
#include "RenderLayer.hpp"
#include "Cubemap.hpp"
#include "EnvironmentalMap.hpp"
#include "StrandsRenderer.hpp"
using namespace EvoEngine;

void EditorLayer::OnCreate()
{
	if (Application::GetLayer<WindowLayer>())
	{
		std::runtime_error("EditorLayer requires WindowLayer!");
	}

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
		const auto scene = Application::GetActiveScene();
		const auto status = scene->GetDataComponent<TransformUpdateFlag>(entity);
		const bool reload = previousEntity != entity || m_transformReload;// || status.m_transformModified || status.m_globalTransformModified;
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
			reload ? ImGuiSliderFlags_ReadOnly : 0))
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
			reload ? ImGuiSliderFlags_ReadOnly : 0))
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
			reload ? ImGuiSliderFlags_ReadOnly : 0))
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
		m_transformReload = false;
		m_transformReadOnly = false;
		return edited;
		});

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

	LoadIcons();

	VkBufferCreateInfo entityIndexReadBuffer{};
	entityIndexReadBuffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	entityIndexReadBuffer.size = sizeof(glm::detail::hdata) * 4;
	entityIndexReadBuffer.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	entityIndexReadBuffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VmaAllocationCreateInfo entityIndexReadBufferCreateInfo{};
	entityIndexReadBufferCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
	entityIndexReadBufferCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	m_entityIndexReadBuffer = std::make_unique<Buffer>(entityIndexReadBuffer, entityIndexReadBufferCreateInfo);
	vmaMapMemory(Graphics::GetVmaAllocator(), m_entityIndexReadBuffer->GetVmaAllocation(), static_cast<void**>(static_cast<void*>(&m_mappedEntityIndexData)));


	const auto sceneCamera = Serialization::ProduceSerializable<Camera>();
	sceneCamera->m_clearColor = glm::vec3(59.0f / 255.0f, 85 / 255.0f, 143 / 255.f);
	sceneCamera->m_useClearColor = false;
	sceneCamera->OnCreate();
	RegisterEditorCamera(sceneCamera);
	m_sceneCameraHandle = sceneCamera->GetHandle();
}

void EditorLayer::OnDestroy()
{
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void EditorLayer::PreUpdate()
{
	m_gizmoMeshTasks.clear();
	m_gizmoInstancedMeshTasks.clear();
	m_gizmoStrandsTasks.clear();

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
		switch (Application::GetApplicationStatus()) {
		case ApplicationStatus::Stop: {
			if (ImGui::ImageButton(
				m_assetsIcons["PlayButton"]->GetImTextureId(),
				{ 15, 15 },
				{ 0, 1 },
				{ 1, 0 })) {
				Application::Play();
			}
			if (ImGui::ImageButton(
				m_assetsIcons["StepButton"]->GetImTextureId(),
				{ 15, 15 },
				{ 0, 1 },
				{ 1, 0 })) {
				Application::Step();
			}
			break;
		}
		case ApplicationStatus::Playing: {
			if (ImGui::ImageButton(
				m_assetsIcons["PauseButton"]->GetImTextureId(),
				{ 15, 15 },
				{ 0, 1 },
				{ 1, 0 })) {
				Application::Pause();
			}
			if (ImGui::ImageButton(
				m_assetsIcons["StopButton"]->GetImTextureId(),
				{ 15, 15 },
				{ 0, 1 },
				{ 1, 0 })) {
				Application::Stop();
			}
			break;
		}
		case ApplicationStatus::Pause: {
			if (ImGui::ImageButton(
				m_assetsIcons["PlayButton"]->GetImTextureId(),
				{ 15, 15 },
				{ 0, 1 },
				{ 1, 0 })) {
				Application::Play();
			}
			if (ImGui::ImageButton(
				m_assetsIcons["StepButton"]->GetImTextureId(),
				{ 15, 15 },
				{ 0, 1 },
				{ 1, 0 })) {
				Application::Step();
			}
			if (ImGui::ImageButton(
				m_assetsIcons["StopButton"]->GetImTextureId(),
				{ 15, 15 },
				{ 0, 1 },
				{ 1, 0 })) {
				Application::Stop();
			}
			break;
		}
		}

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


	m_mouseSceneWindowPosition = glm::vec2(FLT_MAX, FLT_MIN);
	if (m_showSceneWindow) {
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
		if (ImGui::Begin("Scene")) {
			if (ImGui::BeginChild("SceneCameraRenderer", ImVec2(0, 0), false)) {
				// Using a Child allow to fill all the space of the window.
				// It also allows customization
				if (m_sceneCameraWindowFocused) {
					auto mp = ImGui::GetMousePos();
					auto wp = ImGui::GetWindowPos();
					m_mouseSceneWindowPosition = glm::vec2(mp.x - wp.x, mp.y - wp.y);
				}
			}
			ImGui::EndChild();
		}
		ImGui::End();
		ImGui::PopStyleVar();
	}

	m_mouseCameraWindowPosition = glm::vec2(FLT_MAX, FLT_MIN);
	if (m_showCameraWindow) {
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
		if (ImGui::Begin("Camera")) {
			if (ImGui::BeginChild("MainCameraRenderer", ImVec2(0, 0), false)) {
				// Using a Child allow to fill all the space of the window.
				// It also allows customization
				if (m_mainCameraWindowFocused) {
					auto mp = ImGui::GetMousePos();
					auto wp = ImGui::GetWindowPos();
					m_mouseCameraWindowPosition = glm::vec2(mp.x - wp.x, mp.y - wp.y);
				}
			}
			ImGui::EndChild();
		}
		ImGui::End();
		ImGui::PopStyleVar();
	}
	if (m_showSceneWindow) ResizeCameras();

	if (!m_mainCameraWindowFocused)
	{
		const auto activeScene = Application::GetActiveScene();
		auto& pressedKeys = activeScene->m_pressedKeys;
		pressedKeys.clear();
	}

	if (m_applyTransformToMainCamera && !Application::IsPlaying())
	{
		const auto scene = Application::GetActiveScene();
		if (const auto camera = scene->m_mainCamera.Get<Camera>(); camera && scene->IsEntityValid(camera->GetOwner()))
		{
			auto& [sceneCameraRotation, sceneCameraPosition, sceneCamera] = m_editorCameras.at(m_sceneCameraHandle);
			GlobalTransform globalTransform;
			globalTransform.SetPosition(sceneCameraPosition);
			globalTransform.SetRotation(sceneCameraRotation);
			scene->SetDataComponent(camera->GetOwner(), globalTransform);
		}
	}
	const auto& scene = Application::GetActiveScene();
	if (!scene->IsEntityValid(m_selectedEntity))
	{
		SetSelectedEntity(Entity());
	}
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	if (renderLayer && renderLayer->m_needFade != 0 && m_selectionAlpha < 256)
	{
		m_selectionAlpha += Times::DeltaTime() * 5120;
	}

	m_selectionAlpha = glm::clamp(m_selectionAlpha, 0, 256);

}
static const char* HierarchyDisplayMode[]{ "Archetype", "Hierarchy" };
void EditorLayer::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	const auto& windowLayer = Application::GetLayer<WindowLayer>();
	if (!windowLayer) return;

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("View"))
		{
			if (ImGui::BeginMenu("Editor"))
			{
				if (ImGui::BeginMenu("Scene"))
				{
					ImGui::Checkbox("Show Scene Window", &m_showSceneWindow);
					if (m_showSceneWindow)
					{
						ImGui::Checkbox("Show Scene Window Info", &m_showSceneInfo);
						ImGui::Checkbox("View Gizmos", &m_enableViewGizmos);
					}
					ImGui::EndMenu();
				}
				if (ImGui::BeginMenu("Camera"))
				{
					ImGui::Checkbox("Show Camera Window", &m_showCameraWindow);
					if (m_showCameraWindow)
					{
						ImGui::Checkbox("Show Camera Window Info", &m_showCameraInfo);
					}
					ImGui::EndMenu();
				}
				ImGui::Checkbox("Show Entity Explorer", &m_showEntityExplorerWindow);
				ImGui::Checkbox("Show Entity Inspector", &m_showEntityInspectorWindow);
				ImGui::EndMenu();
			}
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	const auto scene = GetScene();
	if (scene && m_showEntityExplorerWindow) {
		ImGui::Begin("Entity Explorer");
		if (ImGui::BeginPopupContextWindow("NewEntityPopup")) {
			if (ImGui::Button("Create new entity")) {
				scene->CreateEntity(m_basicEntityArchetype);
			}
			ImGui::EndPopup();
		}
		ImGui::Combo(
			"Display mode", &m_selectedHierarchyDisplayMode, HierarchyDisplayMode,
			IM_ARRAYSIZE(HierarchyDisplayMode));
		std::string title = scene->GetTitle();
		if (ImGui::CollapsingHeader(title.c_str(),
			ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow)) {
			DraggableAsset(scene);
			RenameAsset(scene);
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
		ImGui::Checkbox("Focus", &m_highlightSelection);
		ImGui::SameLine();
		ImGui::Checkbox("Gizmos", &m_enableGizmos);
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
						DraggablePrivateComponent(data.m_privateComponentData);
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
			SetSelectedEntity(Entity());
		}
		ImGui::End();
	}

	if (scene && m_sceneCameraWindowFocused && Input::GetKey(GLFW_KEY_DELETE) == KeyActionType::Press)
	{
		if (scene->IsEntityValid(m_selectedEntity))
		{
			scene->DeleteEntity(m_selectedEntity);
		}
	}

	if (m_showSceneWindow) SceneCameraWindow();
	if (m_showCameraWindow) MainCameraWindow();

	ProjectManager::OnInspect(editorLayer);
	Resources::OnInspect(editorLayer);
}

void EditorLayer::LateUpdate()
{
	if (m_lockCamera) {
		auto& [sceneCameraRotation, sceneCameraPosition, sceneCamera] = m_editorCameras.at(m_sceneCameraHandle);
		const float elapsedTime = Times::Now() - m_transitionTimer;
		float a = 1.0f - glm::pow(1.0 - elapsedTime / m_transitionTime, 4.0f);
		if (elapsedTime >= m_transitionTime)
			a = 1.0f;
		sceneCameraRotation = glm::mix(m_previousRotation, m_targetRotation, a);
		sceneCameraPosition = glm::mix(m_previousPosition, m_targetPosition, a);
		if (a >= 1.0f) {
			m_lockCamera = false;
			sceneCameraRotation = m_targetRotation;
			sceneCameraPosition = m_targetPosition;
			//Camera::ReverseAngle(m_targetRotation, m_sceneCameraPitchAngle, m_sceneCameraYawAngle);
		}
	}

	Graphics::AppendCommands([&](VkCommandBuffer commandBuffer)
		{
			Graphics::EverythingBarrier(commandBuffer);
			Graphics::TransitImageLayout(commandBuffer,
				Graphics::GetSwapchain()->GetVkImage(), Graphics::GetSwapchain()->GetImageFormat(), 1,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR);

			constexpr VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
			VkRect2D renderArea;
			renderArea.offset = { 0, 0 };
			renderArea.extent = Graphics::GetSwapchain()->GetImageExtent();


			VkRenderingAttachmentInfo colorAttachmentInfo{};
			colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
			colorAttachmentInfo.imageView = Graphics::GetSwapchain()->GetVkImageView();
			colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
			colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachmentInfo.clearValue = clearColor;

			VkRenderingInfo renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = renderArea;
			renderInfo.layerCount = 1;
			renderInfo.colorAttachmentCount = 1;
			renderInfo.pColorAttachments = &colorAttachmentInfo;

			vkCmdBeginRendering(commandBuffer, &renderInfo);
			ImGui::Render();
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

			vkCmdEndRendering(commandBuffer);
			Graphics::TransitImageLayout(commandBuffer,
				Graphics::GetSwapchain()->GetVkImage(), Graphics::GetSwapchain()->GetImageFormat(), 1,
				VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);


		});
}

bool EditorLayer::DrawEntityMenu(const bool& enabled, const Entity& entity) const
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
		ImGui::TreePush(title.c_str());
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

void EditorLayer::SceneCameraWindow()
{
	const auto scene = GetScene();
	auto windowLayer = Application::GetLayer<WindowLayer>();
	const auto& graphics = Graphics::GetInstance();
	auto& [sceneCameraRotation, sceneCameraPosition, sceneCamera] = m_editorCameras.at(m_sceneCameraHandle);
#pragma region Scene Window
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
	if (ImGui::Begin("Scene")) {
		ImVec2 viewPortSize;
		// Using a Child allow to fill all the space of the window.
		// It also allows customization
		static int corner = 1;
		if (ImGui::BeginChild("SceneCameraRenderer", ImVec2(0, 0), false)) {
			viewPortSize = ImGui::GetWindowSize();
			m_sceneCameraResolutionX = viewPortSize.x * m_sceneCameraResolutionMultiplier;
			m_sceneCameraResolutionY = viewPortSize.y * m_sceneCameraResolutionMultiplier;
			const ImVec2 overlayPos = ImGui::GetWindowPos();
			if (sceneCamera && sceneCamera->m_rendered) {
				// Because I use the texture from OpenGL, I need to invert the V from the UV.
				ImGui::Image(sceneCamera->GetRenderTexture()->GetColorImTextureId(),
					ImVec2(viewPortSize.x, viewPortSize.y),
					ImVec2(0, 1),
					ImVec2(1, 0));
				CameraWindowDragAndDrop();
			}
			else {
				ImGui::Text("No active scene camera!");
			}
			const auto windowPos = ImVec2(
				(corner & 1) ? (overlayPos.x + viewPortSize.x) : (overlayPos.x),
				(corner & 2) ? (overlayPos.y + viewPortSize.y) : (overlayPos.y));

			if (m_showSceneInfo) {
				const auto windowPosPivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
				ImGui::SetNextWindowPos(windowPos, ImGuiCond_Always, windowPosPivot);
				ImGui::SetNextWindowBgAlpha(0.35f);
				constexpr auto windowFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDocking |
					ImGuiWindowFlags_AlwaysAutoResize |
					ImGuiWindowFlags_NoSavedSettings |
					ImGuiWindowFlags_NoFocusOnAppearing;
				if (ImGui::BeginChild("Info", ImVec2(200, 350), true, windowFlags)) {
					ImGui::Text("Info & Settings");
					ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
					std::string drawCallInfo = {};
					const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
					if (graphics.m_triangles[currentFrameIndex] < 999)
						drawCallInfo += std::to_string(graphics.m_triangles[currentFrameIndex]);
					else if (graphics.m_triangles[currentFrameIndex] < 999999)
						drawCallInfo += std::to_string((int)(graphics.m_triangles[currentFrameIndex] / 1000)) + "K";
					else
						drawCallInfo += std::to_string((int)(graphics.m_triangles[currentFrameIndex] / 1000000)) + "M";
					drawCallInfo += " tris";
					ImGui::Text(drawCallInfo.c_str());
					ImGui::Text("%d drawcall", graphics.m_drawCall[currentFrameIndex]);
					ImGui::Text("Idle: %.3f", graphics.m_cpuWaitTime);
					ImGui::Separator();
					if (ImGui::IsMousePosValid()) {
						const auto pos = Input::GetMousePosition();
						ImGui::Text("Mouse Pos: (%.1f,%.1f)", pos.x, pos.y);
					}
					else {
						ImGui::Text("Mouse Pos: <invalid>");
					}

					if (ImGui::Button("Reset camera")) {
						MoveCamera(m_defaultSceneCameraRotation, m_defaultSceneCameraPosition);
					}
					if (ImGui::Button("Set default")) {
						m_defaultSceneCameraPosition = sceneCameraPosition;
						m_defaultSceneCameraRotation = sceneCameraRotation;
					}
					ImGui::PushItemWidth(100);
					ImGui::Checkbox("Use clear color", &sceneCamera->m_useClearColor);
					ImGui::ColorEdit3("Clear color", &sceneCamera->m_clearColor.x);
					ImGui::SliderFloat("FOV", &sceneCamera->m_fov, 1.0f, 359.f, "%.1f");
					ImGui::DragFloat3("Position", &sceneCameraPosition.x, 0.1f, 0, 0, "%.1f");
					ImGui::DragFloat("Speed", &m_velocity, 0.1f, 0, 0, "%.1f");
					ImGui::DragFloat("Sensitivity", &m_sensitivity, 0.1f, 0, 0, "%.1f");
					ImGui::Checkbox("Copy Transform", &m_applyTransformToMainCamera);
					ImGui::DragFloat("Resolution", &m_sceneCameraResolutionMultiplier, 0.1f, 0.1f, 4.0f);
					DragAndDropButton<Cubemap>(sceneCamera->m_skybox, "Skybox", true);
					ImGui::PopItemWidth();
				}
				ImGui::EndChild();
			}
			if (m_sceneCameraWindowFocused) {
#pragma region Scene Camera Controller
				static bool isDraggingPreviously = false;
				bool mouseDrag = true;
				if (m_mouseSceneWindowPosition.x < 0 || m_mouseSceneWindowPosition.y < 0 ||
					m_mouseSceneWindowPosition.x > viewPortSize.x ||
					m_mouseSceneWindowPosition.y > viewPortSize.y ||
					Input::GetKey(GLFW_MOUSE_BUTTON_RIGHT) != KeyActionType::Hold) {
					mouseDrag = false;
				}
				static float prevX = 0;
				static float prevY = 0;
				if (mouseDrag && !isDraggingPreviously) {
					prevX = m_mouseSceneWindowPosition.x;
					prevY = m_mouseSceneWindowPosition.y;
				}
				const float xOffset = m_mouseSceneWindowPosition.x - prevX;
				const float yOffset = m_mouseSceneWindowPosition.y - prevY;
				prevX = m_mouseSceneWindowPosition.x;
				prevY = m_mouseSceneWindowPosition.y;
				isDraggingPreviously = mouseDrag;

				if (mouseDrag && !m_lockCamera) {
					glm::vec3 front = sceneCameraRotation * glm::vec3(0, 0, -1);
					const glm::vec3 right = sceneCameraRotation * glm::vec3(1, 0, 0);
					if (Input::GetKey(GLFW_KEY_W) == KeyActionType::Hold) {
						sceneCameraPosition +=
							front * static_cast<float>(Times::DeltaTime()) * m_velocity;
					}
					if (Input::GetKey(GLFW_KEY_S) == KeyActionType::Hold) {
						sceneCameraPosition -=
							front * static_cast<float>(Times::DeltaTime()) * m_velocity;
					}
					if (Input::GetKey(GLFW_KEY_A) == KeyActionType::Hold) {
						sceneCameraPosition -=
							right * static_cast<float>(Times::DeltaTime()) * m_velocity;
					}
					if (Input::GetKey(GLFW_KEY_D) == KeyActionType::Hold) {
						sceneCameraPosition +=
							right * static_cast<float>(Times::DeltaTime()) * m_velocity;
					}
					if (Input::GetKey(GLFW_KEY_LEFT_SHIFT) == KeyActionType::Hold) {
						sceneCameraPosition.y += m_velocity * static_cast<float>(Times::DeltaTime());
					}
					if (Input::GetKey(GLFW_KEY_LEFT_CONTROL) == KeyActionType::Hold) {
						sceneCameraPosition.y -= m_velocity * static_cast<float>(Times::DeltaTime());
					}
					if (xOffset != 0.0f || yOffset != 0.0f) {
						front = glm::rotate(front, glm::radians( - xOffset * m_sensitivity), glm::vec3(0, 1, 0));
						const glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0.0f, 1.0f, 0.0f)));
						if ((front.y < 0.99f && yOffset < 0.0f) || (front.y > -0.99f && yOffset > 0.0f)) {
							front = glm::rotate(front, glm::radians(-yOffset * m_sensitivity), right);
						}
						const glm::vec3 up = glm::normalize(glm::cross(right, front));
						sceneCameraRotation = glm::quatLookAt(front, up);
					}
#pragma endregion
				}
			}

		}
#pragma region Gizmos and Entity Selection
		bool mouseSelectEntity = true;
		if (m_enableGizmos) {
			ImGuizmo::SetOrthographic(false);
			ImGuizmo::SetDrawlist();
			float viewManipulateLeft = ImGui::GetWindowPos().x;
			float viewManipulateTop = ImGui::GetWindowPos().y;
			ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, viewPortSize.x,
				viewPortSize.y);
			glm::mat4 cameraView =
				glm::inverse(glm::translate(sceneCameraPosition) * glm::mat4_cast(sceneCameraRotation));
			glm::mat4 cameraProjection = sceneCamera->GetProjection();
			const auto op = m_localPositionSelected ? ImGuizmo::OPERATION::TRANSLATE
				: m_localRotationSelected ? ImGuizmo::OPERATION::ROTATE
				: ImGuizmo::OPERATION::SCALE;
			if (scene->IsEntityValid(m_selectedEntity)) {
				auto transform = scene->GetDataComponent<Transform>(m_selectedEntity);
				GlobalTransform parentGlobalTransform;
				Entity parentEntity = scene->GetParent(m_selectedEntity);
				if (parentEntity.GetIndex() != 0) {
					parentGlobalTransform = scene->GetDataComponent<GlobalTransform>(
						scene->GetParent(m_selectedEntity));
				}
				auto globalTransform = scene->GetDataComponent<GlobalTransform>(m_selectedEntity);

				ImGuizmo::Manipulate(
					glm::value_ptr(cameraView),
					glm::value_ptr(cameraProjection),
					op,
					ImGuizmo::LOCAL,
					glm::value_ptr(globalTransform.m_value));
				if (ImGuizmo::IsUsing()) {
					transform.m_value = glm::inverse(parentGlobalTransform.m_value) * globalTransform.m_value;
					scene->SetDataComponent(m_selectedEntity, transform);
					transform.Decompose(
						m_previouslyStoredPosition, m_previouslyStoredRotation, m_previouslyStoredScale);
					mouseSelectEntity = false;
				}
			}
			if (m_enableViewGizmos) {
				if (ImGuizmo::ViewManipulate(glm::value_ptr(cameraView), 1.0f, ImVec2(viewManipulateLeft, viewManipulateTop), ImVec2(96, 96), 0)) {
					GlobalTransform gl;
					gl.m_value = glm::inverse(cameraView);
					sceneCameraRotation = gl.GetRotation();
				}
			}
		}
		if (m_sceneCameraWindowFocused && !m_lockEntitySelection && Input::GetKey(GLFW_KEY_ESCAPE) == KeyActionType::Press)
		{
			SetSelectedEntity(Entity());
		}

		if (m_sceneCameraWindowFocused && !m_lockEntitySelection && mouseSelectEntity
			&& Input::GetKey(GLFW_MOUSE_BUTTON_LEFT) == KeyActionType::Press &&
			!(m_mouseSceneWindowPosition.x < 0 || m_mouseSceneWindowPosition.y < 0 ||
				m_mouseSceneWindowPosition.x > viewPortSize.x || m_mouseSceneWindowPosition.y > viewPortSize.y)) {
			if (const auto focusedEntity = MouseEntitySelection(sceneCamera, m_mouseSceneWindowPosition); focusedEntity == Entity()) {
				SetSelectedEntity(Entity());
			}
			else {
				Entity walker = focusedEntity;
				bool found = false;
				while (walker.GetIndex() != 0) {
					if (walker == m_selectedEntity) {
						found = true;
						break;
					}
					walker = scene->GetParent(walker);
				}
				if (found) {
					walker = scene->GetParent(walker);
					if (walker.GetIndex() == 0) {
						SetSelectedEntity(focusedEntity);
					}
					else {
						SetSelectedEntity(walker);
					}
				}
				else {
					SetSelectedEntity(focusedEntity);
				}
			}
		}
#pragma endregion
		if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows)) {
			m_sceneCameraWindowFocused = true;
		}
		else {
			m_sceneCameraWindowFocused = false;
		}
		ImGui::EndChild();
	}
	else {
		m_sceneCameraWindowFocused = false;
	}
	sceneCamera->SetRequireRendering(
		!(ImGui::GetCurrentWindowRead()->Hidden && !ImGui::GetCurrentWindowRead()->Collapsed));

	ImGui::End();

	ImGui::PopStyleVar();

#pragma endregion
}

void EditorLayer::MainCameraWindow()
{
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	if (!renderLayer)
		return;
	const auto& graphics = Graphics::GetInstance();
	const auto scene = GetScene();
#pragma region Window
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
	if (ImGui::Begin("Camera")) {
		static int corner = 1;
		// Using a Child allow to fill all the space of the window.
		// It also allows customization
		if (ImGui::BeginChild("MainCameraRenderer", ImVec2(0, 0), false)) {
			ImVec2 viewPortSize = ImGui::GetWindowSize();
			m_mainCameraResolutionX = viewPortSize.x * m_mainCameraResolutionMultiplier;
			m_mainCameraResolutionY = viewPortSize.y * m_mainCameraResolutionMultiplier;
			//  Get the size of the child (i.e. the whole draw size of the windows).
			ImVec2 overlayPos = ImGui::GetWindowPos();
			// Because I use the texture from OpenGL, I need to invert the V from the UV.
			const auto mainCamera = scene->m_mainCamera.Get<Camera>();
			if (mainCamera && mainCamera->m_rendered) {
				ImGui::Image(mainCamera->GetRenderTexture()->GetColorImTextureId(), ImVec2(viewPortSize.x, viewPortSize.y), ImVec2(0, 1),
					ImVec2(1, 0));
				CameraWindowDragAndDrop();
			}
			else {
				ImGui::Text("No active main camera!");
			}

			const ImVec2 window_pos = ImVec2(
				(corner & 1) ? (overlayPos.x + viewPortSize.x) : (overlayPos.x),
				(corner & 2) ? (overlayPos.y + viewPortSize.y) : (overlayPos.y));
			if (m_showCameraInfo) {
				ImVec2 window_pos_pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
				ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
				ImGui::SetNextWindowBgAlpha(0.35f);
				ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDocking |
					ImGuiWindowFlags_AlwaysAutoResize |
					ImGuiWindowFlags_NoSavedSettings |
					ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;

				if (ImGui::BeginChild("Render Info", ImVec2(300, 150), true, window_flags)) {
					ImGui::Text("Info & Settings");
					ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
					ImGui::PushItemWidth(100);
					ImGui::Checkbox("Auto resize", &m_mainCameraAllowAutoResize);
					if (m_mainCameraAllowAutoResize)
					{
						ImGui::DragFloat("Resolution multiplier", &m_mainCameraResolutionMultiplier, 0.1f, 0.1f, 4.0f);
					}
					ImGui::PopItemWidth();
					std::string drawCallInfo = {};
					const auto currentFrameIndex = Graphics::GetCurrentFrameIndex();
					if (graphics.m_triangles[currentFrameIndex] < 999)
						drawCallInfo += std::to_string(graphics.m_triangles[currentFrameIndex]);
					else if (graphics.m_triangles[currentFrameIndex] < 999999)
						drawCallInfo += std::to_string((int)(graphics.m_triangles[currentFrameIndex] / 1000)) + "K";
					else
						drawCallInfo += std::to_string((int)(graphics.m_triangles[currentFrameIndex] / 1000000)) + "M";
					drawCallInfo += " tris";
					ImGui::Text(drawCallInfo.c_str());
					ImGui::Text("%d drawcall", graphics.m_drawCall[currentFrameIndex]);
					ImGui::Separator();
					if (ImGui::IsMousePosValid()) {
						const auto pos = Input::GetMousePosition();
						ImGui::Text("Mouse Pos: (%.1f,%.1f)", pos.x, pos.y);
					}
					else {
						ImGui::Text("Mouse Pos: <invalid>");
					}
				}
				ImGui::EndChild();
			}

			if (m_mainCameraWindowFocused && !m_lockEntitySelection && Input::GetKey(GLFW_KEY_ESCAPE) == KeyActionType::Press)
			{
				SetSelectedEntity(Entity());
			}
			if (!Application::IsPlaying() && m_mainCameraWindowFocused
				&& !m_lockEntitySelection && Input::GetKey(GLFW_MOUSE_BUTTON_LEFT) == KeyActionType::Press
				&& !(m_mouseCameraWindowPosition.x < 0 || m_mouseCameraWindowPosition.y < 0 ||
					m_mouseCameraWindowPosition.x > viewPortSize.x || m_mouseCameraWindowPosition.y > viewPortSize.y)) {
				if (const auto focusedEntity = MouseEntitySelection(mainCamera, m_mouseCameraWindowPosition); focusedEntity == Entity()) {
					SetSelectedEntity(Entity());
				}
				else {
					Entity walker = focusedEntity;
					bool found = false;
					while (walker.GetIndex() != 0) {
						if (walker == m_selectedEntity) {
							found = true;
							break;
						}
						walker = scene->GetParent(walker);
					}
					if (found) {
						walker = scene->GetParent(walker);
						if (walker.GetIndex() == 0) {
							SetSelectedEntity(focusedEntity);
						}
						else {
							SetSelectedEntity(walker);
						}
					}
					else {
						SetSelectedEntity(focusedEntity);
					}
				}
			}
		}


		if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows)) {
			m_mainCameraWindowFocused = true;
		}
		else {
			m_mainCameraWindowFocused = false;
		}

		ImGui::EndChild();
	}
	else {
		m_mainCameraWindowFocused = false;
	}
	if (const auto mainCamera = scene->m_mainCamera.Get<Camera>()) {
		mainCamera->SetRequireRendering(
			!ImGui::GetCurrentWindowRead()->Hidden && !ImGui::GetCurrentWindowRead()->Collapsed);
	}

	ImGui::End();
	ImGui::PopStyleVar();
#pragma endregion
}

void EditorLayer::OnInputEvent(const InputEvent& inputEvent)
{
	//If main camera is focused, we pass the event to the scene.
	if (m_mainCameraWindowFocused && Application::IsPlaying())
	{
		const auto activeScene = Application::GetActiveScene();
		auto& pressedKeys = activeScene->m_pressedKeys;
		if (inputEvent.m_keyAction == KeyActionType::Press)
		{
			if (const auto search = pressedKeys.find(inputEvent.m_key); search != activeScene->m_pressedKeys.end())
			{
				//Dispatch hold if the key is already pressed.
				search->second = KeyActionType::Hold;
			}
			else
			{
				//Dispatch press if the key is previously released.
				pressedKeys.insert({ inputEvent.m_key, KeyActionType::Press });
			}
		}
		else if (inputEvent.m_keyAction == KeyActionType::Release)
		{
			if (pressedKeys.find(inputEvent.m_key) != pressedKeys.end())
			{
				//Dispatch hold if the key is already pressed.
				pressedKeys.erase(inputEvent.m_key);
			}
		}
	}
}

void EditorLayer::ResizeCameras()
{
	const auto renderLayer = Application::GetLayer<RenderLayer>();
	if (!renderLayer)
		return;
	const auto& sceneCamera = GetSceneCamera();
	const auto resolution = sceneCamera->GetSize();
	if (m_sceneCameraResolutionX != 0 && m_sceneCameraResolutionY != 0 &&
		(resolution.x != m_sceneCameraResolutionX || resolution.y != m_sceneCameraResolutionY)) {
		sceneCamera->Resize({ m_sceneCameraResolutionX, m_sceneCameraResolutionY });
	}
	const auto scene = Application::GetActiveScene();
	if (const std::shared_ptr<Camera> mainCamera = scene->m_mainCamera.Get<Camera>())
	{
		if (m_mainCameraAllowAutoResize) mainCamera->Resize({ m_mainCameraResolutionX, m_mainCameraResolutionY });
	}
}

bool EditorLayer::SceneCameraWindowFocused() const
{
	return m_sceneCameraWindowFocused;
}

bool EditorLayer::MainCameraWindowFocused() const
{
	return m_mainCameraWindowFocused;
}

void EditorLayer::RegisterEditorCamera(const std::shared_ptr<Camera>& camera)
{
	if (m_editorCameras.find(camera->GetHandle()) == m_editorCameras.end()) {
		m_editorCameras[camera->GetHandle()] = {};
		m_editorCameras.at(camera->GetHandle()).m_camera = camera;
	}
}

glm::vec2 EditorLayer::GetMouseSceneCameraPosition() const
{
	return m_mouseSceneWindowPosition;
}

KeyActionType EditorLayer::GetKey(const int key) const
{
	return Input::GetKey(key);
}

std::shared_ptr<Camera> EditorLayer::GetSceneCamera()
{
	return m_editorCameras.at(m_sceneCameraHandle).m_camera;
}

glm::vec3 EditorLayer::GetSceneCameraPosition() const
{
	return m_editorCameras.at(m_sceneCameraHandle).m_position;
}

glm::quat EditorLayer::GetSceneCameraRotation() const
{
	return m_editorCameras.at(m_sceneCameraHandle).m_rotation;
}

void EditorLayer::SetCameraPosition(const std::shared_ptr<Camera>& camera, const glm::vec3& targetPosition)
{
	m_editorCameras.at(camera->GetHandle()).m_position = targetPosition;
}

void EditorLayer::SetCameraRotation(const std::shared_ptr<Camera>& camera, const glm::quat& targetRotation)
{
	m_editorCameras.at(camera->GetHandle()).m_rotation = targetRotation;
}

void EditorLayer::UpdateTextureId(ImTextureID& target, VkSampler imageSampler, const VkImageView imageView, VkImageLayout imageLayout)
{
	if (!Application::GetLayer<EditorLayer>()) return;
	if (target != VK_NULL_HANDLE) ImGui_ImplVulkan_RemoveTexture(static_cast<VkDescriptorSet>(target));
	target = ImGui_ImplVulkan_AddTexture(imageSampler, imageView, imageLayout);
}

Entity EditorLayer::GetSelectedEntity() const
{
	return m_selectedEntity;
}

void EditorLayer::SetSelectedEntity(const Entity& entity, bool openMenu)
{
	if (entity == m_selectedEntity)
		return;
	m_selectedEntityHierarchyList.clear();
	const auto scene = GetScene();
	const auto previousDescendents = scene->GetDescendants(m_selectedEntity);
	for (const auto& i : previousDescendents)
	{
		scene->GetEntityMetadata(i).m_ancestorSelected = false;
	}
	if (scene->IsEntityValid(m_selectedEntity)) scene->GetEntityMetadata(m_selectedEntity).m_ancestorSelected = false;
	if (entity.GetIndex() == 0) {
		m_selectedEntity = Entity();
		m_lockEntitySelection = false;
		m_selectionAlpha = 0;
		return;
	}

	if (!scene->IsEntityValid(entity))
		return;
	m_selectedEntity = entity;
	const auto descendents = scene->GetDescendants(m_selectedEntity);

	for (const auto& i : descendents)
	{
		scene->GetEntityMetadata(i).m_ancestorSelected = true;
	}
	scene->GetEntityMetadata(m_selectedEntity).m_ancestorSelected = true;
	if (!openMenu)
		return;
	auto walker = entity;
	while (walker.GetIndex() != 0) {
		m_selectedEntityHierarchyList.push_back(walker);
		walker = scene->GetParent(walker);
	}
}

bool EditorLayer::GetLockEntitySelection() const
{
	return m_lockEntitySelection;
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

Entity EditorLayer::MouseEntitySelection(const std::shared_ptr<Camera>& targetCamera, const glm::vec2& mousePosition) const
{
	Entity retVal;
	const auto& gBufferNormal = targetCamera->m_gBufferNormal;
	const glm::vec2 resolution = targetCamera->GetSize();
	glm::vec2 point = resolution;
	point.x = mousePosition.x;
	point.y -= mousePosition.y;
	if (point.x >= 0 && point.x < resolution.x && point.y >= 0 && point.y < resolution.y) {
		VkBufferImageCopy imageCopy{};
		imageCopy.bufferOffset = 0;
		imageCopy.bufferRowLength = 0;
		imageCopy.bufferImageHeight = 0;
		imageCopy.imageSubresource.layerCount = 1;
		imageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageCopy.imageSubresource.baseArrayLayer = 0;
		imageCopy.imageSubresource.mipLevel = 0;
		imageCopy.imageExtent.width = 1;
		imageCopy.imageExtent.height = 1;
		imageCopy.imageExtent.depth = 1;
		imageCopy.imageOffset.x = point.x;
		imageCopy.imageOffset.y = point.y;
		imageCopy.imageOffset.z = 0;
		m_entityIndexReadBuffer->CopyFromImage(*gBufferNormal, imageCopy);
		if (const float instanceIndexWithOneAdded = glm::roundEven(glm::detail::toFloat32(m_mappedEntityIndexData[3])); instanceIndexWithOneAdded > 0) {
			const auto renderLayer = Application::GetLayer<RenderLayer>();
			const auto scene = GetScene();
			const auto handle = renderLayer->GetInstanceHandle(static_cast<uint32_t>(instanceIndexWithOneAdded - 1));
			if (handle != 0) retVal = scene->GetEntity(handle);
		}
	}
	return retVal;
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
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0.5f, 0, 1));
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
	ImGui::PopStyleColor(1);
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
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.3f, 0, 1));
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
	ImGui::PopStyleColor(1);
	statusChanged = UnsafeDroppablePrivateComponent(target, acceptableTypeNames) || statusChanged;
	return statusChanged;
}

void EditorLayer::LoadIcons()
{
	m_assetsIcons["Project"] = Resources::CreateResource<Texture2D>("PROJECT_ICON");
	m_assetsIcons["Project"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Assets/project.png");

	m_assetsIcons["Scene"] = Resources::CreateResource<Texture2D>("SCENE_ICON");
	m_assetsIcons["Scene"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Assets/scene.png");

	m_assetsIcons["Binary"] = Resources::CreateResource<Texture2D>("BINARY_ICON");
	m_assetsIcons["Binary"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Assets/binary.png");

	m_assetsIcons["Folder"] = Resources::CreateResource<Texture2D>("FOLDER_ICON");
	m_assetsIcons["Folder"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Assets/folder.png");

	m_assetsIcons["Material"] = Resources::CreateResource<Texture2D>("MATERIAL_ICON");
	m_assetsIcons["Material"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Assets/material.png");


	m_assetsIcons["Mesh"] = Resources::CreateResource<Texture2D>("MESH_ICON");
	m_assetsIcons["Mesh"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Assets/mesh.png");


	m_assetsIcons["Prefab"] = Resources::CreateResource<Texture2D>("PREFAB_ICON");
	m_assetsIcons["Prefab"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Assets/prefab.png");

	m_assetsIcons["Texture2D"] = Resources::CreateResource<Texture2D>("TEXTURE2D_ICON");
	m_assetsIcons["Texture2D"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Assets/texture2d.png");


	m_assetsIcons["PlayButton"] = Resources::CreateResource<Texture2D>("PLAY_BUTTON_ICON");
	m_assetsIcons["PlayButton"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Navigation/PlayButton.png");


	m_assetsIcons["PauseButton"] = Resources::CreateResource<Texture2D>("PAUSE_BUTTON_ICON");
	m_assetsIcons["PauseButton"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Navigation/PauseButton.png");


	m_assetsIcons["StopButton"] = Resources::CreateResource<Texture2D>("STOP_BUTTON_ICON");
	m_assetsIcons["StopButton"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Navigation/StopButton.png");


	m_assetsIcons["StepButton"] = Resources::CreateResource<Texture2D>("STEP_BUTTON_ICON");
	m_assetsIcons["StepButton"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Navigation/StepButton.png");


	m_assetsIcons["BackButton"] = Resources::CreateResource<Texture2D>("BACK_BUTTON_ICON");
	m_assetsIcons["BackButton"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Navigation/back.png");


	m_assetsIcons["LeftButton"] = Resources::CreateResource<Texture2D>("LEFT_BUTTON_ICON");
	m_assetsIcons["LeftButton"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Navigation/left.png");


	m_assetsIcons["RightButton"] = Resources::CreateResource<Texture2D>("RIGHT_BUTTON_ICON");
	m_assetsIcons["RightButton"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Navigation/right.png");


	m_assetsIcons["RefreshButton"] = Resources::CreateResource<Texture2D>("REFRESH_BUTTON_ICON");
	m_assetsIcons["RefreshButton"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Navigation/refresh.png");


	m_assetsIcons["InfoButton"] = Resources::CreateResource<Texture2D>("INFO_BUTTON_ICON");
	m_assetsIcons["InfoButton"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Console/InfoButton.png");


	m_assetsIcons["ErrorButton"] = Resources::CreateResource<Texture2D>("ERROR_BUTTON_ICON");
	m_assetsIcons["ErrorButton"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Console/ErrorButton.png");

	m_assetsIcons["WarningButton"] = Resources::CreateResource<Texture2D>("WARNING_BUTTON_ICON");
	m_assetsIcons["WarningButton"]->LoadInternal(std::filesystem::path("./DefaultResources") / "Editor/Console/WarningButton.png");

}

void EditorLayer::CameraWindowDragAndDrop() {
	AssetRef assetRef;
	if (UnsafeDroppableAsset(assetRef,
		{ "Scene", "Prefab", "Mesh", "Strands", "Cubemap", "EnvironmentalMap" })) {
		auto scene = GetScene();
		auto asset = assetRef.Get<IAsset>();
		if (!Application::IsPlaying() && asset->GetTypeName() == "Scene") {
			auto scene = std::dynamic_pointer_cast<Scene>(asset);
			ProjectManager::SetStartScene(scene);
			Application::Attach(scene);
		}

		else if (asset->GetTypeName() == "Prefab") {
			auto entity = std::dynamic_pointer_cast<Prefab>(asset)->ToEntity(scene);
			scene->SetEntityName(entity, asset->GetTitle());
		}
		else if (asset->GetTypeName() == "Mesh") {
			Entity entity = scene->CreateEntity(asset->GetTitle());
			auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(entity).lock();
			meshRenderer->m_mesh.Set<Mesh>(std::dynamic_pointer_cast<Mesh>(asset));
			auto material = ProjectManager::CreateTemporaryAsset<Material>();
			meshRenderer->m_material.Set<Material>(material);
		}

		else if (asset->GetTypeName() == "Strands") {
			Entity entity = scene->CreateEntity(asset->GetTitle());
			auto strandsRenderer = scene->GetOrSetPrivateComponent<StrandsRenderer>(entity).lock();
			strandsRenderer->m_strands.Set<Strands>(std::dynamic_pointer_cast<Strands>(asset));
			auto material = ProjectManager::CreateTemporaryAsset<Material>();
			strandsRenderer->m_material.Set<Material>(material);
		}
		else if (asset->GetTypeName() == "EnvironmentalMap") {
			scene->m_environment.m_environmentalMap =
				std::dynamic_pointer_cast<EnvironmentalMap>(asset);
		}
		else if (asset->GetTypeName() == "Cubemap") {
			auto mainCamera = scene->m_mainCamera.Get<Camera>();
			mainCamera->m_skybox = std::dynamic_pointer_cast<Cubemap>(asset);
		}

	}
}

void EditorLayer::MoveCamera(
	const glm::quat& targetRotation, const glm::vec3& targetPosition, const float& transitionTime) {
	auto& [sceneCameraRotation, sceneCameraPosition, sceneCamera] = m_editorCameras.at(m_sceneCameraHandle);
	m_previousRotation = sceneCameraRotation;
	m_previousPosition = sceneCameraPosition;
	m_transitionTime = transitionTime;
	m_transitionTimer = Times::Now();
	m_targetRotation = targetRotation;
	m_targetPosition = targetPosition;
	m_lockCamera = true;
}


bool EditorLayer::LocalPositionSelected() const {
	return m_localPositionSelected;
}

bool EditorLayer::LocalRotationSelected() const {
	return m_localRotationSelected;
}

bool EditorLayer::LocalScaleSelected() const {
	return m_localScaleSelected;
}

glm::vec3& EditorLayer::UnsafeGetPreviouslyStoredPosition() {
	return m_previouslyStoredPosition;
}

glm::vec3& EditorLayer::UnsafeGetPreviouslyStoredRotation() {
	return m_previouslyStoredRotation;
}

glm::vec3& EditorLayer::UnsafeGetPreviouslyStoredScale() {
	return m_previouslyStoredScale;
}