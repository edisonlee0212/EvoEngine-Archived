#pragma once
#include "Entity.hpp"
#include "GraphicsResources.hpp"
#include "ILayer.hpp"
#include "ISystem.hpp"
#include "PrivateComponentRef.hpp"
#include "Texture2D.hpp"
#include "Application.hpp"
#include "Camera.hpp"
#include "Material.hpp"
#include "ProjectManager.hpp"
#include "Scene.hpp"
#include "Mesh.hpp"
#include "Strands.hpp"
namespace EvoEngine
{
	struct GizmoSettings {
		DrawSettings m_drawSettings;
		enum class ColorMode {
			Default,
			VertexColor,
			NormalColor
		} m_colorMode = ColorMode::Default;
		bool m_depthTest = false;
		bool m_depthWrite = false;
		void ApplySettings(GraphicsPipelineStates& globalPipelineState) const;
	};
	struct GizmosPushConstant
	{
		glm::mat4 m_model;
		glm::vec4 m_color;
		float m_size;
		uint32_t m_cameraIndex;
	};

	struct EditorCamera
	{
		glm::quat m_rotation = glm::quat(glm::radians(glm::vec3(0.0f, 0.0f, 0.0f)));
		glm::vec3 m_position = glm::vec3(0, 2, 5);
		std::shared_ptr<Camera> m_camera;
	};

	struct GizmoMeshTask
	{
		std::shared_ptr<Mesh> m_mesh;
		std::shared_ptr<Camera> m_editorCameraComponent;
		glm::vec4 m_color;
		glm::mat4 m_model;
		float m_size;
		GizmoSettings m_gizmoSettings;
	};

	struct GizmoInstancedMeshTask
	{
		std::shared_ptr<Mesh> m_mesh;
		std::shared_ptr<Camera> m_editorCameraComponent;
		std::shared_ptr<ParticleInfoList> m_instancedData;
		glm::mat4 m_model;
		float m_size;
		GizmoSettings m_gizmoSettings;
	};

	struct GizmoStrandsTask
	{
		std::shared_ptr<Strands> m_strands;
		std::shared_ptr<Camera> m_editorCameraComponent;
		glm::vec4 m_color;
		glm::mat4 m_model;
		float m_size;
		GizmoSettings m_gizmoSettings;
	};

	class EditorLayer : public ILayer
	{
		void LoadIcons();
		void OnCreate() override;
		void OnDestroy() override;
		void PreUpdate() override;
		void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		void LateUpdate() override;
		
		void SceneCameraWindow();
		void MainCameraWindow();
		void OnInputEvent(const InputEvent& inputEvent) override;
		void ResizeCameras();
		Handle m_sceneCameraHandle = 0;
		std::unordered_map<Handle, EditorCamera> m_editorCameras;

		std::vector<GizmoMeshTask> m_gizmoMeshTasks;
		std::vector<GizmoInstancedMeshTask> m_gizmoInstancedMeshTasks;
		std::vector<GizmoStrandsTask> m_gizmoStrandsTasks;
	public:
		void RegisterEditorCamera(const std::shared_ptr<Camera>& camera);

		[[nodiscard]] glm::vec2 GetMouseSceneCameraPosition() const;

		[[nodiscard]] KeyActionType GetKey(int key) const;
		[[nodiscard]] std::shared_ptr<Camera> GetSceneCamera();
		[[nodiscard]] glm::vec3 GetSceneCameraPosition() const;
		[[nodiscard]] glm::quat GetSceneCameraRotation() const;

		void SetCameraPosition(const std::shared_ptr<Camera>& camera, const glm::vec3& targetPosition);
		void SetCameraRotation(const std::shared_ptr<Camera>& camera, const glm::quat& targetRotation);
		void MoveCamera(
			const glm::quat& targetRotation, const glm::vec3& targetPosition, const float& transitionTime = 1.0f);

		static void UpdateTextureId(ImTextureID& target, VkSampler imageSampler, VkImageView imageView, VkImageLayout imageLayout);
		[[nodiscard]] Entity GetSelectedEntity() const;
		void SetSelectedEntity(const Entity& entity, bool openMenu = true);
		float m_sceneCameraResolutionMultiplier = 1.0f;
		[[nodiscard]] bool GetLockEntitySelection() const;
		bool m_showSceneWindow = true;
		bool m_showCameraWindow = true;
		bool m_showCameraInfo = true;
		bool m_showSceneInfo = true;

		bool m_showEntityExplorerWindow = true;
		bool m_showEntityInspectorWindow = true;

		bool m_mainCameraFocusOverride = false;
		bool m_sceneCameraFocusOverride = false;

		int m_selectedHierarchyDisplayMode = 1;
		float m_sceneCameraYawAngle = -90;
		float m_sceneCameraPitchAngle = 0;
		float m_velocity = 10.0f;
		float m_sensitivity = 0.1f;
		bool m_applyTransformToMainCamera = false;
		bool m_lockCamera;

		glm::quat m_defaultSceneCameraRotation = glm::quat(glm::radians(glm::vec3(0.0f, 0.0f, 0.0f)));
		glm::vec3 m_defaultSceneCameraPosition = glm::vec3(0, 2, 5);

		int m_mainCameraResolutionX = 1;
		int m_mainCameraResolutionY = 1;
		bool m_mainCameraAllowAutoResize = true;
#pragma region ImGui Helpers
		void CameraWindowDragAndDrop();
		[[nodiscard]] bool DrawEntityMenu(const bool& enabled, const Entity& entity) const;
		void DrawEntityNode(const Entity& entity, const unsigned& hierarchyLevel);
		void InspectComponentData(Entity entity, IDataComponent* data, DataComponentType type, bool isRoot);

		std::map<std::string, std::shared_ptr<Texture2D>>& AssetIcons();

		template <typename T1 = IDataComponent>
		void RegisterComponentDataInspector(
			const std::function<bool(Entity entity, IDataComponent* data, bool isRoot)>& func);

		bool DragAndDropButton(
			AssetRef& target,
			const std::string& name,
			const std::vector<std::string>& acceptableTypeNames,
			bool modifiable = true);
		bool DragAndDropButton(
			PrivateComponentRef& target,
			const std::string& name,
			const std::vector<std::string>& acceptableTypeNames,
			bool modifiable = true);

		template <typename T = IAsset>
		bool DragAndDropButton(AssetRef& target, const std::string& name, bool modifiable = true);
		template <typename T = IPrivateComponent>
		bool DragAndDropButton(PrivateComponentRef& target, const std::string& name, bool modifiable = true);
		bool DragAndDropButton(EntityRef& entityRef, const std::string& name, bool modifiable = true);

		template <typename T = IAsset>  void Draggable(AssetRef& target);
		template <typename T = IPrivateComponent>  void Draggable(PrivateComponentRef& target);
		void Draggable(EntityRef& entityRef);

		template <typename T = IAsset>  void DraggableAsset(const std::shared_ptr<T>& target);
		template <typename T = IPrivateComponent>  void DraggablePrivateComponent(const std::shared_ptr<T>& target);
		void DraggableEntity(const Entity& entity);

		bool UnsafeDroppableAsset(AssetRef& target, const std::vector<std::string>& typeNames);
		bool UnsafeDroppablePrivateComponent(PrivateComponentRef& target, const std::vector<std::string>& typeNames);

		template <typename T = IAsset>  bool Droppable(AssetRef& target);
		template <typename T = IPrivateComponent>  bool Droppable(PrivateComponentRef& target);
		bool Droppable(EntityRef& entityRef);

		template <typename T = IAsset>  bool Rename(AssetRef& target);
		bool Rename(EntityRef& entityRef);

		template <typename T = IAsset>  bool RenameAsset(const std::shared_ptr<T>& target);
		[[nodiscard]] bool RenameEntity(const Entity& entity) const;

		template <typename T = IAsset>  bool Remove(AssetRef& target) const;
		template <typename T = IPrivateComponent>  bool Remove(PrivateComponentRef& target) const;
		[[nodiscard]] bool Remove(EntityRef& entityRef);

#pragma endregion

#pragma region Gizmos
		void DrawGizmoMesh(
			const std::shared_ptr<Mesh>& mesh,
			const std::shared_ptr<Camera>& editorCameraComponent,
			const glm::vec4& color = glm::vec4(1.0f),
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		void DrawGizmoStrands(
			const std::shared_ptr<Strands>& strands,
			const std::shared_ptr<Camera>& editorCameraComponent,
			const glm::vec4& color = glm::vec4(1.0f),
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		void DrawGizmoMeshInstancedColored(
			const std::shared_ptr<Mesh>& mesh,
			const std::shared_ptr<Camera>& editorCameraComponent,
			const std::shared_ptr<ParticleInfoList>& instancedData,
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		void DrawGizmoMeshInstancedColored(
			const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<ParticleInfoList>& instancedData,
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		void DrawGizmoMesh(
			const std::shared_ptr<Mesh>& mesh,
			const glm::vec4& color = glm::vec4(1.0f),
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		void DrawGizmoStrands(
			const std::shared_ptr<Strands>& strands,
			const glm::vec4& color = glm::vec4(1.0f),
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		
		void DrawGizmoCubes(const std::shared_ptr<ParticleInfoList>& instancedData,
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		void DrawGizmoCube(
			const glm::vec4& color = glm::vec4(1.0f),
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		void DrawGizmoSpheres(const std::shared_ptr<ParticleInfoList>& instancedData,
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		void DrawGizmoSphere(
			const glm::vec4& color = glm::vec4(1.0f),
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		void DrawGizmoCylinders(const std::shared_ptr<ParticleInfoList>& instancedData,
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		void DrawGizmoCylinder(
			const glm::vec4& color = glm::vec4(1.0f),
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});
#pragma endregion
	private:
		int m_selectionAlpha = 0;

		glm::detail::hdata* m_mappedEntityIndexData;
		std::unique_ptr<Buffer> m_entityIndexReadBuffer;
		[[nodiscard]] Entity MouseEntitySelection(const std::shared_ptr<Camera>& targetCamera, const glm::vec2& mousePosition) const;

		EntityArchetype m_basicEntityArchetype;
		glm::vec3 m_previouslyStoredPosition;
		glm::vec3 m_previouslyStoredRotation;
		glm::vec3 m_previouslyStoredScale;
		bool m_localPositionSelected = true;
		bool m_localRotationSelected = false;
		bool m_localScaleSelected = false;

		bool m_sceneCameraWindowFocused = false;
		bool m_mainCameraWindowFocused = false;
#pragma region Registrations
		friend class ClassRegistry;
		friend class RenderLayer;
		friend class Application;
		friend class ProjectManager;
		friend class Scene;
		std::map<std::string, std::shared_ptr<Texture2D>> m_assetsIcons;
		bool m_enabled = false;
		std::map<size_t, std::function<bool(Entity entity, IDataComponent* data, bool isRoot)>> m_componentDataInspectorMap;
		std::vector<std::pair<size_t, std::function<void(Entity owner)>>> m_privateComponentMenuList;
		std::vector<std::pair<size_t, std::function<void(float rank)>>> m_systemMenuList;
		std::vector<std::pair<size_t, std::function<void(Entity owner)>>> m_componentDataMenuList;
		template <typename T1 = IPrivateComponent>  void RegisterPrivateComponent();
		template <typename T1 = ISystem>  void RegisterSystem();
		template <typename T1 = IDataComponent>  void RegisterDataComponent();

		std::vector<std::weak_ptr<AssetRecord>> m_assetRecordBus;
		std::map<std::string, std::vector<AssetRef>> m_assetRefBus;
		std::map<std::string, std::vector<PrivateComponentRef>> m_privateComponentRefBus;
		std::map<std::string, std::vector<EntityRef>> m_entityRefBus;
#pragma endregion
#pragma region Transfer

		glm::quat m_previousRotation;
		glm::vec3 m_previousPosition;
		glm::quat m_targetRotation;
		glm::vec3 m_targetPosition;
		float m_transitionTime;
		float m_transitionTimer;
#pragma endregion
		std::vector<Entity> m_selectedEntityHierarchyList;

		int m_sceneCameraResolutionX = 1;
		int m_sceneCameraResolutionY = 1;

		bool m_lockEntitySelection = false;

		bool m_highlightSelection = true;
		Entity m_selectedEntity;

		glm::vec2 m_mouseSceneWindowPosition;
		glm::vec2 m_mouseCameraWindowPosition;

		
		float m_mainCameraResolutionMultiplier = 1.0f;
	};

#pragma region ImGui Helpers

	template <typename T1>
	void EditorLayer::RegisterComponentDataInspector(
		const std::function<bool(Entity entity, IDataComponent* data, bool isRoot)>& func)
	{
		m_componentDataInspectorMap.insert_or_assign(typeid(T1).hash_code(), func);
	}

	template <typename T> void EditorLayer::RegisterSystem()
	{
		const auto scene = Application::GetActiveScene();
		auto func = [&](float rank) {
			if (scene->GetSystem<T>())
				return;
			auto systemName = Serialization::GetSerializableTypeName<T>();
			if (ImGui::Button(systemName.c_str()))
			{
				scene->GetOrCreateSystem(systemName, rank);
			}
		};
		for (int i = 0; i < m_systemMenuList.size(); i++)
		{
			if (m_systemMenuList[i].first == typeid(T).hash_code())
			{
				m_systemMenuList[i].second = func;
				return;
			}
		}
		m_systemMenuList.emplace_back(typeid(T).hash_code(), func);
	}

	template <typename T> void EditorLayer::RegisterPrivateComponent()
	{
		
		auto func = [&](const Entity owner) {
			const auto scene = Application::GetActiveScene();
			if (scene->HasPrivateComponent<T>(owner))
				return;
			if (ImGui::Button(Serialization::GetSerializableTypeName<T>().c_str()))
			{
				scene->GetOrSetPrivateComponent<T>(owner);
			}
		};
		for (int i = 0; i < m_privateComponentMenuList.size(); i++)
		{
			if (m_privateComponentMenuList[i].first == typeid(T).hash_code())
			{
				m_privateComponentMenuList[i].second = func;
				return;
			}
		}
		m_privateComponentMenuList.emplace_back(typeid(T).hash_code(), func);
	}

	template <typename T> void EditorLayer::RegisterDataComponent()
	{
		if (const auto id = typeid(T).hash_code(); id == typeid(Transform).hash_code() || id == typeid(GlobalTransform).hash_code() ||
			id == typeid(GlobalTransformUpdateFlag).hash_code())
			return;
		auto func = [](const Entity owner) {
			const auto scene = Application::GetActiveScene();
			if (scene->HasPrivateComponent<T>(owner))
				return;
			if (ImGui::Button(Serialization::GetDataComponentTypeName<T>().c_str()))
			{
				scene->AddDataComponent<T>(owner, T());
			}
		};
		for (int i = 0; i < m_componentDataMenuList.size(); i++)
		{
			if (m_componentDataMenuList[i].first == typeid(T).hash_code())
			{
				m_componentDataMenuList[i].second = func;
				return;
			}
		}
		m_componentDataMenuList.emplace_back(typeid(T).hash_code(), func);
	}

	template <typename T> bool EditorLayer::DragAndDropButton(AssetRef& target, const std::string& name, bool modifiable)
	{
		ImGui::Text(name.c_str());
		ImGui::SameLine();
		const auto ptr = target.Get<IAsset>();
		bool statusChanged = false;
		if (ptr)
		{
			ImGui::Button(ptr->GetTitle().c_str());
			Draggable(target);
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
		statusChanged = Droppable<T>(target) || statusChanged;
		return statusChanged;
	}

	template <typename T>
	bool EditorLayer::DragAndDropButton(PrivateComponentRef& target, const std::string& name, bool modifiable)
	{
		ImGui::Text(name.c_str());
		ImGui::SameLine();
		bool statusChanged = false;
		if (const auto ptr = target.Get<IPrivateComponent>())
		{
			const auto scene = Application::GetActiveScene();
			if (!scene->IsEntityValid(ptr->GetOwner()))
			{
				target.Clear();
				ImGui::Button("none");
				return true;
			}
			ImGui::Button(scene->GetEntityName(ptr->GetOwner()).c_str());
			Draggable(target);
			if (modifiable)
			{
				statusChanged = Remove(target) || statusChanged;
			}
		}
		else
		{
			ImGui::Button("none");
		}
		statusChanged = Droppable<T>(target) || statusChanged;
		return statusChanged;
	}
	template <typename T> void EditorLayer::DraggablePrivateComponent(const std::shared_ptr<T>& target)
	{
		if (const auto ptr = std::dynamic_pointer_cast<IPrivateComponent>(target))
		{
			const auto type = ptr->GetTypeName();
			auto entity = ptr->GetOwner();
			if (const auto scene = Application::GetActiveScene(); scene->IsEntityValid(entity))
			{
				if (ImGui::BeginDragDropSource())
				{
					auto handle = scene->GetEntityHandle(entity);
					ImGui::SetDragDropPayload("PrivateComponent", &handle, sizeof(Handle));
					ImGui::TextColored(ImVec4(0, 0, 1, 1), type.c_str());
					ImGui::EndDragDropSource();
				}
			}
		}
	}
	template <typename T> void EditorLayer::DraggableAsset(const std::shared_ptr<T>& target)
	{
		if (ImGui::BeginDragDropSource())
		{
			const auto ptr = std::dynamic_pointer_cast<IAsset>(target);
			if (ptr)
			{
				const auto title = ptr->GetTitle();
				ImGui::SetDragDropPayload("Asset", &ptr->m_handle, sizeof(Handle));
				ImGui::TextColored(ImVec4(0, 0, 1, 1), title.c_str());
			}
			ImGui::EndDragDropSource();
		}
	}
	template <typename T> void EditorLayer::Draggable(AssetRef& target)
	{
		DraggableAsset(target.Get<IAsset>());
	}
	template <typename T> bool EditorLayer::Droppable(AssetRef& target)
	{
		return UnsafeDroppableAsset(target, { Serialization::GetSerializableTypeName<T>() });
	}

	template <typename T> bool EditorLayer::Rename(AssetRef& target)
	{
		return RenameAsset(target.Get<IAsset>());
	}
	template <typename T> bool EditorLayer::Remove(AssetRef& target) const
	{
		bool statusChanged = false;
		if (const auto ptr = target.Get<IAsset>())
		{
			const std::string type = ptr->GetTypeName();
			const std::string tag = "##" + type + std::to_string(ptr->GetHandle());
			if (ImGui::BeginPopupContextItem(tag.c_str()))
			{
				if (ImGui::Button(("Remove" + tag).c_str()))
				{
					target.Clear();
					statusChanged = true;
				}
				ImGui::EndPopup();
			}
		}
		return statusChanged;
	}
	template <typename T> bool EditorLayer::Remove(PrivateComponentRef& target) const
	{
		bool statusChanged = false;
		if (const auto ptr = target.Get<IPrivateComponent>())
		{
			const std::string type = ptr->GetTypeName();
			const std::string tag = "##" + type + std::to_string(ptr->GetHandle());
			if (ImGui::BeginPopupContextItem(tag.c_str()))
			{
				if (ImGui::Button(("Remove" + tag).c_str()))
				{
					target.Clear();
					statusChanged = true;
				}
				ImGui::EndPopup();
			}
		}
		return statusChanged;
	}

	template <typename T> bool EditorLayer::RenameAsset(const std::shared_ptr<T>& target)
	{
		const bool statusChanged = false;
		auto ptr = std::dynamic_pointer_cast<IAsset>(target);
		const std::string type = ptr->GetTypeName();
		const std::string tag = "##" + type + std::to_string(ptr->GetHandle());
		if (ImGui::BeginPopupContextItem(tag.c_str()))
		{
			if (!ptr->IsTemporary())
			{
				if (ImGui::BeginMenu(("Rename" + tag).c_str()))
				{
					static char newName[256];
					ImGui::InputText(("New name" + tag).c_str(), newName, 256);
					if (ImGui::Button(("Confirm" + tag).c_str()))
					{
						if (bool succeed = ptr->SetPathAndSave(ptr->GetProjectRelativePath().replace_filename(
							std::string(newName) + ptr->GetAssetRecord().lock()->GetAssetExtension())))
							memset(newName, 0, 256);
					}
					ImGui::EndMenu();
				}
			}
			ImGui::EndPopup();
		}
		return statusChanged;
	}
	template <typename T> void EditorLayer::Draggable(PrivateComponentRef& target)
	{
		DraggablePrivateComponent(target.Get<IPrivateComponent>());
	}

	template <typename T> bool EditorLayer::Droppable(PrivateComponentRef& target)
	{
		return UnsafeDroppablePrivateComponent(target, { Serialization::GetSerializableTypeName<T>() });
	}

#pragma endregion
}
