#pragma once
#include "ISingleton.hpp"
#include "Console.hpp"
#include "ILayer.hpp"

namespace EvoEngine {

	struct ApplicationInfo {
		std::filesystem::path m_projectPath;
		std::string m_applicationName = "Application";
		glm::ivec2 m_defaultWindowSize = { 1280, 720 };
		bool m_enableDocking = true;
		bool m_enableViewport = false;
		bool m_fullScreen = false;
	};

	enum class ApplicationStatus {
		Uninitialized,
		NoProject,

		Stop,
		Pause,
		Step,
		Playing,

		OnDestroy
	};

	class Application final : public ISingleton<Application>
	{
		friend class ProjectManager;
		ApplicationInfo m_applicationInfo;
		ApplicationStatus m_applicationStatus = ApplicationStatus::Uninitialized;

		static void PreUpdateInternal();
		static void UpdateInternal();
		static void LateUpdateInternal();

		std::vector<std::shared_ptr<ILayer>> m_layers;
		std::shared_ptr<Scene> m_activeScene;

		std::vector<std::function<void()>> m_externalPreUpdateFunctions;
		std::vector<std::function<void()>> m_externalUpdateFunctions;
		std::vector<std::function<void()>> m_externalFixedUpdateFunctions;
		std::vector<std::function<void()>> m_externalLateUpdateFunctions;

		std::vector<std::function<void(const std::shared_ptr<Scene>& newScene)>> m_postAttachSceneFunctions;

		static void InitializeRegistry();
	public:
		static void RegisterPreUpdateFunction(const std::function<void()>& func);
		static void RegisterUpdateFunction(const std::function<void()>& func);
		static void RegisterLateUpdateFunction(const std::function<void()>& func);
		static void RegisterFixedUpdateFunction(const std::function<void()>& func);
		static void RegisterPostAttachSceneFunction(const std::function<void(const std::shared_ptr<Scene>& newScene)>& func);
		static bool IsPlaying();
		static const ApplicationInfo& GetApplicationInfo();
		static const ApplicationStatus& GetApplicationStatus();
		template <typename T>
		static std::shared_ptr<T> PushLayer();
		template <typename T>
		static std::shared_ptr<T> GetLayer();
		template <typename T>
		static void PopLayer();
		static void Reset();
		static void Initialize(const ApplicationInfo& applicationCreateInfo);
		static void Start();
		static void End();
		static void Terminate();
		static const std::vector<std::shared_ptr<ILayer>>& GetLayers();
		static void Attach(const std::shared_ptr<Scene>& scene);
		static std::shared_ptr<Scene> GetActiveScene();
		static void Play();
		static void Pause();
		static void Step();

		static void Stop();
	};

	template <typename T>
	std::shared_ptr<T> Application::PushLayer()
	{
		auto& application = GetInstance();
		if (application.m_applicationStatus != ApplicationStatus::Uninitialized) {
			EVOENGINE_ERROR("Unable to push layer! Application already started!");
			return nullptr;
		}
		auto test = GetLayer<T>();
		if (!test) {
			test = std::make_shared<T>();
			if (!std::dynamic_pointer_cast<ILayer>(test)) {
				EVOENGINE_ERROR("Not a layer!");
				return nullptr;
			}
			if (!application.m_layers.empty()) application.m_layers.back()->m_subsequentLayer = test;
			application.m_layers.push_back(std::dynamic_pointer_cast<ILayer>(test));
		}
		return test;
	}

	template <typename T> std::shared_ptr<T> Application::GetLayer()
	{
		auto& application = GetInstance();
		for (auto& i : application.m_layers) {
			if (auto test = std::dynamic_pointer_cast<T>(i)) return test;
		}
		return nullptr;
	}
	template <typename T> void Application::PopLayer()
	{
		auto& application = GetInstance();
		int index = 0;
		for (auto& i : application.m_layers) {
			if (auto test = std::dynamic_pointer_cast<T>(i)) {
				std::dynamic_pointer_cast<ILayer>(i)->OnDestroy();
				application.m_layers.erase(application.m_layers.begin() + index);
			}
			index++;
		}
	}
}
