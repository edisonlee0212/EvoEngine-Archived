#pragma once
#include "ISingleton.hpp"
#include "Console.hpp"
#include "ILayer.hpp"

namespace EvoEngine {
	class ApplicationTime
	{
		friend class Scene;
		friend class Application;
		double m_lastFixedUpdateTime = 0;
		double m_lastUpdateTime = 0;
		double m_timeStep = 0.016;
		double m_frameStartTime = 0;
		double m_deltaTime = 0;
		double m_fixedUpdateTimeStamp = 0;
		void StartFixedUpdate();
		void EndFixedUpdate();

	public:
		void Reset();
		void OnInspect();
		void SetTimeStep(const double& value);
		[[nodiscard]] double CurrentTime() const;
		[[nodiscard]] double TimeStep() const;
		[[nodiscard]] double FixedDeltaTime() const;
		[[nodiscard]] double DeltaTime() const;
		[[nodiscard]] double LastFrameTime() const;
	};

	struct ApplicationCreateInfo {
		std::filesystem::path m_projectPath;
		std::string m_applicationName = "Application";
		glm::ivec2 m_defaultWindowSize = { 1280, 720 };
		bool m_enableDocking = true;
		bool m_enableViewport = false;
		bool m_fullScreen = false;
	};

	enum class ApplicationStatus {
		Uninitialized,
		Initialized,
		OnDestroy
	};
	enum class GameStatus {
		Stop,
		Pause,
		Step,
		Playing
	};

	class Application final : public ISingleton<Application>
	{
		std::string m_name;
		bool m_initialized = false;

		ApplicationStatus m_applicationStatus = ApplicationStatus::Uninitialized;
		GameStatus m_gameStatus = GameStatus::Stop;

		static void PreUpdateInternal();
		static void UpdateInternal();
		static void LateUpdateInternal();

		std::vector<std::shared_ptr<ILayer>> m_layers;
		std::shared_ptr<Scene> m_activeScene;

		std::vector<std::function<void()>> m_externalPreUpdateFunctions;
		std::vector<std::function<void()>> m_externalUpdateFunctions;
		std::vector<std::function<void()>> m_externalFixedUpdateFunctions;
		std::vector<std::function<void()>> m_externalLateUpdateFunctions;

		ApplicationTime m_time;

	public:
		static std::shared_ptr<Scene> GetActiveScene();

		static void Initialize(const ApplicationCreateInfo& applicationCreateInfo);
		static void Start();
		static void Terminate();

		static void Play();
		static void Pause();
		static void Step();

		static void Stop();
	};
}
