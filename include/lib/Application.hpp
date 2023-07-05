#pragma once
#include "ISingleton.hpp"
#include "Console.hpp"
namespace EvoEngine {
	struct ApplicationCreateInfo {
		std::filesystem::path m_projectPath;
		std::string m_applicationName = "Application";
		glm::ivec2 m_defaultWindowSize = { 1280, 720 };
		bool m_enableDocking = true;
		bool m_enableViewport = false;
		bool m_fullScreen = false;
	};

	class Application final : public ISingleton<Application>
	{
		std::string m_name;

		
		bool m_initialized = false;

		


	public:
		static void Initialize(const ApplicationCreateInfo& applicationCreateInfo);
		static void Start();
		static void Terminate();

		static void Play();
		static void Pause();
		static void Step();

		static void Stop();
	};
}