#pragma once
namespace EvoEngine {
	class Times
	{
		friend class Scene;
		friend class Application;
		static std::chrono::time_point<std::chrono::system_clock> m_startTime;
		static std::chrono::time_point<std::chrono::system_clock> m_lastFixedUpdateTime;
		static std::chrono::time_point<std::chrono::system_clock> m_lastUpdateTime;
		static double m_deltaTime;
		static double m_fixedDeltaTime;
		static size_t m_frames;
		static size_t m_steps;
		static double m_timeStep;
	public:
		static void OnInspect();
		static void SetTimeStep(double value);
		[[nodiscard]] static double TimeStep();
		[[nodiscard]] static double CurrentTime();
		[[nodiscard]] static double FixedDeltaTime();
		[[nodiscard]] static double DeltaTime();
		[[nodiscard]] static double LastUpdateTime();
		[[nodiscard]] static double LastFixedUpdateTime();
	};
}