#include "Time.hpp"
using namespace EvoEngine;

double Time::m_timeStep = 0.016;
size_t Time::m_frames = 0;
size_t Time::m_steps = 0;
std::chrono::time_point<std::chrono::system_clock> Time::m_startTime = {};
std::chrono::time_point<std::chrono::system_clock> Time::m_lastFixedUpdateTime = {};
std::chrono::time_point<std::chrono::system_clock> Time::m_lastUpdateTime = {};

void Time::OnInspect()
{
	if (ImGui::CollapsingHeader("Time Settings"))
	{
		float timeStep = m_timeStep;
		if (ImGui::DragFloat("Time step", &timeStep, 0.001f, 0.001f, 1.0f))
		{
			m_timeStep = timeStep;
		}
	}
}

double Time::TimeStep()
{
	return m_timeStep;
}
void Time::SetTimeStep(const double value)
{
	m_timeStep = value;
}
double Time::FixedDeltaTime()
{
	const auto now = std::chrono::system_clock::now();
	const std::chrono::duration<double> duration = now - m_lastFixedUpdateTime;
	return duration.count();
}

double Time::DeltaTime()
{
	const auto now = std::chrono::system_clock::now();
	const std::chrono::duration<double> duration = now - m_lastUpdateTime;
	return duration.count();
}

double Time::CurrentTime()
{
	const auto now = std::chrono::system_clock::now();
	const std::chrono::duration<double> duration = now - m_startTime;
	return duration.count();
}

double Time::LastUpdateTime()
{
	const std::chrono::duration<double> duration = m_lastUpdateTime - m_startTime;
	return duration.count();
}

double Time::LastFixedUpdateTime()
{
	const std::chrono::duration<double> duration = m_lastFixedUpdateTime - m_startTime;
	return duration.count();
}