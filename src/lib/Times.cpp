#include "Times.hpp"
using namespace EvoEngine;

double Times::m_timeStep = 0.016;
double Times::m_deltaTime = 0;
double Times::m_fixedDeltaTime = 0;
size_t Times::m_frames = 0;
size_t Times::m_steps = 0;
std::chrono::time_point<std::chrono::system_clock> Times::m_startTime = {};
std::chrono::time_point<std::chrono::system_clock> Times::m_lastFixedUpdateTime = {};
std::chrono::time_point<std::chrono::system_clock> Times::m_lastUpdateTime = {};

void Times::OnInspect()
{
	if (ImGui::CollapsingHeader("Times Settings"))
	{
		float timeStep = m_timeStep;
		if (ImGui::DragFloat("Times step", &timeStep, 0.001f, 0.001f, 1.0f))
		{
			m_timeStep = timeStep;
		}
	}
}

double Times::TimeStep()
{
	return m_timeStep;
}
void Times::SetTimeStep(const double value)
{
	m_timeStep = value;
}
double Times::FixedDeltaTime()
{
	return m_fixedDeltaTime;
}

double Times::DeltaTime()
{
	return m_deltaTime;
}

double Times::Now()
{
	const auto now = std::chrono::system_clock::now();
	const std::chrono::duration<double> duration = now - m_startTime;
	return duration.count();
}

double Times::LastUpdateTime()
{
	const std::chrono::duration<double> duration = m_lastUpdateTime - m_startTime;
	return duration.count();
}

double Times::LastFixedUpdateTime()
{
	const std::chrono::duration<double> duration = m_lastFixedUpdateTime - m_startTime;
	return duration.count();
}