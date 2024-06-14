#include "Console.hpp"
#include "EditorLayer.hpp"
#include "Times.hpp"
using namespace evo_engine;

void Console::Log(const std::string& msg)
{
    const auto consoleLayer = Application::GetLayer<EditorLayer>();
    if (!consoleLayer) return;
    std::lock_guard lock(consoleLayer->m_consoleMessageMutex);
    ConsoleMessage cm;
    cm.m_value = msg;
    cm.m_type = ConsoleMessageType::Log;
    cm.m_time = Times::Now();
    consoleLayer->m_consoleMessages.push_back(cm);
    
}
void Console::Error(const std::string& msg)
{
    
    const auto consoleLayer = Application::GetLayer<EditorLayer>();
    if (!consoleLayer) return;
    std::lock_guard lock(consoleLayer->m_consoleMessageMutex);
    ConsoleMessage cm;
    cm.m_value = msg;
    cm.m_type = ConsoleMessageType::Error;
    cm.m_time = Times::Now();
    consoleLayer->m_consoleMessages.push_back(cm);
    
}

void Console::Warning(const std::string& msg)
{
    const auto consoleLayer = Application::GetLayer<EditorLayer>();
    if (!consoleLayer) return;
    std::lock_guard lock(consoleLayer->m_consoleMessageMutex);
    ConsoleMessage cm;
    cm.m_value = msg;
    cm.m_type = ConsoleMessageType::Warning;
    cm.m_time = Times::Now();
    consoleLayer->m_consoleMessages.push_back(cm);
}

