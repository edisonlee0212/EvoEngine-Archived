#include <pybind11/pybind11.h>

#include "AnimationLayer.hpp"
#include "Application.hpp"
#include "WindowLayer.hpp"
#include "RenderLayer.hpp"
#include "EditorLayer.hpp"
using namespace EvoEngine;
namespace py = pybind11;

int RunApplication() {
    Application::PushLayer<WindowLayer>();
    Application::PushLayer<EditorLayer>();
    Application::PushLayer<RenderLayer>();
    Application::PushLayer<AnimationLayer>();

    ApplicationInfo applicationInfo;
    Application::Initialize(applicationInfo);
    Application::Start();

    Application::Terminate();
    return 0;
}

PYBIND11_MODULE(PyEvoEngine, m) {
    m.doc() = "EvoEngine"; // optional module docstring

    m.def("RunApplication", &RunApplication, "Run EvoEngine");
}