#include "Lights.hpp"
#include "Serialization.hpp"
using namespace EvoEngine;


void SpotLight::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
    ImGui::Checkbox("Cast Shadow", &m_castShadow);
    ImGui::ColorEdit3("Color", &m_diffuse[0]);
    ImGui::DragFloat("Intensity", &m_diffuseBrightness, 0.1f, 0.0f, 999.0f);
    ImGui::DragFloat("Bias", &m_bias, 0.001f, 0.0f, 999.0f);

    ImGui::DragFloat("Constant", &m_constant, 0.01f, 0.0f, 999.0f);
    ImGui::DragFloat("Linear", &m_linear, 0.001f, 0, 1, "%.3f");
    ImGui::DragFloat("Quadratic", &m_quadratic, 0.001f, 0, 10, "%.4f");

    ImGui::DragFloat("Inner Degrees", &m_innerDegrees, 0.1f, 0.0f, m_outerDegrees);
    ImGui::DragFloat("Outer Degrees", &m_outerDegrees, 0.1f, m_innerDegrees, 180.0f);
    ImGui::DragFloat("Light Size", &m_lightSize, 0.01f, 0.0f, 999.0f);
}

void SpotLight::OnCreate()
{
    SetEnabled(true);
}

void SpotLight::Serialize(YAML::Emitter& out)
{
    out << YAML::Key << "m_castShadow" << YAML::Value << m_castShadow;
    out << YAML::Key << "m_innerDegrees" << YAML::Value << m_innerDegrees;
    out << YAML::Key << "m_outerDegrees" << YAML::Value << m_outerDegrees;
    out << YAML::Key << "m_constant" << YAML::Value << m_constant;
    out << YAML::Key << "m_linear" << YAML::Value << m_linear;
    out << YAML::Key << "m_quadratic" << YAML::Value << m_quadratic;
    out << YAML::Key << "m_bias" << YAML::Value << m_bias;
    out << YAML::Key << "m_diffuse" << YAML::Value << m_diffuse;
    out << YAML::Key << "m_diffuseBrightness" << YAML::Value << m_diffuseBrightness;
    out << YAML::Key << "m_lightSize" << YAML::Value << m_lightSize;
}

void SpotLight::Deserialize(const YAML::Node& in)
{
    m_castShadow = in["m_castShadow"].as<bool>();
    m_innerDegrees = in["m_innerDegrees"].as<float>();
    m_outerDegrees = in["m_outerDegrees"].as<float>();
    m_constant = in["m_constant"].as<float>();
    m_linear = in["m_linear"].as<float>();
    m_quadratic = in["m_quadratic"].as<float>();
    m_bias = in["m_bias"].as<float>();
    m_diffuse = in["m_diffuse"].as<glm::vec3>();
    m_diffuseBrightness = in["m_diffuseBrightness"].as<float>();
    m_lightSize = in["m_lightSize"].as<float>();
}

float PointLight::GetFarPlane() const
{
    float lightMax = glm::max(glm::max(m_diffuse.x, m_diffuse.y), m_diffuse.z);
    return (-m_linear + glm::sqrt(m_linear * m_linear - 4 * m_quadratic * (m_constant - (256.0 / 5.0) * lightMax)))
        / (2 * m_quadratic);
}

float SpotLight::GetFarPlane() const
{
    float lightMax = glm::max(glm::max(m_diffuse.x, m_diffuse.y), m_diffuse.z);
    return (-m_linear + glm::sqrt(m_linear * m_linear - 4 * m_quadratic * (m_constant - (256.0 / 5.0) * lightMax)))
        / (2 * m_quadratic);
}

void PointLight::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
    ImGui::Checkbox("Cast Shadow", &m_castShadow);
    ImGui::ColorEdit3("Color", &m_diffuse[0]);
    ImGui::DragFloat("Intensity", &m_diffuseBrightness, 0.1f, 0.0f, 999.0f);
    ImGui::DragFloat("Bias", &m_bias, 0.001f, 0.0f, 999.0f);

    ImGui::DragFloat("Constant", &m_constant, 0.01f, 0.0f, 999.0f);
    ImGui::DragFloat("Linear", &m_linear, 0.0001f, 0, 1, "%.4f");
    ImGui::DragFloat("Quadratic", &m_quadratic, 0.00001f, 0, 10, "%.5f");

    // ImGui::InputFloat("Normal Offset", &dl->normalOffset, 0.01f);
    ImGui::DragFloat("Light Size", &m_lightSize, 0.01f, 0.0f, 999.0f);
}

void PointLight::OnCreate()
{
    SetEnabled(true);
}

void PointLight::Serialize(YAML::Emitter& out)
{
    out << YAML::Key << "m_castShadow" << YAML::Value << m_castShadow;
    out << YAML::Key << "m_constant" << YAML::Value << m_constant;
    out << YAML::Key << "m_linear" << YAML::Value << m_linear;
    out << YAML::Key << "m_quadratic" << YAML::Value << m_quadratic;
    out << YAML::Key << "m_bias" << YAML::Value << m_bias;
    out << YAML::Key << "m_diffuse" << YAML::Value << m_diffuse;
    out << YAML::Key << "m_diffuseBrightness" << YAML::Value << m_diffuseBrightness;
    out << YAML::Key << "m_lightSize" << YAML::Value << m_lightSize;
}

void PointLight::Deserialize(const YAML::Node& in)
{
    m_castShadow = in["m_castShadow"].as<bool>();
    m_constant = in["m_constant"].as<float>();
    m_linear = in["m_linear"].as<float>();
    m_quadratic = in["m_quadratic"].as<float>();
    m_bias = in["m_bias"].as<float>();
    m_diffuse = in["m_diffuse"].as<glm::vec3>();
    m_diffuseBrightness = in["m_diffuseBrightness"].as<float>();
    m_lightSize = in["m_lightSize"].as<float>();
}

void DirectionalLight::OnCreate()
{
    SetEnabled(true);
}

void DirectionalLight::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
    ImGui::Checkbox("Cast Shadow", &m_castShadow);
    ImGui::ColorEdit3("Color", &m_diffuse[0]);
    ImGui::DragFloat("Intensity", &m_diffuseBrightness, 0.1f, 0.0f, 999.0f);
    ImGui::DragFloat("Bias", &m_bias, 0.001f, 0.0f, 999.0f);
    ImGui::DragFloat("Normal Offset", &m_normalOffset, 0.001f, 0.0f, 999.0f);
    ImGui::DragFloat("Light Size", &m_lightSize, 0.01f, 0.0f, 999.0f);
}

void DirectionalLight::Serialize(YAML::Emitter& out)
{
    out << YAML::Key << "m_castShadow" << YAML::Value << m_castShadow;
    out << YAML::Key << "m_bias" << YAML::Value << m_bias;
    out << YAML::Key << "m_diffuse" << YAML::Value << m_diffuse;
    out << YAML::Key << "m_diffuseBrightness" << YAML::Value << m_diffuseBrightness;
    out << YAML::Key << "m_lightSize" << YAML::Value << m_lightSize;
    out << YAML::Key << "m_normalOffset" << YAML::Value << m_normalOffset;
}

void DirectionalLight::Deserialize(const YAML::Node& in)
{
    m_castShadow = in["m_castShadow"].as<bool>();
    m_bias = in["m_bias"].as<float>();
    m_diffuse = in["m_diffuse"].as<glm::vec3>();
    m_diffuseBrightness = in["m_diffuseBrightness"].as<float>();
    m_lightSize = in["m_lightSize"].as<float>();
    m_normalOffset = in["m_normalOffset"].as<float>();
}
void DirectionalLight::PostCloneAction(const std::shared_ptr<IPrivateComponent>& target)
{
}

void PointLight::PostCloneAction(const std::shared_ptr<IPrivateComponent>& target)
{
}

void SpotLight::PostCloneAction(const std::shared_ptr<IPrivateComponent>& target)
{
}
