#include "PhysicsMaterial.hpp"
#include "Application.hpp"
#include "PhysicsLayer.hpp"
void EvoEngine::PhysicsMaterial::OnCreate()
{
    auto physicsLayer = Application::GetLayer<PhysicsLayer>();
    if(!physicsLayer) return;
    m_value = physicsLayer->m_physics->createMaterial(m_staticFriction, m_dynamicFriction, m_restitution);
}
EvoEngine::PhysicsMaterial::~PhysicsMaterial()
{
    if (m_value)
        m_value->release();
}
void EvoEngine::PhysicsMaterial::SetDynamicFriction(const float &value)
{
    m_dynamicFriction = value;
    m_value->setDynamicFriction(m_dynamicFriction);
    m_saved = false;
}
void EvoEngine::PhysicsMaterial::SetStaticFriction(const float &value)
{
    m_staticFriction = value;
    m_value->setStaticFriction(m_staticFriction);
    m_saved = false;
}
void EvoEngine::PhysicsMaterial::SetRestitution(const float &value)
{
    m_restitution = value;
    m_value->setRestitution(m_restitution);
    m_saved = false;
}
void EvoEngine::PhysicsMaterial::OnGui()
{
    if (ImGui::DragFloat("Dynamic Friction", &m_dynamicFriction))
    {
        SetDynamicFriction(m_dynamicFriction);
    }
    if (ImGui::DragFloat("Static Friction", &m_staticFriction))
    {
        SetStaticFriction(m_staticFriction);
    }
    if (ImGui::DragFloat("Restitution", &m_restitution))
    {
        SetRestitution(m_restitution);
    }
}
void EvoEngine::PhysicsMaterial::Serialize(YAML::Emitter &out)
{
    out << YAML::Key << "m_staticFriction" << YAML::Value << m_staticFriction;
    out << YAML::Key << "m_dynamicFriction" << YAML::Value << m_dynamicFriction;
    out << YAML::Key << "m_restitution" << YAML::Value << m_restitution;
}
void EvoEngine::PhysicsMaterial::Deserialize(const YAML::Node &in)
{
    m_staticFriction = in["m_staticFriction"].as<float>();
    m_restitution = in["m_restitution"].as<float>();
    m_dynamicFriction = in["m_dynamicFriction"].as<float>();
    SetStaticFriction(m_staticFriction);
    SetRestitution(m_restitution);
    SetDynamicFriction(m_dynamicFriction);
}