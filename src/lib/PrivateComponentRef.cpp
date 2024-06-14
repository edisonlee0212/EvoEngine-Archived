//
// Created by Bosheng Li on 10/10/2021.
//

#include "PrivateComponentRef.hpp"
//#include "ProjectManager.hpp"
#include "Entities.hpp"
#include "Scene.hpp"
using namespace evo_engine;
bool PrivateComponentRef::Update()
{
    if (m_entityHandle.GetValue() == 0 || m_scene.expired())
    {
        Clear();
        return false;
    }
    if (m_value.expired())
    {
        auto scene = m_scene.lock();
        auto entity = scene->GetEntity(m_entityHandle);
        if (entity.GetIndex() != 0)
        {
            if (scene->HasPrivateComponent(entity, m_privateComponentTypeName))
            {
                m_value = scene->GetPrivateComponent(entity, m_privateComponentTypeName);
                return true;
            }
        }
        Clear();
        return false;
    }
    return true;
}
void PrivateComponentRef::Clear()
{
    m_value.reset();
    m_entityHandle = m_handle = Handle(0);
    m_scene.reset();
    m_privateComponentTypeName = {};
}

