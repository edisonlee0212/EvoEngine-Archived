//
// Created by Bosheng Li on 10/12/2021.
//

#include "AssetRef.hpp"
#include "ProjectManager.hpp"

using namespace evo_engine;
bool AssetRef::Update()
{
    if (m_assetHandle.GetValue() == 0)
    {
        m_value.reset();
        return false;
    }
    
    if (!m_value)
    {
	    if (const auto ptr = ProjectManager::GetAsset(m_assetHandle))
        {
            m_value = ptr;
            m_assetTypeName = ptr->GetTypeName();
            return true;
        }
        Clear();
        return false;
    }
    
    return true;
}

void AssetRef::Clear()
{
    m_value.reset();
    m_assetHandle = Handle(0);
}
void AssetRef::Set(const AssetRef &target)
{
    m_assetHandle = target.m_assetHandle;
    Update();
}