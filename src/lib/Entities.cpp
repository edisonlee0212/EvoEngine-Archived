#include "Entities.hpp"
#include "Entity.hpp"
#include "Scene.hpp"
using namespace evo_engine;

#pragma region EntityManager



size_t Entities::GetArchetypeChunkSize()
{
    auto &entityManager = GetInstance();
    return entityManager.archetype_chunk_size_;
}

EntityArchetype Entities::CreateEntityArchetype(const std::string &name, const std::vector<DataComponentType> &types)
{
    auto &entityManager = GetInstance();
    EntityArchetypeInfo entityArchetypeInfo;
    entityArchetypeInfo.archetype_name = name;
    std::vector<DataComponentType> actualTypes;
    actualTypes.push_back(Typeof<Transform>());
    actualTypes.push_back(Typeof<GlobalTransform>());
    actualTypes.push_back(Typeof<TransformUpdateFlag>());
    actualTypes.insert(actualTypes.end(), types.begin(), types.end());
    std::sort(actualTypes.begin() + 3, actualTypes.end(), ComponentTypeComparator);
    size_t offset = 0;
    DataComponentType prev = actualTypes[0];
    // Erase duplicates
    std::vector<DataComponentType> copy;
    copy.insert(copy.begin(), actualTypes.begin(), actualTypes.end());
    actualTypes.clear();
    for (const auto &i : copy)
    {
        bool found = false;
        for (const auto j : actualTypes)
        {
            if (i == j)
            {
                found = true;
                break;
            }
        }
        if (found)
            continue;
        actualTypes.push_back(i);
    }

    for (auto &i : actualTypes)
    {
        i.type_offset = offset;
        offset += i.type_size;
    }
    entityArchetypeInfo.data_component_types = actualTypes;
    entityArchetypeInfo.entity_size = entityArchetypeInfo.data_component_types.back().type_offset +
                                       entityArchetypeInfo.data_component_types.back().type_size;
    entityArchetypeInfo.chunk_capacity = entityManager.archetype_chunk_size_ / entityArchetypeInfo.entity_size;
    return CreateEntityArchetypeHelper(entityArchetypeInfo);
}

EntityArchetype Entities::GetDefaultEntityArchetype()
{
    auto &entityManager = GetInstance();
    return entityManager.basic_archetype_;
}

EntityArchetypeInfo Entities::GetArchetypeInfo(const EntityArchetype &entity_archetype)
{
    auto &entityManager = GetInstance();
    return entityManager.entity_archetype_infos_[entity_archetype.index_];
}


EntityQuery Entities::CreateEntityQuery()
{
    EntityQuery retVal;
    auto &entityManager = GetInstance();
    retVal.index_ = entityManager.entity_query_infos_.size();
    EntityQueryInfo info;
    info.query_index = retVal.index_;
    entityManager.entity_query_infos_.resize(entityManager.entity_query_infos_.size() + 1);
    entityManager.entity_query_infos_[info.query_index] = info;
    return retVal;
}


std::string Entities::GetEntityArchetypeName(const EntityArchetype &entity_archetype)
{
    auto &entityManager = GetInstance();
    return entityManager.entity_archetype_infos_[entity_archetype.index_].archetype_name;
}

void Entities::SetEntityArchetypeName(const EntityArchetype &entity_archetype, const std::string &name)
{
    auto &entityManager = GetInstance();
    entityManager.entity_archetype_infos_[entity_archetype.index_].archetype_name = name;
}

void Entities::Initialize()
{
    auto &entityManager = GetInstance();
    entityManager.entity_archetype_infos_.emplace_back();
    entityManager.entity_query_infos_.emplace_back();

    entityManager.basic_archetype_ =
        CreateEntityArchetype("Basic", Transform(), GlobalTransform(), TransformUpdateFlag());
}

EntityArchetype Entities::CreateEntityArchetypeHelper(const EntityArchetypeInfo &info)
{
    EntityArchetype retVal = EntityArchetype();
    auto &entityManager = GetInstance();
    auto &entityArchetypeInfos = entityManager.entity_archetype_infos_;
    int duplicateIndex = -1;
    for (size_t i = 1; i < entityArchetypeInfos.size(); i++)
    {
        EntityArchetypeInfo &compareInfo = entityArchetypeInfos[i];
        if (info.chunk_capacity != compareInfo.chunk_capacity)
            continue;
        if (info.entity_size != compareInfo.entity_size)
            continue;
        bool typeCheck = true;

        for (auto &componentType : info.data_component_types)
        {
            if (!compareInfo.HasType(componentType.type_index))
                typeCheck = false;
        }
        if (typeCheck)
        {
            duplicateIndex = i;
            break;
        }
    }
    if (duplicateIndex == -1)
    {
        retVal.index_ = entityArchetypeInfos.size();
        entityArchetypeInfos.push_back(info);
    }
    else
    {
        retVal.index_ = duplicateIndex;
    }
    return retVal;
}

#pragma endregion
