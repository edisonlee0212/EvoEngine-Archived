#include "Console.hpp"
#include "Serialization.hpp"
using namespace EvoEngine;

std::string Serialization::GetSerializableTypeName(const size_t &typeId)
{
    return GetInstance().m_serializableNames.find(typeId)->second;
}

bool Serialization::RegisterDataComponentType(
    const std::string &typeName,
    const size_t &typeId,
    const std::function<std::shared_ptr<IDataComponent>(size_t &, size_t &)> &func)
{
    auto& serialization = GetInstance();
    if (serialization.m_dataComponentNames.find(typeId) != GetInstance().m_dataComponentNames.end())
    {
        EVOENGINE_ERROR("DataComponent already registered!");
        return false;
    }
    serialization.m_dataComponentNames[typeId] = typeName;
    serialization.m_dataComponentIds[typeName] = typeId;
    return serialization.m_dataComponentGenerators.insert({typeName, func}).second;
}

std::shared_ptr<IDataComponent> Serialization::ProduceDataComponent(
    const std::string &typeName, size_t &hashCode, size_t &size)
{
    auto &factory = GetInstance();
    const auto it = factory.m_dataComponentGenerators.find(typeName);
    if (it != factory.m_dataComponentGenerators.end())
    {
        return it->second(hashCode, size);
    }
    EVOENGINE_ERROR("DataComponent " + typeName + "is not registered!");
    throw 1;
}

bool Serialization::RegisterSerializableType(
    const std::string &typeName,
    const size_t &typeId,
    const std::function<std::shared_ptr<ISerializable>(size_t &)> &func)
{
    auto& serializationManger = GetInstance();
    if (serializationManger.m_serializableNames.find(typeId) != serializationManger.m_serializableNames.end())
    {
        EVOENGINE_ERROR(typeName + " already registered!");
        return false;
    }
    serializationManger.m_serializableNames[typeId] = typeName;
    serializationManger.m_serializableIds[typeName] = typeId;
    return serializationManger.m_serializableGenerators.insert({typeName, func}).second;
}
bool Serialization::RegisterPrivateComponentType(
    const std::string &typeName,
    const size_t &typeId,
    const std::function<void(std::shared_ptr<IPrivateComponent>, const std::shared_ptr<IPrivateComponent> &)> &cloneFunc)
{
    auto& serializationManger = GetInstance();
    return serializationManger.m_privateComponentCloners.insert({typeName, cloneFunc}).second;
}
std::shared_ptr<ISerializable> Serialization::ProduceSerializable(const std::string &typeName, size_t &hashCode)
{
    auto &serializationManager = GetInstance();
    const auto it = serializationManager.m_serializableGenerators.find(typeName);
    if (it != serializationManager.m_serializableGenerators.end())
    {
        auto retVal = it->second(hashCode);
        retVal->m_typeName = typeName;
        retVal->m_handle = Handle();
        return retVal;
    }
    EVOENGINE_ERROR("Serializable " + typeName + " is not registered!");
    return nullptr;
}
std::shared_ptr<ISerializable> Serialization::ProduceSerializable(const std::string &typeName, size_t &hashCode, const Handle &handle)
{
    auto &serializationManager = GetInstance();
    const auto it = serializationManager.m_serializableGenerators.find(typeName);
    if (it != serializationManager.m_serializableGenerators.end())
    {
        auto retVal = it->second(hashCode);
        retVal->m_typeName = typeName;
        retVal->m_handle = handle;
        return retVal;
    }
    EVOENGINE_ERROR("PrivateComponent " + typeName + " is not registered!");
    return nullptr;
}
YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::vec2 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << YAML::EndSeq;
    return out;
}

YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::vec3 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq;
    return out;
}

YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::vec4 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq;
    return out;
}

YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::quat &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq;
    return out;
}

YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::mat4 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v[0] << v[1] << v[2] << v[3] << YAML::EndSeq;
    return out;
}
YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::dvec2 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << YAML::EndSeq;
    return out;
}

YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::dvec3 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq;
    return out;
}

YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::dvec4 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq;
    return out;
}

YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::ivec2 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << YAML::EndSeq;
    return out;
}

YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::ivec3 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq;
    return out;
}

YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::ivec4 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq;
    return out;
}
YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::uvec2 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << YAML::EndSeq;
    return out;
}

YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::uvec3 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq;
    return out;
}

YAML::Emitter &EvoEngine::operator<<(YAML::Emitter &out, const glm::uvec4 &v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq;
    return out;
}

YAML::Emitter& EvoEngine::operator<<(YAML::Emitter& out, const glm::u16vec4& v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq;
    return out;
}

size_t Serialization::GetDataComponentTypeId(const std::string &typeName)
{
    auto& serializationManager = GetInstance();
    if(HasComponentDataType(typeName)){
        return serializationManager.m_dataComponentIds[typeName];
    }
    return 0;
}

void Serialization::SaveAssetList(const std::string& name, const std::vector<AssetRef>& target, YAML::Emitter& out)
{
    if (target.empty()) return;
    out << YAML::Key << name << YAML::Value << YAML::BeginSeq;
    for (auto& i : target) {
        out << YAML::BeginMap;
        i.Serialize(out);
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;
}

void Serialization::LoadAssetList(const std::string& name, std::vector<AssetRef>& target, const YAML::Node& in)
{
    if (in[name]) {
        target.clear();
        for (const auto& i : in[name]) {
            AssetRef instance;
            instance.Deserialize(i);
            target.push_back(instance);
        }
    }
}

size_t Serialization::GetSerializableTypeId(const std::string &typeName)
{
    auto& serializationManager = GetInstance();
    if(HasSerializableType(typeName)){
        return serializationManager.m_serializableIds[typeName];
    }
    return 0;
}
bool Serialization::HasSerializableType(const std::string &typeName)
{
    return GetInstance().m_serializableIds.find(typeName) != GetInstance().m_serializableIds.end();
}
bool Serialization::HasComponentDataType(const std::string &typeName)
{
    return GetInstance().m_dataComponentIds.find(typeName) != GetInstance().m_dataComponentIds.end();
}

void Serialization::ClonePrivateComponent(const std::shared_ptr<IPrivateComponent>& target, const std::shared_ptr<IPrivateComponent>& source)
{
    const auto targetTypeName = target->GetTypeName();
    const auto sourceTypeName = source->GetTypeName();
    assert(targetTypeName == sourceTypeName);
    if(auto& serializationManager = GetInstance(); serializationManager.HasSerializableType(targetTypeName)){
        serializationManager.m_privateComponentCloners[targetTypeName](target, source);
    }else {
        EVOENGINE_ERROR("PrivateComponent " + targetTypeName + "is not registered!");
    }
}
void Serialization::CloneSystem(const std::shared_ptr<ISystem>& target, const std::shared_ptr<ISystem> &source)
{
    auto targetTypeName = target->GetTypeName();
    auto sourceTypeName = source->GetTypeName();
    assert(targetTypeName == sourceTypeName);
    auto& serializationManager = GetInstance();
    if(serializationManager.HasSerializableType(targetTypeName)){
        serializationManager.m_systemCloners[targetTypeName](target, source);
    }else {
        EVOENGINE_ERROR("System " + targetTypeName + "is not registered!");
    }
}
bool Serialization::RegisterSystemType(
    const std::string &typeName,
    const std::function<void(std::shared_ptr<ISystem>, const std::shared_ptr<ISystem> &)> &cloneFunc)
{
    auto& serializationManger = GetInstance();
    return serializationManger.m_systemCloners.insert({typeName, cloneFunc}).second;
}

YAML::Emitter& EvoEngine::operator<<(YAML::Emitter& out, const glm::u8vec4& v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq;
    return out;
}

YAML::Emitter& EvoEngine::operator<<(YAML::Emitter& out, const glm::i8vec4& v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq;
    return out;
}

YAML::Emitter& EvoEngine::operator<<(YAML::Emitter& out, const glm::i16vec4& v)
{
    out << YAML::Flow;
    out << YAML::BeginSeq << v.x << v.y << v.z << v.w << YAML::EndSeq;
    return out;
}

