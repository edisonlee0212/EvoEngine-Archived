#pragma once
#include <IHandle.hpp>
namespace YAML
{
class Emitter;
class Node;
} // namespace YAML

namespace EvoEngine
{
class ISerializable : public IHandle
{
    friend class Serialization;
    friend class IAsset;
    friend class Entities;
    friend class Scene;
    friend class Serialization;
    friend class EntityMetadata;
    friend class AssetRecord;
    friend class Folder;
    std::string m_typeName;
  public:
    void Save(const std::string &name, YAML::Emitter &out) const;
    void Load(const std::string &name, const YAML::Node &in);
    [[nodiscard]] std::string GetTypeName()
    {
        return m_typeName;
    }
    virtual ~ISerializable() = default;
    virtual void Serialize(YAML::Emitter &out) const
    {
    }
    virtual void Deserialize(const YAML::Node &in)
    {
    }
};
} // namespace EvoEngine
