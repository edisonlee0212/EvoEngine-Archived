#pragma once
namespace EvoEngine
{
/**
 * The "GUID" for all instances in EvoEngine that requires a unique identifier for hashing/serialization.
 */
struct Handle
{
    friend class IAsset;
    friend class EntityMetadata;
    friend class Resources;

    /**
     * Default constructor, will allocate a random number to the handle. EvoEngine will not handle the collision since possibility is extremely small and negligible.
     */
    Handle();
    /**
     * Constructor that takes a value for the handle.
     */
    Handle(uint64_t value);
    Handle(const Handle &other);

    operator uint64_t()
    {
        return m_value;
    }
    operator const uint64_t() const
    {
        return m_value;
    }
    [[nodiscard]] uint64_t GetValue() const {
        return m_value;
    }
  private:
    uint64_t m_value;
};
class IHandle{
    friend class Prefab;
    friend class Entities;
    friend class EntityMetadata;
    
    friend class EditorLayer;
    friend class Resources;
    friend class Serialization;
    friend class IAsset;
    friend class AssetRef;
    friend class Scene;
    friend class AssetRecord;
    friend class Folder;
    friend class PrivateComponentStorage;
    Handle m_handle;
  public:
    Handle GetHandle() const
    {
        return m_handle;
    }
};
} // namespace EvoEngine

namespace std
{
template <>
struct hash<EvoEngine::Handle>
    {
    size_t operator()(const EvoEngine::Handle &handle) const
    {
        return hash<uint64_t>()((uint64_t)handle);
    }
    };
}