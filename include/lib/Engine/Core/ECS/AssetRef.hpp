#pragma once
#include <IHandle.hpp>
#include <IAsset.hpp>
namespace EvoEngine
{
class AssetRef final : public ISerializable
{
    friend class Prefab;
    
    friend class EditorLayer;
    std::shared_ptr<IAsset> m_value = {};
    Handle m_assetHandle = Handle(0);
    std::string m_assetTypeName;
    bool Update();
  public:
    void Serialize(YAML::Emitter &out) const override
    {
        out << YAML::Key << "m_assetHandle" << YAML::Value << m_assetHandle;
        out << YAML::Key << "m_assetTypeName" << YAML::Value << m_assetTypeName;
    }
    void Deserialize(const YAML::Node &in) override
    {
        m_assetHandle = Handle(in["m_assetHandle"].as<uint64_t>());
        m_assetTypeName = in["m_assetTypeName"].as<std::string>();
        Update();
    }
    AssetRef()
    {
        m_assetHandle = Handle(0);
        m_assetTypeName = "";
        m_value.reset();
    }
    ~AssetRef() override
    {
        m_assetHandle = Handle(0);
        m_assetTypeName = "";
        m_value.reset();
    }
    template <typename T = IAsset> AssetRef(const std::shared_ptr<T> &other)
    {
        Set(other);
    }
    template <typename T = IAsset> AssetRef &operator=(const std::shared_ptr<T> &other)
    {
        Set(other);
        return *this;
    }
    template <typename T = IAsset> AssetRef &operator=(std::shared_ptr<T> &&other) noexcept
    {
        Set(other);
        return *this;
    }
    bool operator==(const AssetRef &rhs) const
    {
        return m_assetHandle == rhs.m_assetHandle;
    }
    bool operator!=(const AssetRef &rhs) const
    {
        return m_assetHandle != rhs.m_assetHandle;
    }

    template <typename T = IAsset> [[nodiscard]] std::shared_ptr<T> Get()
    {
        if (Update())
        {
            return std::dynamic_pointer_cast<T>(m_value);
        }
        return nullptr;
    }
    template <typename T = IAsset> void Set(std::shared_ptr<T> target)
    {
        if (target)
        {
            auto asset = std::dynamic_pointer_cast<IAsset>(target);
            m_assetTypeName = asset->GetTypeName();
            m_assetHandle = asset->GetHandle();
            m_value = asset;
        }
        else
        {
            m_assetHandle = Handle(0);
            m_value.reset();
        }
    }
    void Set(const AssetRef &target);
    void Clear();
    [[nodiscard]] Handle GetAssetHandle() const
    {
        return m_assetHandle;
    }
};
} // namespace EvoEngine