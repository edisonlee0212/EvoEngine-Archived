#pragma once
#include "Serialization.hpp"
#include "Entity.hpp"
#include "IAsset.hpp"
#include "IPrivateComponent.hpp"
#include "ISerializable.hpp"
namespace EvoEngine
{
	class PrivateComponentRef final : public ISerializable
	{
		friend class Prefab;
		friend class Scene;
		std::weak_ptr<IPrivateComponent> m_value;
		Handle m_entityHandle = Handle(0);
		std::weak_ptr<Scene> m_scene;
		std::string m_privateComponentTypeName{};
		bool Update();

	public:
		void Serialize(YAML::Emitter& out) const override
		{
			out << YAML::Key << "m_entityHandle" << YAML::Value << m_entityHandle;
			out << YAML::Key << "m_privateComponentTypeName" << YAML::Value << m_privateComponentTypeName;
		}

		void Deserialize(const YAML::Node& in) override
		{
			m_entityHandle = Handle(in["m_entityHandle"].as<uint64_t>());
			m_privateComponentTypeName = in["m_privateComponentTypeName"].as<std::string>();
			m_scene.reset();
		}

		void Deserialize(const YAML::Node& in, const std::shared_ptr<Scene>& scene)
		{
			m_entityHandle = Handle(in["m_entityHandle"].as<uint64_t>());
			m_privateComponentTypeName = in["m_privateComponentTypeName"].as<std::string>();
			m_scene = scene;
		}

		PrivateComponentRef()
		{
			m_entityHandle = Handle(0);
			m_privateComponentTypeName = "";
			m_scene.reset();
		}

		template <typename T = IPrivateComponent> PrivateComponentRef(const std::shared_ptr<T>& other)
		{
			Set(other);
		}
		template <typename T = IPrivateComponent> PrivateComponentRef& operator=(const std::shared_ptr<T>& other)
		{
			Set(other);
			return *this;
		}
		template <typename T = IPrivateComponent> PrivateComponentRef& operator=(std::shared_ptr<T>&& other) noexcept
		{
			Set(other);
			return *this;
		}

		void Relink(const std::unordered_map<Handle, Handle>& map, const std::shared_ptr<Scene>& scene)
		{
			auto search = map.find(m_entityHandle);
			if (search != map.end())
			{
				m_entityHandle = search->second;
				m_value.reset();
				m_scene = scene;
			}
			else
				Clear();
		}

		void ResetScene(const std::shared_ptr<Scene>& scene)
		{
			m_value.reset();
			m_scene = scene;
		}

		template <typename T = IPrivateComponent> [[nodiscard]] std::shared_ptr<T> Get()
		{
			if (Update())
			{
				return std::dynamic_pointer_cast<T>(m_value.lock());
			}
			return nullptr;
		}
		template <typename T = IPrivateComponent> void Set(const std::shared_ptr<T>& target)
		{
			if (target)
			{
				auto privateComponent = std::dynamic_pointer_cast<IPrivateComponent>(target);
				m_scene = privateComponent->GetScene();
				m_privateComponentTypeName = privateComponent->GetTypeName();
				m_entityHandle = privateComponent->GetScene()->GetEntityHandle(privateComponent->GetOwner());
				m_value = privateComponent;
				m_handle = privateComponent->GetHandle();
			}
			else
			{
				Clear();
			}
		}

		void Clear();

		[[nodiscard]] Handle GetEntityHandle() const
		{
			return m_entityHandle;
		}

		void Load(const std::string& name, const YAML::Node& in, const std::shared_ptr<Scene>& scene)
		{
			if (in[name]) Deserialize(in[name], scene);
		}
	};
} // namespace EvoEngine