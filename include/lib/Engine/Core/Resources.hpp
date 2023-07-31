#pragma once
#include "AssetRef.hpp"
#include "ISingleton.hpp"
#include "Serialization.hpp"
namespace EvoEngine
{
	class Resources : ISingleton<Resources>
	{
		Handle m_currentMaxHandle = Handle(1);
		std::unordered_map<std::string, std::unordered_map<Handle, std::shared_ptr<IAsset>>> m_typedResources;
		std::unordered_map<std::string, std::shared_ptr<IAsset>> m_namedResources;
		std::unordered_map<Handle, std::string> m_resourceNames;
		std::unordered_map<Handle, std::shared_ptr<IAsset>> m_resources;
		void LoadShaders();
		void LoadPrimitives() const;
		bool m_showLoadedAssets = false;
		static void Initialize();
		static void LateInitialization();
		[[nodiscard]] Handle GenerateNewHandle();
		friend class ProjectManager;
		friend class Application;
	public:
		static void OnInspect(const std::shared_ptr<EditorLayer>& editorLayer);
		template <class T>
		static std::shared_ptr<T> CreateResource(const std::string& name);

		[[nodiscard]] static bool IsResource(const Handle& handle);
		[[nodiscard]] static bool IsResource(const std::shared_ptr<IAsset>& target);
		[[nodiscard]] static bool IsResource(const AssetRef& target);
		[[nodiscard]] static std::shared_ptr<IAsset> GetResource(const std::string& name);
		[[nodiscard]] static std::shared_ptr<IAsset> GetResource(const Handle& handle);
	};

	template <class T>
	std::shared_ptr<T> Resources::CreateResource(const std::string& name)
	{
		auto& resources = GetInstance();
		assert(resources.m_namedResources.find(name) == resources.m_namedResources.end());
		auto typeName = Serialization::GetSerializableTypeName<T>();
		const auto handle = resources.GenerateNewHandle();
		auto retVal = std::dynamic_pointer_cast<IAsset>(Serialization::ProduceSerializable<T>());
		retVal->m_self = retVal;
		retVal->m_handle = handle;

		resources.m_resourceNames[handle] = name;
		resources.m_typedResources[typeName][handle] = retVal;
		resources.m_namedResources[name] = retVal;
		resources.m_resources[handle] = retVal;
		retVal->OnCreate();
		return std::dynamic_pointer_cast<T>(retVal);
	}
}
