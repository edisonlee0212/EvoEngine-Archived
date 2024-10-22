#include "ProjectManager.hpp"
#include "Entities.hpp"
#include "Application.hpp"
#include "Scene.hpp"
#include "EditorLayer.hpp"
#include "Lights.hpp"
#include "MeshRenderer.hpp"
#include "Prefab.hpp"
#include "Resources.hpp"
#include "TransformGraph.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#include "shellapi.h"
#endif

using namespace EvoEngine;

std::shared_ptr<IAsset> AssetRecord::GetAsset()
{
	if (!m_asset.expired())
		return m_asset.lock();
	if (!m_assetTypeName.empty() && m_assetTypeName != "Binary" && m_assetHandle != 0)
	{
		size_t hashCode;
		auto retVal = std::dynamic_pointer_cast<IAsset>(
			Serialization::ProduceSerializable(m_assetTypeName, hashCode, m_assetHandle));
		retVal->m_assetRecord = m_self;
		retVal->m_self = retVal;
		retVal->OnCreate();
		auto absolutePath = GetAbsolutePath();
		if (std::filesystem::exists(absolutePath))
		{
			retVal->Load();
		}
		else
		{
			retVal->Save();
		}
		m_asset = retVal;
		auto& projectManager = ProjectManager::GetInstance();
		projectManager.m_assetRegistry[m_assetHandle] = retVal;
		projectManager.m_loadedAssets[m_assetHandle] = retVal;
		projectManager.m_assetRecordRegistry[m_assetHandle] = m_self;
		return retVal;
	}
	return nullptr;
}
std::string AssetRecord::GetAssetTypeName() const
{
	return m_assetTypeName;
}
std::string AssetRecord::GetAssetFileName() const
{
	return m_assetFileName;
}
std::string AssetRecord::GetAssetExtension() const
{
	return m_assetExtension;
}
std::filesystem::path AssetRecord::GetProjectRelativePath() const
{
	if (m_folder.expired())
	{
		EVOENGINE_ERROR("Folder expired!");
		return {};
	}
	return m_folder.lock()->GetProjectRelativePath() / (m_assetFileName + m_assetExtension);
}
std::filesystem::path AssetRecord::GetAbsolutePath() const
{
	if (m_folder.expired())
	{
		EVOENGINE_ERROR("Folder expired!");
		return {};
	}
	return m_folder.lock()->GetAbsolutePath() / (m_assetFileName + m_assetExtension);
}
void AssetRecord::SetAssetFileName(const std::string& newName)
{
	if (m_assetFileName == newName)
		return;
	// TODO: Check invalid filename.
	auto oldPath = GetAbsolutePath();
	auto newPath = oldPath;
	newPath.replace_filename(newName + oldPath.extension().string());
	if (std::filesystem::exists(newPath))
	{
		EVOENGINE_ERROR("File with new name already exists!");
		return;
	}
	DeleteMetadata();
	m_assetFileName = newName;
	if (std::filesystem::exists(oldPath))
	{
		std::filesystem::rename(oldPath, newPath);
	}
	Save();
}
void AssetRecord::SetAssetExtension(const std::string& newExtension)
{
	if (m_assetTypeName == "Binary")
	{
		EVOENGINE_ERROR("File is binary!");
		return;
	}
	auto validExtensions = ProjectManager::GetExtension(m_assetTypeName);
	bool found = false;
	for (const auto& i : validExtensions)
	{
		if (i == newExtension)
		{
			found = true;
			break;
		}
	}
	if (!found)
	{
		EVOENGINE_ERROR("Extension not valid!");
		return;
	}
	auto oldPath = GetAbsolutePath();
	auto newPath = oldPath;
	newPath.replace_extension(newExtension);
	if (std::filesystem::exists(newPath))
	{
		EVOENGINE_ERROR("File with new name already exists!");
		return;
	}
	DeleteMetadata();
	m_assetExtension = newExtension;
	if (std::filesystem::exists(oldPath))
	{
		std::filesystem::rename(oldPath, newPath);
	}
	Save();
}
void AssetRecord::Save() const
{
	auto path = GetAbsolutePath().string() + ".evefilemeta";
	YAML::Emitter out;
	out << YAML::BeginMap;
	out << YAML::Key << "m_assetExtension" << YAML::Value << m_assetExtension;
	out << YAML::Key << "m_assetFileName" << YAML::Value << m_assetFileName;
	out << YAML::Key << "m_assetTypeName" << YAML::Value << m_assetTypeName;
	out << YAML::Key << "m_assetHandle" << YAML::Value << m_assetHandle;
	out << YAML::EndMap;
	std::ofstream fout(path);
	fout << out.c_str();
	fout.close();
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
	DWORD attributes = GetFileAttributes(path.c_str());
	SetFileAttributes(path.c_str(), attributes | FILE_ATTRIBUTE_HIDDEN);
#endif
}

Handle AssetRecord::GetAssetHandle() const
{
	return m_assetHandle;
}
void AssetRecord::DeleteMetadata() const
{
	auto path = GetAbsolutePath().string() + ".evefilemeta";
	std::filesystem::remove(path);
}
void AssetRecord::Load(const std::filesystem::path& path)
{
	if (!std::filesystem::exists(path))
	{
		EVOENGINE_ERROR("Metadata not exist!");
		return;
	}
	std::ifstream stream(path.string());
	std::stringstream stringStream;
	stringStream << stream.rdbuf();
	YAML::Node in = YAML::Load(stringStream.str());
	if (in["m_assetFileName"])
		m_assetFileName = in["m_assetFileName"].as<std::string>();
	if (in["m_assetExtension"])
		m_assetExtension = in["m_assetExtension"].as<std::string>();
	if (in["m_assetTypeName"])
		m_assetTypeName = in["m_assetTypeName"].as<std::string>();
	if (in["m_assetHandle"])
		m_assetHandle = in["m_assetHandle"].as<uint64_t>();

	if (!Serialization::HasSerializableType(m_assetTypeName))
	{
		m_assetTypeName = "Binary";
	}
}
std::weak_ptr<Folder> AssetRecord::GetFolder() const
{
	return m_folder;
}
std::filesystem::path Folder::GetProjectRelativePath() const
{
	if (m_parent.expired())
	{
		return "";
	}
	return m_parent.lock()->GetProjectRelativePath() / m_name;
}
std::filesystem::path Folder::GetAbsolutePath() const
{
	auto& projectManager = ProjectManager::GetInstance();
	auto projectPath = projectManager.m_projectPath.parent_path();
	return projectPath / GetProjectRelativePath();
}

Handle Folder::GetHandle() const
{
	return m_handle;
}
std::string Folder::GetName() const
{
	return m_name;
}
void Folder::Rename(const std::string& newName)
{
	auto oldPath = GetAbsolutePath();
	auto newPath = oldPath;
	newPath.replace_filename(newName);
	if (std::filesystem::exists(newPath))
	{
		EVOENGINE_ERROR("Folder with new name already exists!");
		return;
	}
	DeleteMetadata();
	m_name = newName;
	if (std::filesystem::exists(oldPath))
	{
		std::filesystem::rename(oldPath, newPath);
	}
	Save();
}
void Folder::Save() const
{
	auto path = GetAbsolutePath().string() + ".evefoldermeta";
	YAML::Emitter out;
	out << YAML::BeginMap;
	out << YAML::Key << "m_handle" << YAML::Value << m_handle;
	out << YAML::Key << "m_name" << YAML::Value << m_name;
	out << YAML::EndMap;
	std::ofstream fout(path);
	fout << out.c_str();
	fout.close();
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
	DWORD attributes = GetFileAttributes(path.c_str());
	SetFileAttributes(path.c_str(), attributes | FILE_ATTRIBUTE_HIDDEN);
#endif
}
void Folder::Load(const std::filesystem::path& path)
{
	if (!std::filesystem::exists(path))
	{
		EVOENGINE_ERROR("Folder metadata not exist!");
		return;
	}
	std::ifstream stream(path.string());
	std::stringstream stringStream;
	stringStream << stream.rdbuf();
	YAML::Node in = YAML::Load(stringStream.str());
	if (in["m_handle"])
		m_handle = in["m_handle"].as<uint64_t>();
	if (in["m_name"])
		m_name = in["m_name"].as<std::string>();
}
void Folder::DeleteMetadata() const
{
	auto path = GetAbsolutePath().replace_extension(".evefoldermeta");
	std::filesystem::remove(path);
}
void Folder::MoveChild(const Handle& childHandle, const std::shared_ptr<Folder>& dest)
{
	auto search = m_children.find(childHandle);
	if (search == m_children.end())
	{
		EVOENGINE_ERROR("Child not exist!");
		return;
	}
	auto child = search->second;
	auto newPath = dest->GetAbsolutePath() / child->GetName();
	if (std::filesystem::exists(newPath))
	{
		EVOENGINE_ERROR("Destination folder already exists!");
		return;
	}
	auto oldPath = child->GetAbsolutePath();
	child->DeleteMetadata();
	m_children.erase(childHandle);
	if (std::filesystem::exists(oldPath))
	{
		std::filesystem::rename(oldPath, newPath);
	}
	dest->m_children.insert({ childHandle, child });
	child->m_parent = dest;
	child->Save();
}
std::weak_ptr<Folder> Folder::GetChild(const Handle& childHandle)
{
	auto search = m_children.find(childHandle);
	if (search == m_children.end())
	{
		return {};
	}
	return search->second;
}
std::weak_ptr<Folder> Folder::GetOrCreateChild(const std::string& folderName)
{
	for (const auto& i : m_children)
	{
		if (i.second->m_name == folderName)
			return i.second;
	}
	auto newFolder = std::make_shared<Folder>();
	newFolder->m_name = folderName;
	newFolder->m_handle = Handle();
	newFolder->m_self = newFolder;
	m_children[newFolder->m_handle] = newFolder;
	ProjectManager::GetInstance().m_folderRegistry[newFolder->m_handle] = newFolder;
	newFolder->m_parent = m_self;
	auto newFolderPath = newFolder->GetAbsolutePath();
	if (!std::filesystem::exists(newFolderPath))
	{
		std::filesystem::create_directories(newFolderPath);
	}
	newFolder->Save();
	return newFolder;
}
void Folder::DeleteChild(const Handle& childHandle)
{
	auto child = GetChild(childHandle).lock();
	auto childFolderPath = child->GetAbsolutePath();
	std::filesystem::remove_all(childFolderPath);
	child->DeleteMetadata();
	m_children.erase(childHandle);
}
std::shared_ptr<IAsset> Folder::GetOrCreateAsset(const std::string& fileName, const std::string& extension)
{
	auto& projectManager = ProjectManager::GetInstance();
	auto typeName = projectManager.GetTypeName(extension);
	if (typeName.empty())
	{
		EVOENGINE_ERROR("Asset type not exist!");
		return {};
	}
	for (const auto& i : m_assetRecords)
	{
		if (i.second->m_assetFileName == fileName && i.second->m_assetExtension == extension)
			return i.second->GetAsset();
	}
	auto record = std::make_shared<AssetRecord>();
	record->m_folder = m_self;
	record->m_assetTypeName = typeName;
	record->m_assetExtension = extension;
	record->m_assetFileName = fileName;
	record->m_assetHandle = Handle();
	record->m_self = record;
	m_assetRecords[record->m_assetHandle] = record;
	auto asset = record->GetAsset();
	record->Save();
	return asset;
}
std::shared_ptr<IAsset> Folder::GetAsset(const Handle& assetHandle)
{
	auto search = m_assetRecords.find(assetHandle);
	if (search != m_assetRecords.end())
	{
		return search->second->GetAsset();
	}
	return {};
}
void Folder::MoveAsset(const Handle& assetHandle, const std::shared_ptr<Folder>& dest)
{
	auto search = m_assetRecords.find(assetHandle);
	if (search == m_assetRecords.end())
	{
		EVOENGINE_ERROR("AssetRecord not exist!");
		return;
	}
	auto assetRecord = search->second;
	auto newPath = dest->GetAbsolutePath() / (assetRecord->m_assetFileName + assetRecord->m_assetExtension);
	if (std::filesystem::exists(newPath))
	{
		EVOENGINE_ERROR("Destination file already exists!");
		return;
	}
	auto oldPath = assetRecord->GetAbsolutePath();
	assetRecord->DeleteMetadata();
	m_assetRecords.erase(assetHandle);
	if (std::filesystem::exists(oldPath))
	{
		std::filesystem::rename(oldPath, newPath);
	}
	dest->m_assetRecords.insert({ assetHandle, assetRecord });
	assetRecord->m_folder = dest;
	assetRecord->Save();
}
void Folder::DeleteAsset(const Handle& assetHandle)
{
	auto& projectManager = ProjectManager::GetInstance();
	auto assetRecord = m_assetRecords[assetHandle];
	projectManager.m_assetRecordRegistry.erase(assetRecord->m_assetHandle);
	projectManager.m_loadedAssets.erase(assetRecord->m_assetHandle);
	auto assetPath = assetRecord->GetAbsolutePath();
	std::filesystem::remove(assetPath);
	assetRecord->DeleteMetadata();
	m_assetRecords.erase(assetHandle);
}
void Folder::Refresh(const std::filesystem::path& parentAbsolutePath)
{
	auto& projectManager = ProjectManager::GetInstance();
	auto path = parentAbsolutePath / m_name;
	/**
	 * 1. Scan folder for any unregistered folders and assets.
	 */
	std::vector<std::filesystem::path> childFolderMetadataList;
	std::vector<std::filesystem::path> childFolderList;
	std::vector<std::filesystem::path> assetMetadataList;
	std::vector<std::filesystem::path> fileList;
	for (const auto& entry : std::filesystem::directory_iterator(path))
	{
		if (std::filesystem::is_directory(entry.path()))
		{
			childFolderList.push_back(entry.path());
		}
		else if (entry.path().extension() == ".evefoldermeta")
		{
			childFolderMetadataList.push_back(entry.path());
		}
		else if (entry.path().extension() == ".evefilemeta")
		{
			assetMetadataList.push_back(entry.path());
		}
		else if (entry.path().filename() != "" && entry.path().filename() != "." && entry.path().filename() != ".." && entry.path().extension() != ".eveproj")
		{
			fileList.push_back(entry.path());
		}
	}
	for (const auto& childFolderMetadataPath : childFolderMetadataList)
	{
		auto childFolderPath = childFolderMetadataPath;
		childFolderPath.replace_extension("");
		if (!std::filesystem::exists(childFolderPath))
		{
			std::filesystem::remove(childFolderMetadataPath);
		}
		else
		{
			auto folderName = childFolderMetadataPath.filename();
			folderName.replace_extension("");
			std::shared_ptr<Folder> child;
			for (const auto& i : m_children)
			{
				if (i.second->m_name == folderName)
				{
					child = i.second;
				}
			}
			if (!child)
			{
				auto newFolder = std::make_shared<Folder>();
				newFolder->m_self = newFolder;
				newFolder->m_name = folderName.string();
				newFolder->m_parent = m_self;
				newFolder->Load(childFolderMetadataPath);
				m_children[newFolder->m_handle] = newFolder;

				projectManager.m_folderRegistry[newFolder->m_handle] = newFolder;
			}
		}
	}
	for (const auto& childFolderPath : childFolderList)
	{
		auto childFolder = GetOrCreateChild(childFolderPath.filename().string()).lock();
		childFolder->Refresh(path);
	}
	for (const auto& assetMetadataPath : assetMetadataList)
	{
		auto assetName = assetMetadataPath.filename();
		assetName.replace_extension("").replace_extension("");
		auto assetExtension = assetMetadataPath.filename().replace_extension("").extension();
		bool exist = false;
		for (const auto& i : m_assetRecords)
		{
			if (i.second->m_assetFileName == assetName && i.second->m_assetExtension == assetExtension)
			{
				exist = true;
			}
		}

		if (!exist)
		{
			auto newAssetRecord = std::make_shared<AssetRecord>();
			newAssetRecord->m_folder = m_self.lock();
			newAssetRecord->m_self = newAssetRecord;
			newAssetRecord->Load(assetMetadataPath);
			if (!std::filesystem::exists(newAssetRecord->GetAbsolutePath()))
			{
				std::filesystem::remove(assetMetadataPath);
			}
			else
			{
				m_assetRecords[newAssetRecord->m_assetHandle] = newAssetRecord;
				projectManager.m_assetRecordRegistry[newAssetRecord->m_assetHandle] = newAssetRecord;
			}
		}
	}
	for (const auto& filePath : fileList)
	{
		auto filename = filePath.filename().replace_extension("").replace_extension("").string();
		auto extension = filePath.extension().string();
		auto typeName = ProjectManager::GetTypeName(extension);
		if (!HasAsset(filename, extension))
		{
			std::shared_ptr<AssetRecord> newAssetRecord = std::make_shared<AssetRecord>();
			newAssetRecord->m_folder = m_self.lock();
			newAssetRecord->m_assetTypeName = typeName;
			newAssetRecord->m_assetExtension = extension;
			newAssetRecord->m_assetFileName = filename;
			newAssetRecord->m_assetHandle = Handle();
			newAssetRecord->m_self = newAssetRecord;
			m_assetRecords[newAssetRecord->m_assetHandle] = newAssetRecord;
			projectManager.m_assetRecordRegistry[newAssetRecord->m_assetHandle] = newAssetRecord;
			newAssetRecord->Save();
		}
	}
	/**
	 * 2. Clear deleted asset and folder.
	 */
	std::vector<Handle> assetToRemove;
	for (const auto& i : m_assetRecords)
	{
		auto absolutePath = i.second->GetAbsolutePath();
		if (!std::filesystem::exists(absolutePath))
		{
			assetToRemove.push_back(i.first);
		}
	}
	for (const auto& i : assetToRemove)
	{
		DeleteAsset(i);
	}
	std::vector<Handle> folderToRemove;
	for (const auto& i : m_children)
	{
		if (!std::filesystem::exists(i.second->GetAbsolutePath()))
		{
			folderToRemove.push_back(i.first);
		}
	}
	for (const auto& i : folderToRemove)
	{
		DeleteChild(i);
	}
}
void Folder::RegisterAsset(
	const std::shared_ptr<IAsset>& asset, const std::string& fileName, const std::string& extension)
{
	auto& projectManager = ProjectManager::GetInstance();
	auto record = std::make_shared<AssetRecord>();
	record->m_folder = m_self;
	record->m_assetTypeName = asset->GetTypeName();
	record->m_assetExtension = extension;
	record->m_assetFileName = fileName;
	record->m_assetHandle = asset->m_handle;
	record->m_self = record;
	record->m_asset = asset;
	m_assetRecords[record->m_assetHandle] = record;
	projectManager.m_assetRegistry[record->m_assetHandle] = asset;
	projectManager.m_loadedAssets[record->m_assetHandle] = asset;
	projectManager.m_assetRecordRegistry[record->m_assetHandle] = record;
	asset->m_assetRecord = record;
	asset->m_saved = false;
	record->Save();
}
bool Folder::HasAsset(const std::string& fileName, const std::string& extension) const
{
	auto& projectManager = ProjectManager::GetInstance();
	auto typeName = projectManager.GetTypeName(extension);
	if (typeName.empty())
	{
		EVOENGINE_ERROR("Asset type not exist!");
		return false;
	}
	for (const auto& i : m_assetRecords)
	{
		if (i.second->m_assetFileName == fileName && i.second->m_assetExtension == extension)
			return true;
	}
	return false;
}
Folder::~Folder()
{
	auto& projectManager = ProjectManager::GetInstance();
	projectManager.m_folderRegistry.erase(m_handle);
}
bool Folder::IsSelfOrAncestor(const Handle& handle)
{
	std::shared_ptr<Folder> walker = m_self.lock();
	while (true)
	{
		if (walker->GetHandle() == handle)
			return true;
		if (walker->m_parent.expired())
			return false;
		walker = walker->m_parent.lock();
	}
	return false;
}

std::weak_ptr<Folder> ProjectManager::GetOrCreateFolder(const std::filesystem::path& projectRelativePath)
{
	auto& projectManager = GetInstance();
	if (!projectRelativePath.is_relative())
	{
		EVOENGINE_ERROR("Path not relative!");
		return {};
	}
	auto dirPath = projectManager.m_projectFolder->GetAbsolutePath().parent_path() / projectRelativePath;
	std::shared_ptr<Folder> retVal = projectManager.m_projectFolder;
	for (auto it = projectRelativePath.begin(); it != projectRelativePath.end(); ++it)
	{
		retVal = retVal->GetOrCreateChild(it->filename().string()).lock();
	}
	return retVal;
}
std::shared_ptr<IAsset> ProjectManager::GetOrCreateAsset(const std::filesystem::path& projectRelativePath)
{
	if (std::filesystem::is_directory(projectRelativePath))
	{
		EVOENGINE_ERROR("Path is directory!");
		return {};
	}
	auto folder = GetOrCreateFolder(projectRelativePath.parent_path()).lock();
	auto stem = projectRelativePath.stem().string();
	auto fileName = projectRelativePath.filename().string();
	auto extension = projectRelativePath.extension().string();
	if (fileName == stem)
	{
		stem = "";
		extension = fileName;
	}
	return folder->GetOrCreateAsset(stem, extension);
}

void ProjectManager::GetOrCreateProject(const std::filesystem::path& path)
{
	auto& projectManager = GetInstance();
	auto projectAbsolutePath = std::filesystem::absolute(path);
	if (std::filesystem::is_directory(projectAbsolutePath))
	{
		EVOENGINE_ERROR("Path is directory!");
		return;
	}
	if (!projectAbsolutePath.is_absolute())
	{
		EVOENGINE_ERROR("Path not absolute!");
		return;
	}
	if (projectAbsolutePath.extension() != ".eveproj")
	{
		EVOENGINE_ERROR("Wrong extension!");
		return;
	}
	projectManager.m_projectPath = projectAbsolutePath;
	projectManager.m_assetRegistry.clear();
	projectManager.m_loadedAssets.clear();
	projectManager.m_assetRecordRegistry.clear();
	projectManager.m_folderRegistry.clear();
	Application::Reset();

	std::shared_ptr<Scene> scene;

	projectManager.m_currentFocusedFolder = projectManager.m_projectFolder = std::make_shared<Folder>();
	projectManager.m_folderRegistry[0] = projectManager.m_projectFolder;
	projectManager.m_projectFolder->m_self = projectManager.m_projectFolder;
	if (!std::filesystem::exists(projectManager.m_projectFolder->GetAbsolutePath()))
	{
		std::filesystem::create_directories(projectManager.m_projectFolder->GetAbsolutePath());
	}
	ScanProject();

	bool foundScene = false;
	if (std::filesystem::exists(projectAbsolutePath))
	{
		std::ifstream stream(projectAbsolutePath.string());
		std::stringstream stringStream;
		stringStream << stream.rdbuf();
		YAML::Node in = YAML::Load(stringStream.str());
		auto temp = GetAsset(in["m_startSceneHandle"].as<uint64_t>());
		if (temp)
		{
			scene = std::dynamic_pointer_cast<Scene>(temp);
			SetStartScene(scene);
			Application::Attach(scene);
			foundScene = true;
		}
		EVOENGINE_LOG("Found and loaded project");
		if (projectManager.m_scenePostLoadFunction.has_value()) {
			projectManager.m_scenePostLoadFunction.value()(scene);
			TransformGraph::CalculateTransformGraphs(scene);
		}
	}
	if (!foundScene)
	{
		scene = CreateTemporaryAsset<Scene>();
		std::filesystem::path newSceneRelativePath = GenerateNewProjectRelativePath("New Scene", ".evescene");
		bool succeed = scene->SetPathAndSave(newSceneRelativePath);
		if (succeed)
		{
			EVOENGINE_LOG("Created new start scene!");
		}
		SetStartScene(scene);
		Application::Attach(scene);

		if (projectManager.m_newSceneCustomizer.has_value()) {
			projectManager.m_newSceneCustomizer.value()(scene);
			TransformGraph::CalculateTransformGraphs(scene);
		}
	}
}
void ProjectManager::SaveProject()
{
	auto& projectManager = GetInstance();
	auto directory = projectManager.m_projectPath.parent_path();
	if (!std::filesystem::exists(directory))
	{
		std::filesystem::create_directories(directory);
	}
	YAML::Emitter out;
	out << YAML::BeginMap;
	out << YAML::Key << "m_startSceneHandle" << YAML::Value << projectManager.m_startScene->GetHandle();
	out << YAML::EndMap;
	std::ofstream fout(projectManager.m_projectPath.string());
	fout << out.c_str();
	fout.flush();
}
std::filesystem::path ProjectManager::GetProjectPath()
{
	auto& projectManager = GetInstance();
	return projectManager.m_projectPath;
}
std::string ProjectManager::GetProjectName()
{
	auto& projectManager = GetInstance();
	return projectManager.m_projectPath.stem().string();
}
std::weak_ptr<Folder> ProjectManager::GetCurrentFocusedFolder()
{
	auto& projectManager = GetInstance();
	return projectManager.m_currentFocusedFolder;
}

bool ProjectManager::IsAsset(const std::string& typeName)
{
	auto& projectManager = GetInstance();
	return projectManager.m_assetExtensions.find(typeName) != projectManager.m_assetExtensions.end();
}

std::shared_ptr<IAsset> ProjectManager::GetAsset(const Handle& handle)
{
	auto& projectManager = GetInstance();
	auto search = projectManager.m_assetRegistry.find(handle);
	if (search != projectManager.m_assetRegistry.end())
		return search->second.lock();
	auto search2 = projectManager.m_assetRecordRegistry.find(handle);
	if (search2 != projectManager.m_assetRecordRegistry.end())
		return search2->second.lock()->GetAsset();

	if (Resources::IsResource(handle))
	{
		return Resources::GetResource<IAsset>(handle);
	}
	return {};
}

std::vector<std::string> ProjectManager::GetExtension(const std::string& typeName)
{
	auto& projectManager = GetInstance();
	auto search = projectManager.m_assetExtensions.find(typeName);
	if (search != projectManager.m_assetExtensions.end())
		return search->second;
	return {};
}
std::string ProjectManager::GetTypeName(const std::string& extension)
{
	auto& projectManager = GetInstance();
	auto search = projectManager.m_typeNames.find(extension);
	if (search != projectManager.m_typeNames.end())
		return search->second;
	return "Binary";
}

std::shared_ptr<IAsset> ProjectManager::CreateTemporaryAsset(const std::string& typeName)
{
	size_t hashCode;
	auto retVal = std::dynamic_pointer_cast<IAsset>(Serialization::ProduceSerializable(typeName, hashCode, Handle()));
	auto& projectManager = GetInstance();
	projectManager.m_assetRegistry[retVal->GetHandle()] = retVal;
	retVal->m_self = retVal;
	retVal->OnCreate();
	return retVal;
}
std::shared_ptr<IAsset> ProjectManager::CreateTemporaryAsset(const std::string& typeName, const Handle& handle)
{
	size_t hashCode;
	auto retVal = std::dynamic_pointer_cast<IAsset>(Serialization::ProduceSerializable(typeName, hashCode, handle));
	if (!retVal) {
		return nullptr;
	}
	auto& projectManager = GetInstance();
	projectManager.m_assetRegistry[retVal->GetHandle()] = retVal;
	retVal->m_self = retVal;
	retVal->OnCreate();
	return retVal;
}
bool ProjectManager::IsInProjectFolder(const std::filesystem::path& absolutePath)
{
	if (!absolutePath.is_absolute())
	{
		EVOENGINE_ERROR("Not absolute path!");
		return false;
	}
	auto& projectManager = GetInstance();
	auto projectFolderPath = projectManager.m_projectPath.parent_path();
	auto it = std::search(absolutePath.begin(), absolutePath.end(), projectFolderPath.begin(), projectFolderPath.end());
	return it != absolutePath.end();
}
bool ProjectManager::IsValidAssetFileName(const std::filesystem::path& path)
{
	auto stem = path.stem().string();
	auto fileName = path.filename().string();
	auto extension = path.extension().string();
	if (fileName == stem)
	{
		stem = "";
		extension = fileName;
	}
	auto typeName = GetTypeName(extension);
	return typeName == "Binary";
}
std::filesystem::path ProjectManager::GenerateNewProjectRelativePath(const std::string& relativeStem, const std::string& postfix)
{
	assert(std::filesystem::path(relativeStem + postfix).is_relative());
	const auto& projectManager = GetInstance();
	const auto projectPath = projectManager.m_projectPath.parent_path();
	std::filesystem::path testPath = projectPath / (relativeStem + postfix);
	int i = 0;
	while (std::filesystem::exists(testPath))
	{
		i++;
		testPath = projectPath / (relativeStem + " (" + std::to_string(i) + ")" + postfix);
	}
	if (i == 0)
		return relativeStem + postfix;
	return relativeStem + " (" + std::to_string(i) + ")" + postfix;
}

std::filesystem::path ProjectManager::GenerateNewAbsolutePath(const std::string& absoluteStem,
	const std::string& postfix)
{
	std::filesystem::path testPath = absoluteStem + postfix;
	int i = 0;
	while (std::filesystem::exists(testPath))
	{
		i++;
		testPath = absoluteStem + " (" + std::to_string(i) + ")" + postfix;
	}
	if (i == 0)
		return absoluteStem + postfix;
	return absoluteStem + " (" + std::to_string(i) + ")" + postfix;
}

void ProjectManager::SetActionAfterSceneLoad(const std::function<void(const std::shared_ptr<Scene>&)>& actions)
{
	auto& projectManager = GetInstance();
	projectManager.m_scenePostLoadFunction = actions;
}

void ProjectManager::SetActionAfterNewScene(const std::function<void(const std::shared_ptr<Scene>&)>& actions)
{
	auto& projectManager = GetInstance();
	projectManager.m_newSceneCustomizer = actions;
}

void ProjectManager::ScanProject()
{
	const auto& projectManager = GetInstance();
	if (!projectManager.m_projectFolder)
		return;
	const auto directory = projectManager.m_projectPath.parent_path().parent_path();
	projectManager.m_projectFolder->m_handle = 0;
	projectManager.m_projectFolder->m_name = projectManager.m_projectPath.parent_path().stem().string();
	projectManager.m_projectFolder->Refresh(directory);
}

void ProjectManager::OnDestroy()
{
	auto& projectManager = GetInstance();

	projectManager.m_projectFolder.reset();
	projectManager.m_newSceneCustomizer.reset();

	projectManager.m_currentFocusedFolder.reset();
	projectManager.m_loadedAssets.clear();
	projectManager.m_assetRegistry.clear();
	projectManager.m_assetRecordRegistry.clear();
	projectManager.m_folderRegistry.clear();
	projectManager.m_startScene.reset();

	projectManager.m_assetThumbnails.clear();
	projectManager.m_assetThumbnailStorage.clear();

	projectManager.m_inspectingAsset.reset();
}

void ProjectManager::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	auto& projectManager = GetInstance();
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("Project"))
		{
			ImGui::Text(("Current Project path: " + projectManager.m_projectPath.string()).c_str());

			FileUtils::SaveFile(
				"Create or load New Project##ProjectManager",
				"Project",
				{ ".eveproj" },
				[](const std::filesystem::path& filePath) {
					try
					{
						GetOrCreateProject(filePath);
					}
					catch (std::exception& e)
					{
						EVOENGINE_ERROR("Failed to create/load from " + filePath.string());
					}
				},
				false);

			if (ImGui::Button("Save"))
			{
				SaveProject();
			}
			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("View"))
		{
			ImGui::Checkbox("Project", &projectManager.m_showProjectWindow);
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}
	if (projectManager.m_showProjectWindow) {
		if (ImGui::Begin("Project"))
		{
			if (projectManager.m_projectFolder)
			{
				auto currentFocusedFolder = projectManager.m_currentFocusedFolder.lock();
				auto currentFolderPath = currentFocusedFolder->GetProjectRelativePath();
				if (ImGui::BeginDragDropTarget())
				{
					if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Asset"))
					{
						IM_ASSERT(payload->DataSize == sizeof(Handle));
						Handle handle = *(Handle*)payload->Data;

						auto assetSearch = projectManager.m_assetRegistry.find(handle);
						if (assetSearch != projectManager.m_assetRegistry.end() &&
							!assetSearch->second.expired())
						{
							auto asset = assetSearch->second.lock();
							if (asset->IsTemporary())
							{
								auto fileExtension =
									projectManager.m_assetExtensions[asset->GetTypeName()].front();
								auto fileName = "New " + asset->GetTypeName();
								auto filePath = GenerateNewProjectRelativePath(
									(currentFocusedFolder->GetProjectRelativePath() / fileName).string(), fileExtension);
								asset->SetPathAndSave(filePath);
							}
							else
							{
								auto assetRecord = asset->m_assetRecord.lock();
								if (assetRecord->GetFolder().lock().get() != currentFocusedFolder.get())
								{
									auto fileExtension = assetRecord->GetAssetExtension();
									auto fileName = assetRecord->GetAssetFileName();
									auto filePath = GenerateNewProjectRelativePath(
										(currentFocusedFolder->GetProjectRelativePath() / fileName).string(),
										fileExtension);
									asset->SetPathAndSave(filePath);
								}
							}
						}
						else
						{
							auto assetRecordSearch = projectManager.m_assetRecordRegistry.find(handle);
							if (assetRecordSearch != projectManager.m_assetRecordRegistry.end() &&
								!assetRecordSearch->second.expired())
							{
								auto assetRecord = assetRecordSearch->second.lock();
								auto folder = assetRecord->GetFolder().lock();
								if (folder.get() != currentFocusedFolder.get())
								{
									folder->MoveAsset(assetRecord->GetAssetHandle(), currentFocusedFolder);
								}
							}
						}
					}

					if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Entity"))
					{
						IM_ASSERT(payload->DataSize == sizeof(Handle));
						auto prefab = std::dynamic_pointer_cast<Prefab>(CreateTemporaryAsset<Prefab>());
						auto entityHandle = *static_cast<Handle*>(payload->Data);
						auto scene = Application::GetActiveScene();
						auto entity = scene->GetEntity(entityHandle);
						if (scene->IsEntityValid(entity))
						{
							prefab->FromEntity(entity);
							// If current folder doesn't contain file with same name
							auto fileName = scene->GetEntityName(entity);
							auto fileExtension = projectManager.m_assetExtensions["Prefab"].at(0);
							auto filePath =
								GenerateNewProjectRelativePath((currentFolderPath / fileName).string(), fileExtension);
							prefab->SetPathAndSave(filePath);
						}
					}

					ImGui::EndDragDropTarget();

				}
				static glm::vec2 thumbnailSizePadding = { 96.0f, 8.0f };
				float cellSize = thumbnailSizePadding.x + thumbnailSizePadding.y;
				static float size1 = 200;
				static float size2 = 200;
				static float h = 100;
				auto avail = ImGui::GetContentRegionAvail();
				size2 = glm::max(avail.x - size1, cellSize + 8.0f);
				size1 = glm::max(avail.x - size2, 32.0f);
				h = avail.y;
				ImGui::Splitter(true, 8.0, size1, size2, 32.0f, cellSize + 8.0f, h);
				ImGui::BeginChild("1", ImVec2(size1, h), true);
				FolderHierarchyHelper(projectManager.m_projectFolder);
				ImGui::EndChild();

				ImGui::SameLine();

				ImGui::BeginChild("2", ImVec2(size2 - 5.0f, h), true);
				if (ImGui::ImageButton(editorLayer->AssetIcons()["RefreshButton"]->GetImTextureId(),
					{ 16, 16 },
					{ 0, 1 },
					{ 1, 0 }))
				{
					projectManager.ScanProject();
				}

				if (currentFocusedFolder != projectManager.m_projectFolder)
				{
					ImGui::SameLine();
					if (ImGui::ImageButton(editorLayer->AssetIcons()["BackButton"]->GetImTextureId(),
						{ 16, 16 },
						{ 0, 1 },
						{ 1, 0 }))
					{
						projectManager.m_currentFocusedFolder = currentFocusedFolder->m_parent;
					}
				}
				ImGui::SameLine();
				ImGui::Text(currentFocusedFolder->GetProjectRelativePath().string().c_str());
				ImGui::Separator();

				bool updated = false;
				if (ImGui::BeginPopupContextWindow("NewAssetPopup"))
				{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
					if (ImGui::Button("Show in Explorer..."))
					{
						const auto folderPath = currentFocusedFolder->GetAbsolutePath().string();
						ShellExecuteA(NULL, "open", folderPath.c_str(), NULL, NULL, SW_SHOWDEFAULT);
					}
#else
#endif

					FileUtils::OpenFile("Import model...", "Model", { ".eveprefab", ".obj", ".gltf", ".glb", ".blend", ".ply", ".fbx", ".dae", ".x3d" }, [&](const std::filesystem::path& path)
					{
						const auto prefab = CreateTemporaryAsset<Prefab>();
						if(prefab->Import(path))
						{
							prefab->SetPathAndSave(currentFocusedFolder->GetProjectRelativePath() / path.filename().replace_extension(".eveprefab"));
						}
					}, false);

					if (ImGui::Button("New folder..."))
					{
						auto newPath = GenerateNewProjectRelativePath(
							(currentFocusedFolder->GetProjectRelativePath() / "New Folder").string(), "");
						GetOrCreateFolder(newPath);
					}
					if (ImGui::BeginMenu("New asset..."))
					{
						for (auto& i : projectManager.m_assetExtensions)
						{
							if (ImGui::Button(i.first.c_str()))
							{
								std::string newFileName = "New " + i.first;
								std::filesystem::path newPath = GenerateNewProjectRelativePath(
									(currentFocusedFolder->GetProjectRelativePath() / newFileName).string(),
									i.second.front());
								currentFocusedFolder->GetOrCreateAsset(
									newPath.stem().string(), newPath.extension().string());
							}
						}
						ImGui::EndMenu();
					}
					ImGui::EndPopup();
				}

				float panelWidth = ImGui::GetWindowContentRegionMax().x;
				int columnCount = glm::max(1, static_cast<int>(panelWidth / cellSize));
				ImGui::Columns(columnCount, nullptr, false);
				if (!updated)
				{
					for (auto& i : currentFocusedFolder->m_children)
					{
						ImGui::Image(editorLayer->AssetIcons()["Folder"]->GetImTextureId(),
							{ thumbnailSizePadding.x, thumbnailSizePadding.x },
							{ 0, 1 },
							{ 1, 0 });
						const std::string tag = "##Folder" + std::to_string(i.second->m_handle);
						if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
						{
							ImGui::SetDragDropPayload("Folder", &i.second->m_handle, sizeof(Handle));
							ImGui::TextColored(ImVec4(0, 0, 1, 1), ("Folder" + tag).c_str());
							ImGui::EndDragDropSource();
						}
						if (i.second->GetHandle() != 0)
						{
							if (ImGui::BeginPopupContextItem(tag.c_str()))
							{
								if (ImGui::BeginMenu(("Rename" + tag).c_str()))
								{
									static char newName[256] = { 0 };
									ImGui::InputText(("New name" + tag).c_str(), newName, 256);
									if (ImGui::Button(("Confirm" + tag).c_str()))
									{
										i.second->Rename(std::string(newName));
										memset(newName, 0, 256);
										ImGui::CloseCurrentPopup();
									}
									ImGui::EndMenu();
								}
								if (ImGui::Button(("Remove" + tag).c_str()))
								{
									i.second->m_parent.lock()->DeleteChild(i.second->m_handle);
									updated = true;
									ImGui::CloseCurrentPopup();
									ImGui::EndPopup();
									break;
								}
								ImGui::EndPopup();
							}
						}
						if (ImGui::BeginDragDropTarget())
						{
							if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Folder"))
							{
								IM_ASSERT(payload->DataSize == sizeof(Handle));
								Handle payload_n = *(Handle*)payload->Data;
								if (payload_n.GetValue() != 0)
								{
									auto temp = projectManager.m_folderRegistry[payload_n];
									if (!temp.expired())
									{
										auto actualFolder = temp.lock();
										if (!i.second->IsSelfOrAncestor(actualFolder->m_handle) &&
											actualFolder->m_parent.lock().get() != i.second.get())
										{
											actualFolder->m_parent.lock()->MoveChild(actualFolder->GetHandle(), i.second);
										}
									}
								}
							}

							for (const auto& extension : projectManager.m_assetExtensions)
							{
								if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Asset"))
								{
									IM_ASSERT(payload->DataSize == sizeof(Handle));
									Handle payload_n = *(Handle*)payload->Data;
									auto assetSearch = projectManager.m_assetRegistry.find(payload_n);
									if (assetSearch != projectManager.m_assetRegistry.end() &&
										!assetSearch->second.expired())
									{
										auto asset = assetSearch->second.lock();
										if (asset->IsTemporary())
										{
											auto fileExtension =
												projectManager.m_assetExtensions[asset->GetTypeName()].front();
											auto fileName = "New " + asset->GetTypeName();
											int index = 0;
											auto filePath = GenerateNewProjectRelativePath(
												(i.second->GetProjectRelativePath() / fileName).string(), fileExtension);
											asset->SetPathAndSave(filePath);
										}
										else
										{
											auto assetRecord = asset->m_assetRecord.lock();
											if (assetRecord->GetFolder().lock().get() != i.second.get())
											{
												auto fileExtension = assetRecord->GetAssetExtension();
												auto fileName = assetRecord->GetAssetFileName();
												auto filePath = GenerateNewProjectRelativePath(
													(i.second->GetProjectRelativePath() / fileName).string(),
													fileExtension);
												asset->SetPathAndSave(filePath);
											}
										}
									}
									else
									{
										auto assetRecordSearch = projectManager.m_assetRecordRegistry.find(payload_n);
										if (assetRecordSearch != projectManager.m_assetRecordRegistry.end() &&
											!assetRecordSearch->second.expired())
										{
											auto assetRecord = assetRecordSearch->second.lock();
											auto folder = assetRecord->GetFolder().lock();
											if (folder.get() != i.second.get())
											{
												folder->MoveAsset(assetRecord->GetAssetHandle(), i.second);
											}
										}
									}
								}
							}
							if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Binary"))
							{
								IM_ASSERT(payload->DataSize == sizeof(Handle));
								Handle payload_n = *(Handle*)payload->Data;
								auto record = projectManager.m_assetRecordRegistry[payload_n];
								if (!record.expired())
									record.lock()->GetFolder().lock()->MoveAsset(payload_n, i.second);
							}

							if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Entity"))
							{
								IM_ASSERT(payload->DataSize == sizeof(Entity));
								auto prefab =
									std::dynamic_pointer_cast<Prefab>(CreateTemporaryAsset<Prefab>());
								auto entity = *static_cast<Entity*>(payload->Data);
								prefab->FromEntity(entity);
								// If current folder doesn't contain file with same name
								auto scene = Application::GetActiveScene();
								auto fileName = scene->GetEntityName(entity);
								auto fileExtension = projectManager.m_assetExtensions["Prefab"].at(0);
								auto filePath =
									GenerateNewProjectRelativePath((currentFolderPath / fileName).string(), fileExtension);
								prefab->SetPathAndSave(filePath);
							}

							ImGui::EndDragDropTarget();
						}
						bool itemHovered = false;
						if (ImGui::IsItemHovered())
						{
							itemHovered = true;
							if (ImGui::IsMouseDoubleClicked(0))
							{
								projectManager.m_currentFocusedFolder = i.second;
								updated = true;
								break;
							}
						}

						if (itemHovered)
							ImGui::PushStyleColor(ImGuiCol_Text, { 1, 1, 0, 1 });
						ImGui::TextWrapped(i.second->m_name.c_str());
						if (itemHovered)
							ImGui::PopStyleColor(1);
						ImGui::NextColumn();
					}
				}
				if (!updated)
				{
					for (auto& i : currentFocusedFolder->m_assetRecords)
					{
						ImTextureID textureId = 0;
						auto fileName = i.second->GetProjectRelativePath().filename();
						if (fileName.string() == ".eveproj" || fileName.extension().string() == ".eveproj")
							continue;
						if (fileName.extension().string() == ".eveproj")
						{
							textureId = editorLayer->AssetIcons()["Project"]->GetImTextureId();
						}
						else
						{
							auto iconSearch = editorLayer->AssetIcons().find(i.second->GetAssetTypeName());
							if (iconSearch != editorLayer->AssetIcons().end())
							{
								textureId = iconSearch->second->GetImTextureId();
							}
							else
							{
								textureId = editorLayer->AssetIcons()["Binary"]->GetImTextureId();
							}
						}
						static Handle focusedAssetHandle;
						bool itemFocused = false;
						if (focusedAssetHandle == i.first.GetValue())
						{
							itemFocused = true;
						}
						ImGui::Image(textureId, { thumbnailSizePadding.x, thumbnailSizePadding.x }, { 0, 1 }, { 1, 0 });

						bool itemHovered = false;
						if (ImGui::IsItemHovered())
						{
							itemHovered = true;
							if (ImGui::IsMouseDoubleClicked(0) && i.second->GetAssetTypeName() != "Binary")
							{
								// If it's an asset then inspect.
								auto asset = i.second->GetAsset();
								if (asset)
									projectManager.m_inspectingAsset = asset;
							}
						}
						const std::string tag = "##" + i.second->GetAssetTypeName() + std::to_string(i.first.GetValue());
						if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
						{
							ImGui::SetDragDropPayload("Asset", &i.first, sizeof(Handle));
							ImGui::TextColored(ImVec4(0, 0, 1, 1), (i.second->GetAssetFileName()).c_str());
							ImGui::EndDragDropSource();
						}

						if (ImGui::BeginPopupContextItem(tag.c_str()))
						{
							if (ImGui::Button("Duplicate"))
							{
								auto ptr = i.second->GetAsset();
								auto newAsset = DuplicateAsset(ptr);
							}
							if (i.second->GetAssetTypeName() != "Binary" && ImGui::BeginMenu(("Rename" + tag).c_str()))
							{
								static char newName[256] = { 0 };
								ImGui::InputText(("New name" + tag).c_str(), newName, 256);
								if (ImGui::Button(("Confirm" + tag).c_str()))
								{
									auto ptr = i.second->GetAsset();
									bool succeed = ptr->SetPathAndSave(ptr->GetProjectRelativePath().replace_filename(
										std::string(newName) + ptr->GetAssetRecord().lock()->GetAssetExtension()));
									memset(newName, 0, 256);
								}
								ImGui::EndMenu();
							}
							if (ImGui::Button(("Delete" + tag).c_str()))
							{
								currentFocusedFolder->DeleteAsset(i.first);
								ImGui::EndPopup();
								break;
							}
							ImGui::EndPopup();
						}

						if (itemFocused)
							ImGui::PushStyleColor(ImGuiCol_Text, { 1, 0, 0, 1 });
						else if (itemHovered)
							ImGui::PushStyleColor(ImGuiCol_Text, { 1, 1, 0, 1 });
						ImGui::TextWrapped(fileName.string().c_str());
						if (itemFocused || itemHovered)
							ImGui::PopStyleColor(1);
						ImGui::NextColumn();
					}
				}

				ImGui::Columns(1);
				// ImGui::SliderFloat("Thumbnail Size", &thumbnailSizePadding.x, 16, 512);
				ImGui::EndChild();
			}
			else
			{
				ImGui::Text("No project loaded!");
			}
		}
		ImGui::End();
	}
}

void ProjectManager::FolderHierarchyHelper(const std::shared_ptr<Folder>& folder)
{
	auto& projectManager = GetInstance();
	auto focusFolder = projectManager.m_currentFocusedFolder.lock();
	const bool opened = ImGui::TreeNodeEx(
		folder->m_name.c_str(),
		ImGuiTreeNodeFlags_OpenOnArrow |
		(folder == focusFolder ? ImGuiTreeNodeFlags_Selected : ImGuiTreeNodeFlags_None));
	if (ImGui::BeginDragDropTarget())
	{
		if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Folder"))
		{
			IM_ASSERT(payload->DataSize == sizeof(Handle));
			Handle payload_n = *(Handle*)payload->Data;
			if (payload_n.GetValue() != 0)
			{
				auto temp = projectManager.m_folderRegistry[payload_n];
				if (!temp.expired())
				{
					auto actualFolder = temp.lock();
					if (!folder->IsSelfOrAncestor(actualFolder->m_handle) &&
						actualFolder->m_parent.lock().get() != folder.get())
					{
						actualFolder->m_parent.lock()->MoveChild(actualFolder->GetHandle(), folder);
					}
				}
			}
		}
		for (const auto& extension : projectManager.m_assetExtensions)
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("Asset"))
			{
				IM_ASSERT(payload->DataSize == sizeof(Handle));
				Handle payload_n = *(Handle*)payload->Data;
				auto assetSearch = projectManager.m_assetRegistry.find(payload_n);
				if (assetSearch != projectManager.m_assetRegistry.end() && !assetSearch->second.expired())
				{
					auto asset = assetSearch->second.lock();
					if (asset->IsTemporary())
					{
						auto fileExtension = projectManager.m_assetExtensions[asset->GetTypeName()].front();
						auto fileName = "New " + asset->GetTypeName();
						int index = 0;
						auto filePath = ProjectManager::GenerateNewProjectRelativePath(
							(folder->GetProjectRelativePath() / fileName).string(), fileExtension);
						asset->SetPathAndSave(filePath);
					}
					else
					{
						auto assetRecord = asset->m_assetRecord.lock();
						if (assetRecord->GetFolder().lock().get() != folder.get())
						{
							auto fileExtension = assetRecord->GetAssetExtension();
							auto fileName = assetRecord->GetAssetFileName();
							auto filePath = ProjectManager::GenerateNewProjectRelativePath(
								(folder->GetProjectRelativePath() / fileName).string(), fileExtension);
							asset->SetPathAndSave(filePath);
						}
					}
				}
				else
				{
					auto assetRecordSearch = projectManager.m_assetRecordRegistry.find(payload_n);
					if (assetRecordSearch != projectManager.m_assetRecordRegistry.end() &&
						!assetRecordSearch->second.expired())
					{
						auto assetRecord = assetRecordSearch->second.lock();
						auto previousFolder = assetRecord->GetFolder().lock();
						if (folder && previousFolder.get() != folder.get())
						{
							previousFolder->MoveAsset(assetRecord->GetAssetHandle(), folder);
						}
					}
				}
			}
		}
		ImGui::EndDragDropTarget();
	}
	if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
	{
		projectManager.m_currentFocusedFolder = folder;
	}
	const std::string tag = "##Folder" + std::to_string(folder->m_handle);
	if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
	{
		ImGui::SetDragDropPayload("Folder", &folder->m_handle, sizeof(Handle));
		ImGui::TextColored(ImVec4(0, 0, 1, 1), ("Folder" + tag).c_str());
		ImGui::EndDragDropSource();
	}
	if (folder->GetHandle() != 0)
	{
		if (ImGui::BeginPopupContextItem(tag.c_str()))
		{
			if (ImGui::BeginMenu(("Rename" + tag).c_str()))
			{
				static char newName[256] = { 0 };
				ImGui::InputText(("New name" + tag).c_str(), newName, 256);
				if (ImGui::Button(("Confirm" + tag).c_str()))
				{
					folder->Rename(std::string(newName));
					memset(newName, 0, 256);
					ImGui::CloseCurrentPopup();
				}
				ImGui::EndMenu();
			}
			if (ImGui::Button(("Remove" + tag).c_str()))
			{
				folder->m_parent.lock()->DeleteChild(folder->m_handle);
				ImGui::CloseCurrentPopup();
				ImGui::EndPopup();
				return;
			}
			ImGui::EndPopup();
		}
	}
	if (opened)
	{
		for (const auto& i : folder->m_children)
		{
			FolderHierarchyHelper(i.second);
		}
		for (const auto& i : folder->m_assetRecords)
		{
			if (ImGui::TreeNodeEx(
				(i.second->GetAssetFileName() + i.second->GetAssetExtension()).c_str(), ImGuiTreeNodeFlags_Bullet))
			{
				ImGui::TreePop();
			}
			bool itemHovered = false;
			if (ImGui::IsItemHovered())
			{
				itemHovered = true;
				if (ImGui::IsMouseDoubleClicked(0) && i.second->GetAssetTypeName() != "Binary")
				{
					// If it's an asset then inspect.
					if (auto asset = i.second->GetAsset())
						projectManager.m_inspectingAsset = asset;
				}
			}
			if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
			{
				ImGui::SetDragDropPayload("Asset", &i.first, sizeof(Handle));
				ImGui::TextColored(ImVec4(0, 0, 1, 1), i.second->GetAssetFileName().c_str());
				ImGui::EndDragDropSource();
			}
		}
		ImGui::TreePop();
	}
}

std::shared_ptr<IAsset> ProjectManager::DuplicateAsset(const std::shared_ptr<IAsset>& target)
{
	const auto& projectManager = GetInstance();
	const auto folder = target->GetAssetRecord().lock()->GetFolder().lock();
	const auto path = target->GetProjectRelativePath();
	const auto prefix = (folder->GetProjectRelativePath() / path.stem()).string();
	const auto postfix = path.extension().string();
	const auto newPath = projectManager.GenerateNewProjectRelativePath(prefix, postfix);
	try {
		std::filesystem::copy(target->GetAbsolutePath(), projectManager.GetProjectPath().parent_path() / newPath, std::filesystem::copy_options::overwrite_existing);
	}
	catch (const std::exception& e)
	{
		EVOENGINE_ERROR(e.what());
	}
	auto newAsset = folder->GetOrCreateAsset(
		newPath.stem().string(), newPath.extension().string());
	return newAsset;
}

std::weak_ptr<Scene> ProjectManager::GetStartScene()
{
	auto& projectManager = ProjectManager::GetInstance();
	return projectManager.m_startScene;
}
void ProjectManager::SetStartScene(const std::shared_ptr<Scene>& scene)
{
	auto& projectManager = ProjectManager::GetInstance();
	projectManager.m_startScene = scene;
	SaveProject();
}
std::weak_ptr<Folder> ProjectManager::GetFolder(const Handle& handle)
{
	auto& projectManager = GetInstance();
	auto search = projectManager.m_folderRegistry.find(handle);
	if (search != projectManager.m_folderRegistry.end())
	{
		return search->second;
	}
	return {};
}
std::filesystem::path ProjectManager::GetPathRelativeToProject(const std::filesystem::path& absolutePath)
{
	auto& projectManager = GetInstance();
	if (!projectManager.m_projectFolder)
		return {};
	if (!absolutePath.is_absolute())
		return {};
	if (!IsInProjectFolder(absolutePath))
		return {};
	return std::filesystem::relative(absolutePath, projectManager.GetProjectPath().parent_path());
}
