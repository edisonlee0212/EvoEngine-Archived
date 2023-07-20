#include "Utilities.hpp"
#include "shlobj.h"
#include "Application.hpp"
#include "Console.hpp"
#include "ProjectManager.hpp"
#include "WindowLayer.hpp"
#include "EditorLayer.hpp"
using namespace EvoEngine;
std::string FileUtils::LoadFileAsString(const std::filesystem::path& path)
{
    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try
    {
        // open files
        file.open(path);
        std::stringstream stream;
        // read file's buffer contents into streams
        stream << file.rdbuf();
        // close file handlers
        file.close();
        // convert stream into string
        return stream.str();
    }
    catch (std::ifstream::failure e)
    {
        EVOENGINE_ERROR("Load file failed!")
            throw;
    }
}

void FileUtils::OpenFolder(const std::string& dialogTitle,
	const std::function<void(const std::filesystem::path& path)>& func, bool projectDirCheck)
{
    if (ImGui::Button(dialogTitle.c_str()))
    {
        TCHAR path[MAX_PATH];
        BROWSEINFO bi = { 0 };
        bi.lpszTitle = dialogTitle.c_str();
        bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
        //bi.lpfn       = BrowseCallbackProc;
        //bi.lParam     = (LPARAM) path_param;
        LPITEMIDLIST pidl = SHBrowseForFolder(&bi);
        if (pidl != nullptr)
        {
            // get the name of the folder and put it in path
            SHGetPathFromIDList(pidl, path);
            // free memory used
            IMalloc* imalloc = nullptr;
            if (SUCCEEDED(SHGetMalloc(&imalloc)))
            {
                imalloc->Free(pidl);
                imalloc->Release();
            }
            std::string retVal = path;
            const std::string search = "\\";
            size_t pos = retVal.find(search);
            // Repeat till end is reached
            while (pos != std::string::npos)
            {
                // Replace this occurrence of Sub String
                retVal.replace(pos, 1, "/");
                // Get the next occurrence from the current position
                pos = retVal.find(search, pos + 1);
            }
            std::filesystem::path path = retVal;
            if (!projectDirCheck || ProjectManager::IsInProjectFolder(path))
                func(path);
        }
    }
}

void FileUtils::OpenFile(
    const std::string& dialogTitle,
    const std::string& fileType,
    const std::vector<std::string>& extensions,
    const std::function<void(const std::filesystem::path& path)>& func,
    bool projectDirCheck)
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    auto windowLayer = Application::GetLayer<WindowLayer>();
    if (windowLayer && ImGui::Button(dialogTitle.c_str()))
    {
        OPENFILENAMEA ofn;
        CHAR szFile[260] = { 0 };
        ZeroMemory(&ofn, sizeof(OPENFILENAME));
        ofn.lStructSize = sizeof(OPENFILENAME);
        ofn.hwndOwner = glfwGetWin32Window(windowLayer->GetGlfwWindow());
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile);
        std::string filters = fileType + " (";
        for (int i = 0; i < extensions.size(); i++)
        {
            filters += "*" + extensions[i];
            if (i < extensions.size() - 1)
                filters += ", ";
        }
        filters += ") ";
        std::string filters2;
        for (int i = 0; i < extensions.size(); i++)
        {
            filters2 += "*" + extensions[i];
            if (i < extensions.size() - 1)
                filters2 += ";";
        }
        char actualFilter[256];
        char title[256];
        strcpy(title, dialogTitle.c_str());
        int index = 0;
        for (auto& i : filters)
        {
            actualFilter[index] = i;
            index++;
        }
        actualFilter[index] = 0;
        index++;
        for (auto& i : filters2)
        {
            actualFilter[index] = i;
            index++;
        }
        actualFilter[index] = 0;
        index++;
        actualFilter[index] = 0;
        index++;
        ofn.lpstrFilter = actualFilter;
        ofn.nFilterIndex = 1;
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
        ofn.lpstrTitle = title;
        if (GetOpenFileNameA(&ofn) == TRUE)
        {
            std::string retVal = ofn.lpstrFile;
            const std::string search = "\\";
            size_t pos = retVal.find(search);
            // Repeat till end is reached
            while (pos != std::string::npos)
            {
                // Replace this occurrence of Sub String
                retVal.replace(pos, 1, "/");
                // Get the next occurrence from the current position
                pos = retVal.find(search, pos + 1);
            }
            std::filesystem::path path = retVal;
            //if (!projectDirCheck || ProjectManager::IsInProjectFolder(path))
                func(path);
        }
    }
#else
    if (windowLayer && ImGui::Button(dialogTitle.c_str()))
        ImGui::OpenPopup(dialogTitle.c_str());
    static imgui_addons::ImGuiFileBrowser file_dialog;
    std::string filters;
    for (int i = 0; i < extensions.size(); i++)
    {
        filters += extensions[i];
        if (i < extensions.size() - 1)
            filters += ",";
    }
    if (file_dialog.showFileDialog(
        dialogTitle, imgui_addons::ImGuiFileBrowser::DialogMode::OPEN, ImVec2(700, 310), filters))
    {
        func(file_dialog.selected_path);
    }
#endif
}

void FileUtils::SaveFile(const std::string& dialogTitle, const std::string& fileType,
	const std::vector<std::string>& extensions, const std::function<void(const std::filesystem::path& path)>& func,
	bool projectDirCheck)
{
    const auto windowLayer = Application::GetLayer<WindowLayer>();
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    if (ImGui::Button(dialogTitle.c_str()))
    {
        OPENFILENAMEA ofn;
        CHAR szFile[260] = { 0 };
        ZeroMemory(&ofn, sizeof(OPENFILENAME));
        ofn.lStructSize = sizeof(OPENFILENAME);
        ofn.hwndOwner = glfwGetWin32Window(windowLayer->GetGlfwWindow());
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile);
        std::string filters = fileType + " (";
        for (int i = 0; i < extensions.size(); i++)
        {
            filters += "*" + extensions[i];
            if (i < extensions.size() - 1)
                filters += ", ";
        }
        filters += ") ";
        std::string filters2;
        for (int i = 0; i < extensions.size(); i++)
        {
            filters2 += "*" + extensions[i];
            if (i < extensions.size() - 1)
                filters2 += ";";
        }
        char actualFilter[256];
        char title[256];
        strcpy(title, dialogTitle.c_str());
        int index = 0;
        for (auto& i : filters)
        {
            actualFilter[index] = i;
            index++;
        }
        actualFilter[index] = 0;
        index++;
        for (auto& i : filters2)
        {
            actualFilter[index] = i;
            index++;
        }
        actualFilter[index] = 0;
        index++;
        actualFilter[index] = 0;
        index++;
        ofn.lpstrFilter = actualFilter;
        ofn.nFilterIndex = 1;
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
        ofn.lpstrTitle = title;
        // Sets the default extension by extracting it from the filter
        ofn.lpstrDefExt = strchr(actualFilter, '\0') + 1;

        if (GetSaveFileNameA(&ofn) == TRUE)
        {
            std::string retVal = ofn.lpstrFile;
            const std::string search = "\\";
            size_t pos = retVal.find(search);
            // Repeat till end is reached
            while (pos != std::string::npos)
            {
                // Replace this occurrence of Sub String
                retVal.replace(pos, 1, "/");
                // Get the next occurrence from the current position
                pos = retVal.find(search, pos + 1);
            }
            std::filesystem::path path = retVal;
            if (!projectDirCheck || ProjectManager::IsInProjectFolder(path))
                func(path);
        }
    }
#else
    if (ImGui::Button(dialogTitle.c_str()))
        ImGui::OpenPopup(dialogTitle.c_str());
    static imgui_addons::ImGuiFileBrowser file_dialog;
    std::string filters;
    for (int i = 0; i < extensions.size(); i++)
    {
        filters += extensions[i];
        if (i < extensions.size() - 1)
            filters += ",";
    }
    if (file_dialog.showFileDialog(
        dialogTitle, imgui_addons::ImGuiFileBrowser::DialogMode::SAVE, ImVec2(700, 310), filters))
    {
        std::filesystem::path path = file_dialog.selected_path;
        func(path);
    }
#endif
}

std::string ShaderUtils::PreprocessShader(const std::string& sourceName, shaderc_shader_kind kind,
                                          const std::string& source)
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    options.AddMacroDefinition("MY_DEFINE", "1");

    shaderc::PreprocessedSourceCompilationResult result =
        compiler.PreprocessGlsl(source, kind, sourceName.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << result.GetErrorMessage();
        return "";
    }

    return { result.cbegin(), result.cend() };
}

std::string ShaderUtils::CompileFileToAssembly(const std::string& sourceName, shaderc_shader_kind kind,
	const std::string& source, bool optimize)
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    options.AddMacroDefinition("MY_DEFINE", "1");
    if (optimize) options.SetOptimizationLevel(shaderc_optimization_level_size);

    shaderc::AssemblyCompilationResult result = compiler.CompileGlslToSpvAssembly(
        source, kind, sourceName.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << result.GetErrorMessage();
        return "";
    }

    return { result.cbegin(), result.cend() };
}

std::vector<uint32_t> ShaderUtils::CompileFile(const std::string& sourceName, shaderc_shader_kind kind,
	const std::string& source, bool optimize)
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    options.AddMacroDefinition("MY_DEFINE", "1");
    if (optimize) options.SetOptimizationLevel(shaderc_optimization_level_size);

    shaderc::SpvCompilationResult module =
        compiler.CompileGlslToSpv(source, kind, sourceName.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << module.GetErrorMessage();
        return std::vector<uint32_t>();
    }

    return { module.cbegin(), module.cend() };
}
