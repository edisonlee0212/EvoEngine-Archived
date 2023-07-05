#include "Utilities.hpp"

#include "Console.hpp"
#include "Graphics.hpp"
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

void FileUtils::OpenFile(
    const std::string& dialogTitle,
    const std::string& fileType,
    const std::vector<std::string>& extensions,
    const std::function<void(const std::filesystem::path& path)>& func,
    bool projectDirCheck)
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    if (ImGui::Button(dialogTitle.c_str()))
    {
        OPENFILENAMEA ofn;
        CHAR szFile[260] = { 0 };
        ZeroMemory(&ofn, sizeof(OPENFILENAME));
        ofn.lStructSize = sizeof(OPENFILENAME);
        ofn.hwndOwner = glfwGetWin32Window(Graphics::GetGlfwWindow());
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
        dialogTitle, imgui_addons::ImGuiFileBrowser::DialogMode::OPEN, ImVec2(700, 310), filters))
    {
        func(file_dialog.selected_path);
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
