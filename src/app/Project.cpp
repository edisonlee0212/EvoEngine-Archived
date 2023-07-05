#include "Application.hpp"
#include <shaderc/shaderc.hpp>
using namespace EvoEngine;
// Returns GLSL shader source text after preprocessing.
std::string preprocess_shader(const std::string& source_name,
    shaderc_shader_kind kind,
    const std::string& source) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    options.AddMacroDefinition("MY_DEFINE", "1");

    shaderc::PreprocessedSourceCompilationResult result =
        compiler.PreprocessGlsl(source, kind, source_name.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << result.GetErrorMessage();
        return "";
    }

    return { result.cbegin(), result.cend() };
}

// Compiles a shader to SPIR-V assembly. Returns the assembly text
// as a string.
std::string compile_file_to_assembly(const std::string& source_name,
    shaderc_shader_kind kind,
    const std::string& source,
    bool optimize = false) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    options.AddMacroDefinition("MY_DEFINE", "1");
    if (optimize) options.SetOptimizationLevel(shaderc_optimization_level_size);

    shaderc::AssemblyCompilationResult result = compiler.CompileGlslToSpvAssembly(
        source, kind, source_name.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << result.GetErrorMessage();
        return "";
    }

    return { result.cbegin(), result.cend() };
}

// Compiles a shader to a SPIR-V binary. Returns the binary as
// a vector of 32-bit words.
std::vector<uint32_t> compile_file(const std::string& source_name,
    shaderc_shader_kind kind,
    const std::string& source,
    bool optimize = false) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    options.AddMacroDefinition("MY_DEFINE", "1");
    if (optimize) options.SetOptimizationLevel(shaderc_optimization_level_size);

    shaderc::SpvCompilationResult module =
        compiler.CompileGlslToSpv(source, kind, source_name.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << module.GetErrorMessage();
        return std::vector<uint32_t>();
    }

    return { module.cbegin(), module.cend() };
}

int main() {
    const char kShaderSource[] =
        "#version 310 es\n"
        "void main() { int x = MY_DEFINE; }\n";

    {  // Preprocessing
        auto preprocessed = preprocess_shader(
            "shader_src", shaderc_glsl_vertex_shader, kShaderSource);
        std::cout << "Compiled a vertex shader resulting in preprocessed text:"
            << std::endl
            << preprocessed << std::endl;
    }

    {  // Compiling
        auto assembly = compile_file_to_assembly(
            "shader_src", shaderc_glsl_vertex_shader, kShaderSource);
        std::cout << "SPIR-V assembly:" << std::endl << assembly << std::endl;

        auto spirv =
            compile_file("shader_src", shaderc_glsl_vertex_shader, kShaderSource);
        std::cout << "Compiled to a binary module with " << spirv.size()
            << " words." << std::endl;
    }

    {  // Compiling with optimizing
        auto assembly =
            compile_file_to_assembly("shader_src", shaderc_glsl_vertex_shader,
                kShaderSource, /* optimize = */ true);
        std::cout << "Optimized SPIR-V assembly:" << std::endl
            << assembly << std::endl;

        auto spirv = compile_file("shader_src", shaderc_glsl_vertex_shader,
            kShaderSource, /* optimize = */ true);
        std::cout << "Compiled to an optimized binary module with " << spirv.size()
            << " words." << std::endl;
    }

    Application::Initialize({});
    Application::Terminate();
    return 0;
}