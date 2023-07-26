#include "Resources.hpp"
#include "RenderLayer.hpp"
#include "Utilities.hpp"
#include "ProjectManager.hpp"
#include "EditorLayer.hpp"
#include "Shader.hpp"
using namespace EvoEngine;

void Resources::LoadShaders()
{
#pragma region Shaders
#pragma region Shader Includes
	std::string add;

	add += "\n#define MAX_BONES_AMOUNT " + std::to_string(Graphics::GetMaxBoneAmount()) +
		"\n#define MAX_TEXTURES_AMOUNT " + std::to_string(Graphics::GetMaxMaterialAmount()) +
		"\n#define DIRECTIONAL_LIGHTS_AMOUNT " + std::to_string(Graphics::GetMaxDirectionalLightAmount()) +
		"\n#define POINT_LIGHTS_AMOUNT " + std::to_string(Graphics::GetMaxPointLightAmount()) +
		"\n#define SHADOW_CASCADE_AMOUNT " + std::to_string(Graphics::GetMaxShadowCascadeAmount()) +
		"\n#define MAX_KERNEL_AMOUNT " + std::to_string(Graphics::GetMaxKernelAmount()) +
		"\n#define SPOT_LIGHTS_AMOUNT " + std::to_string(Graphics::GetMaxSpotLightAmount()) + "\n";

	Graphics::GetInstance().m_standardShaderIncludes = std::make_unique<std::string>(
		add +
		FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Include/Uniform.glsl"));

#pragma endregion

#pragma region Standard Shader
	{
		auto vertShaderCode =
			std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Standard.vert");
		auto fragShaderCode = std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/StandardForward.frag");

		auto standardVert = CreateResource<Shader>("STANDARD_VERT");
		standardVert->Set(ShaderType::Vertex, vertShaderCode);
		auto standardFrag = CreateResource<Shader>("STANDARD_FORWARD_FRAG");
		standardFrag->Set(ShaderType::Fragment, fragShaderCode);

		

		/*
		vertShaderCode = std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Vertex/StandardSkinned.vert");
		auto standardSkinnedVert = CreateResource<Shader>("STANDARD_SKINNED_VERT");
		standardSkinnedVert->Set(ShaderType::Vertex, vertShaderCode);
		auto standardSkinnedProgram = CreateResource<GraphicsPipeline>("STANDARD_SKINNED_PROGRAM");
		standardSkinnedProgram->m_vertexShader = standardSkinnedVert;
		standardSkinnedProgram->m_fragmentShader =  standardFrag;


		vertShaderCode = std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Vertex/StandardInstanced.vert");

		auto standardInstancedVert = CreateResource<Shader>("STANDARD_INSTANCED_VERT");
		standardInstancedVert->Set(ShaderType::Vertex, vertShaderCode);
		auto standardInstancedProgram = CreateResource<GraphicsPipeline>("STANDARD_INSTANCED");
		standardInstancedProgram->m_vertexShader = standardInstancedVert;
		standardInstancedProgram->m_fragmentShader = standardFrag;

		vertShaderCode = std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/StandardInstancedColored.vert");

		auto standardInstancedColoredVert= CreateResource<Shader>("STANDARD_INSTANCED_COLORED_VERT");
		standardInstancedColoredVert->Set(ShaderType::Vertex, vertShaderCode);

		auto fragColoredShaderCode = std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/StandardForwardColored.frag");

		auto standardColoredFrag = CreateResource<Shader>("STANDARD_FORWARD_COLORED_FRAG");
		standardColoredFrag->Set(ShaderType::Fragment, fragColoredShaderCode);
		auto standardInstancedColoredProgram = CreateResource<GraphicsPipeline>("STANDARD_INSTANCED_COLORED");
		standardInstancedColoredProgram->m_vertexShader = standardInstancedColoredVert;
		standardInstancedColoredProgram->m_fragmentShader = standardColoredFrag;


		vertShaderCode = std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/StandardInstancedSkinned.vert");

		auto standardInstancedSkinnedVert = CreateResource<Shader>("STANDARD_INSTANCED_SKINNED_VERT");
		standardInstancedSkinnedVert->Set(ShaderType::Vertex, vertShaderCode);
		auto standardInstancedSkinned = CreateResource<GraphicsPipeline>("STANDARD_INSTANCED_SKINNED");
		standardInstancedSkinned->m_vertexShader = standardInstancedSkinnedVert;
		standardInstancedSkinned->m_fragmentShader = standardFrag;


		auto strandsVertCode =
			std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/StandardStrands.vert");

		auto tessContShaderCode =
			std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/TessellationControl/StandardStrands.tesc");

		auto tessEvalShaderCode =
			std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/TessellationEvaluation/StandardStrands.tese");

		auto geomShaderCode = std::string("#version 450 core\n") + std::string("#extension GL_EXT_geometry_shader4 : enable\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Geometry/StandardStrands.geom");

		auto standardStrandsVert = CreateResource<Shader>("STANDARD_STRANDS_VERT");
		standardStrandsVert->Set(ShaderType::Vertex, strandsVertCode);

		auto standardStrandsTessCont = CreateResource<Shader>("STANDARD_STRANDS_TESC");
		standardStrandsTessCont->Set(ShaderType::TessellationControl, tessContShaderCode);

		auto standardStrandsTessEval = CreateResource<Shader>("STANDARD_STRANDS_TESE");
		standardStrandsTessEval->Set(ShaderType::TessellationEvaluation, tessEvalShaderCode);

		auto standardStrandsGeometry = CreateResource<Shader>("STANDARD_STRANDS_GEOM");
		standardStrandsGeometry->Set(ShaderType::Geometry, geomShaderCode);

		auto standardStrandsProgram = CreateResource<GraphicsPipeline>("STANDARD_STRANDS_PROGRAM");
		standardStrandsProgram->m_vertexShader = standardStrandsVert;
		standardStrandsProgram->m_tessellationControlShader = standardStrandsTessCont;
		standardStrandsProgram->m_tessellationEvaluationShader = standardStrandsTessEval;
		standardStrandsProgram->m_geometryShader = standardStrandsGeometry;
		*/
	}


#pragma endregion
	auto texPassVertCode = std::string("#version 450 core\n") +
		FileUtils::LoadFileAsString(
			std::filesystem::path("./DefaultResources") / "Shaders/Vertex/TexturePassThrough.vert");
	auto texPassVert = CreateResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
	texPassVert->Set(ShaderType::Vertex, texPassVertCode);
#pragma region GBuffer
	{
		auto fragShaderCode =
			std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/StandardDeferredLighting.frag");
		auto standardDeferredLightingFrag = CreateResource<Shader>("STANDARD_DEFERRED_LIGHTING_FRAG");
		standardDeferredLightingFrag->Set(ShaderType::Fragment, fragShaderCode);
		
		
		fragShaderCode = std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/StandardDeferred.frag");

		auto standardDeferredPrepassFrag = CreateResource<Shader>("STANDARD_DEFERRED_FRAG");
		standardDeferredPrepassFrag->Set(ShaderType::Fragment, fragShaderCode);
		/*
		auto gBufferSkinnedPrepassProgram = CreateResource<GraphicsPipeline>("STANDARD_DEFERRED_SKINNED_PREPASS_PROGRAM");
		gBufferSkinnedPrepassProgram->m_vertexShader = GetResource("STANDARD_SKINNED_VERT");
		gBufferSkinnedPrepassProgram->m_fragmentShader = standardDeferredPrepassFrag;

		auto gBufferInstancedPrepassProgram = CreateResource<GraphicsPipeline>("STANDARD_DEFERRED_INSTANCED_PREPASS_PROGRAM");
		gBufferInstancedPrepassProgram->m_vertexShader = GetResource("STANDARD_INSTANCED_VERT");
		gBufferInstancedPrepassProgram->m_fragmentShader = standardDeferredPrepassFrag;

		auto tessContShaderCode =
			std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/TessellationControl/StandardStrands.tesc");

		auto tessEvalShaderCode =
			std::string("#version 450 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/TessellationEvaluation/StandardStrands.tese");

		auto geomShaderCode = std::string("#version 450 core\n") + std::string("#extension GL_EXT_geometry_shader4 : enable\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Geometry/StandardStrands.geom");


		auto standardTessCont = CreateResource<Shader>("StandardStrands.tesc");
		standardTessCont->Set(ShaderType::TessellationControl, tessContShaderCode);

		auto standardTessEval = CreateResource<Shader>("StandardStrands.tese");
		standardTessEval->Set(ShaderType::TessellationEvaluation, tessEvalShaderCode);

		auto standardGeometry = CreateResource<Shader>("StandardStrands.geom");
		standardGeometry->Set(ShaderType::Geometry, geomShaderCode);

		auto gBufferStrandsPrepassProgram = CreateResource<GraphicsPipeline>("STANDARD_DEFERRED_STRANDS_PREPASS_PROGRAM");
		gBufferStrandsPrepassProgram->m_vertexShader = GetResource("STANDARD_STRANDS_VERT");
		gBufferStrandsPrepassProgram->m_tessellationControlShader = standardTessCont;
		gBufferStrandsPrepassProgram->m_tessellationEvaluationShader = standardTessEval;
		gBufferStrandsPrepassProgram->m_geometryShader = standardGeometry;
		gBufferStrandsPrepassProgram->m_fragmentShader = standardDeferredPrepassFrag;
		*/
	}
#pragma endregion

	/*
	
	
#pragma region Post - Processing

	{
		auto bloomSeparatorProgram = CreateResource<GraphicsPipeline>("BLOOM_SEPARATOR_PROGRAM");
		auto fragShaderCode = std::string("#version 450 core\n") +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/BloomSeparator.frag");
		auto bloomSeparatorFrag = CreateResource<Shader>("BLOOM_SEPARATOR_FRAG");
		bloomSeparatorFrag->Set(ShaderType::Fragment, fragShaderCode);
		bloomSeparatorProgram->m_vertexShader = texPassVert;
		bloomSeparatorProgram->m_fragmentShader = bloomSeparatorFrag;

		auto bloomFilterProgram = CreateResource<GraphicsPipeline>("BLOOM_FILTER_PROGRAM");
		fragShaderCode = std::string("#version 450 core\n") +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/BlurFilter.frag");
		auto bloomFilterFrag = CreateResource<Shader>("BLOOM_FILTER_FRAG");
		bloomFilterFrag->Set(ShaderType::Fragment, fragShaderCode);
		bloomFilterProgram->m_vertexShader = texPassVert;
		bloomFilterProgram->m_fragmentShader = bloomFilterFrag;

		auto bloomCombineProgram = CreateResource<GraphicsPipeline>("BLOOM_COMBINE_PROGRAM");
		fragShaderCode = std::string("#version 450 core\n") +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/BloomCombine.frag");
		auto bloomCombineFrag = CreateResource<Shader>("BLOOM_COMBINE_FRAG");
		bloomCombineFrag->Set(ShaderType::Fragment, fragShaderCode);
		bloomCombineProgram->m_vertexShader = texPassVert;
		bloomCombineProgram->m_fragmentShader = bloomCombineFrag;

		fragShaderCode = std::string("#version 460 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/SSAOGeometry.frag");

		auto ssaoGeomProgram = CreateResource<GraphicsPipeline>("SSAO_GEOMETRY_PROGRAM");
		auto ssaoGeomFrag = CreateResource<Shader>("SSAO_GEOMETRY_FRAG");
		ssaoGeomFrag->Set(ShaderType::Fragment, fragShaderCode);
		ssaoGeomProgram->m_vertexShader = texPassVert;
		ssaoGeomProgram->m_fragmentShader = ssaoGeomFrag;

		fragShaderCode = std::string("#version 460 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/BlurFilter.frag");

		auto ssaoBlurProgram = CreateResource<GraphicsPipeline>("SSAO_BLUR_PROGRAM");
		auto ssaoBlurFrag = CreateResource<Shader>("SSAO_BLUR_FRAG");
		ssaoBlurFrag->Set(ShaderType::Fragment, fragShaderCode);
		ssaoBlurProgram->m_vertexShader = texPassVert;
		ssaoBlurProgram->m_fragmentShader = ssaoBlurFrag;

		fragShaderCode = std::string("#version 460 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/SSAOCombine.frag");

		auto ssaoCombineProgram = CreateResource<GraphicsPipeline>("SSAO_COMBINE_PROGRAM");
		auto ssaoCombineFrag = CreateResource<Shader>("SSAO_COMBINE_FRAG");
		ssaoCombineFrag->Set(ShaderType::Fragment, fragShaderCode);
		ssaoCombineProgram->m_vertexShader = texPassVert;
		ssaoCombineProgram->m_fragmentShader = ssaoCombineFrag;


		fragShaderCode = std::string("#version 460 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/SSRReflect.frag");

		auto ssrReflectProgram = CreateResource<GraphicsPipeline>("SSR_REFLECT_PROGRAM");
		auto ssrReflectFrag = CreateResource<Shader>("SSR_REFLECT_FRAG");
		ssrReflectFrag->Set(ShaderType::Fragment, fragShaderCode);
		ssrReflectProgram->m_vertexShader = texPassVert;
		ssrReflectProgram->m_fragmentShader = ssrReflectFrag;

		fragShaderCode = std::string("#version 460 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/BlurFilter.frag");

		auto ssrBlurProgram = CreateResource<GraphicsPipeline>("SSR_BLUR_PROGRAM");
		auto ssrBlurFrag = CreateResource<Shader>("SSR_BLUR_FRAG");
		ssrBlurFrag->Set(ShaderType::Fragment, fragShaderCode);
		ssrBlurProgram->m_vertexShader = texPassVert;
		ssrBlurProgram->m_fragmentShader = ssrBlurFrag;

		fragShaderCode = std::string("#version 460 core\n") + Graphics::GetStandardShaderIncludes() + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/SSRCombine.frag");
		auto ssrCombineProgram = CreateResource<GraphicsPipeline>("SSR_COMBINE_PROGRAM");
		auto ssrCombineFrag = CreateResource<Shader>("SSR_COMBINE_FRAG");
		ssrCombineFrag->Set(ShaderType::Fragment, fragShaderCode);
		ssrCombineProgram->m_vertexShader = texPassVert;
		ssrCombineProgram->m_fragmentShader = texPassVert; ssrCombineFrag;
	}

#pragma endregion
#pragma endregion
*/
}

void Resources::LoadPrimitives() const
{
	{
		const auto quad = CreateResource<Mesh>("PRIMITIVE_QUAD");
		quad->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/quad.uemesh");
	}
	{
		const auto sphere = CreateResource<Mesh>("PRIMITIVE_SPHERE");
		sphere->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/sphere.uemesh");
	}
	{
		const auto cube = CreateResource<Mesh>("PRIMITIVE_CUBE");
		cube->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/cube.uemesh");
	}
	{
		const auto cone = CreateResource<Mesh>("PRIMITIVE_CONE");
		cone->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/cone.uemesh");
	}
	{
		const auto cylinder = CreateResource<Mesh>("PRIMITIVE_CYLINDER");
		cylinder->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/cylinder.uemesh");
	}
	{
		const auto torus = CreateResource<Mesh>("PRIMITIVE_TORUS");
		torus->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/torus.uemesh");
	}
	{
		const auto monkey = CreateResource<Mesh>("PRIMITIVE_MONKEY");
		monkey->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/monkey.uemesh");
	}
	{
		const auto capsule = CreateResource<Mesh>("PRIMITIVE_CAPSULE");
		capsule->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/capsule.uemesh");
	}
}

void Resources::Initialize()
{
	auto& resources = GetInstance();
	resources.m_typedResources.clear();
	resources.m_namedResources.clear();
	resources.m_resources.clear();

	resources.m_currentMaxHandle = Handle(1);

	resources.LoadShaders();
	resources.LoadPrimitives();
}

Handle Resources::GenerateNewHandle()
{
	return m_currentMaxHandle.m_value++;
}

void Resources::OnInspect(const std::shared_ptr<EditorLayer>& editorLayer)
{
	auto& resources = GetInstance();
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("View"))
		{
			ImGui::Checkbox("Loaded Assets", &resources.m_showLoadedAssets);
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}
	auto& projectManager = ProjectManager::GetInstance();

	if (resources.m_showLoadedAssets)
	{
		ImGui::Begin("Loaded Assets");
		if (ImGui::BeginTabBar(
			"##LoadedAssets", ImGuiTabBarFlags_Reorderable | ImGuiTabBarFlags_NoCloseWithMiddleMouseButton))
		{
			if (ImGui::BeginTabItem("Resources"))
			{
				for (auto& collection : resources.m_typedResources)
				{
					if (ImGui::CollapsingHeader(collection.first.c_str()))
					{
						for (auto& i : collection.second)
						{
							ImGui::Button(resources.m_resourceNames[i.second->GetHandle()].c_str());
							editorLayer->DraggableAsset(i.second);
						}
					}
				}
				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Assets"))
			{
				if (ImGui::CollapsingHeader("Assets"))
				{
					for (auto& asset : projectManager.m_residentAsset)
					{
						if (asset.second->IsTemporary()) continue;
						ImGui::Button(asset.second->GetTitle().c_str());
						editorLayer->DraggableAsset(asset.second);
					}
				}
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}
		ImGui::End();
	}
}

bool Resources::IsResource(const Handle& handle)
{
	auto& resources = GetInstance();
	return resources.m_resources.find(handle) != resources.m_resources.end();
}

bool Resources::IsResource(const std::shared_ptr<IAsset>& target)
{
	auto& resources = GetInstance();
	return resources.m_resources.find(target->GetHandle()) != resources.m_resources.end();
}

bool Resources::IsResource(const AssetRef& target)
{
	auto& resources = GetInstance();
	return resources.m_resources.find(target.GetAssetHandle()) != resources.m_resources.end();
}

std::shared_ptr<IAsset> Resources::GetResource(const std::string& name)
{
	auto& resources = GetInstance();
	return resources.m_namedResources.at(name);
}

std::shared_ptr<IAsset> Resources::GetResource(const Handle& handle)
{
	auto& resources = GetInstance();
	return resources.m_resources.at(handle);
}
