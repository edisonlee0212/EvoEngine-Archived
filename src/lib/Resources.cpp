#include "Resources.hpp"

#include "Cubemap.hpp"
#include "RenderLayer.hpp"
#include "Utilities.hpp"
#include "ProjectManager.hpp"
#include "EditorLayer.hpp"
#include "MeshStorage.hpp"
#include "Shader.hpp"
using namespace EvoEngine;

void Resources::LoadShaders()
{
#pragma region Shaders
#pragma region Shader Includes
	std::string add;

	add += "\n#define MAX_DIRECTIONAL_LIGHT_SIZE " + std::to_string(Graphics::Constants::MAX_DIRECTIONAL_LIGHT_SIZE)
		 + "\n#define MAX_KERNEL_AMOUNT " + std::to_string(Graphics::Constants::MAX_KERNEL_AMOUNT)
		+ "\n#define MESHLET_MAX_VERTICES_SIZE " + std::to_string(Graphics::Constants::MESHLET_MAX_VERTICES_SIZE)
		+ "\n#define MESHLET_MAX_TRIANGLES_SIZE " + std::to_string(Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE)
		+ "\n#define MESHLET_MAX_INDICES_SIZE " + std::to_string(Graphics::Constants::MESHLET_MAX_TRIANGLES_SIZE * 3)
	+ "\n";

	Graphics::GetInstance().m_shaderBasic = add + FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Include/Basic.glsl");
	Graphics::GetInstance().m_shaderLighting = FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Include/Lighting.glsl");

#pragma endregion

#pragma region Standard Shader
	{
		auto vertShaderCode =
			std::string("#version 450 core\n") + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Standard/Standard.vert");

		auto standardVert = CreateResource<Shader>("STANDARD_VERT");
		standardVert->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 450 core\n") + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Standard/StandardSkinned.vert");

		standardVert = CreateResource<Shader>("STANDARD_SKINNED_VERT");
		standardVert->Set(ShaderType::Vertex, vertShaderCode);
	}

#pragma endregion
	auto texPassVertCode = std::string("#version 450 core\n") +
		FileUtils::LoadFileAsString(
			std::filesystem::path("./DefaultResources") / "Shaders/Vertex/TexturePassThrough.vert");
	auto texPassVert = CreateResource<Shader>("TEXTURE_PASS_THROUGH_VERT");
	texPassVert->Set(ShaderType::Vertex, texPassVertCode);

	auto texPassFragCode = std::string("#version 450 core\n") +
		FileUtils::LoadFileAsString(
			std::filesystem::path("./DefaultResources") / "Shaders/Fragment/TexturePassThrough.frag");
	auto texPassFrag = CreateResource<Shader>("TEXTURE_PASS_THROUGH_FRAG");
	texPassFrag->Set(ShaderType::Fragment, texPassFragCode);

#pragma region GBuffer
	{
		auto fragShaderCode =
			std::string("#version 450 core\n") + Graphics::GetInstance().m_shaderBasic + "\n" + Graphics::GetInstance().m_shaderLighting + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Standard/StandardDeferredLighting.frag");
		auto standardDeferredLightingFrag = CreateResource<Shader>("STANDARD_DEFERRED_LIGHTING_FRAG");
		standardDeferredLightingFrag->Set(ShaderType::Fragment, fragShaderCode);


		fragShaderCode =
			std::string("#version 450 core\n") + Graphics::GetInstance().m_shaderBasic + "\n" + Graphics::GetInstance().m_shaderLighting + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Standard/StandardDeferredLightingSceneCamera.frag");
		standardDeferredLightingFrag = CreateResource<Shader>("STANDARD_DEFERRED_LIGHTING_SCENE_CAMERA_FRAG");
		standardDeferredLightingFrag->Set(ShaderType::Fragment, fragShaderCode);
		
		fragShaderCode = std::string("#version 450 core\n") + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Standard/StandardDeferred.frag");

		auto standardDeferredPrepassFrag = CreateResource<Shader>("STANDARD_DEFERRED_FRAG");
		standardDeferredPrepassFrag->Set(ShaderType::Fragment, fragShaderCode);
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

#pragma region Shadow Maps
	{
		auto vertShaderCode =
			std::string("#version 450 core\n") + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/DirectionalLightShadowMap.vert");

		auto vertShader = CreateResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 450 core\n") + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/DirectionalLightShadowMapSkinned.vert");

		vertShader = CreateResource<Shader>("DIRECTIONAL_LIGHT_SHADOW_MAP_SKINNED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 450 core\n") + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/PointLightShadowMap.vert");

		vertShader = CreateResource<Shader>("POINT_LIGHT_SHADOW_MAP_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 450 core\n") + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/PointLightShadowMapSkinned.vert");

		vertShader = CreateResource<Shader>("POINT_LIGHT_SHADOW_MAP_SKINNED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 450 core\n") + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/SpotLightShadowMap.vert");

		vertShader = CreateResource<Shader>("SPOT_LIGHT_SHADOW_MAP_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		vertShaderCode =
			std::string("#version 450 core\n") + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/SpotLightShadowMapSkinned.vert");

		vertShader = CreateResource<Shader>("SPOT_LIGHT_SHADOW_MAP_SKINNED_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		auto fragShaderCode =
			std::string("#version 450 core\n") + Graphics::GetInstance().m_shaderBasic + "\n" +
			FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Empty.frag");
		auto fragShader = CreateResource<Shader>("EMPTY_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);
	}
#pragma endregion
#pragma region Environmental
	{
		auto vertShaderCode =
			std::string("#version 450 core\n") + 
			FileUtils::LoadFileAsString(std::filesystem::path("./DefaultResources") / "Shaders/Vertex/Lighting/EquirectangularMapToCubemap.vert");
		auto vertShader = CreateResource<Shader>("EQUIRECTANGULAR_MAP_TO_CUBEMAP_VERT");
		vertShader->Set(ShaderType::Vertex, vertShaderCode);

		auto fragShaderCode =
			std::string("#version 450 core\n") + FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Lighting/EnvironmentalMapBrdf.frag");
		auto fragShader = CreateResource<Shader>("ENVIRONMENTAL_MAP_BRDF_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);

		fragShaderCode =
			std::string("#version 450 core\n") + FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Lighting/EquirectangularMapToCubemap.frag");
		fragShader = CreateResource<Shader>("EQUIRECTANGULAR_MAP_TO_CUBEMAP_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);

		fragShaderCode =
			std::string("#version 450 core\n") + FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Lighting/EnvironmentalMapIrradianceConvolution.frag");
		fragShader = CreateResource<Shader>("IRRADIANCE_CONSTRUCT_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);

		fragShaderCode =
			std::string("#version 450 core\n") + FileUtils::LoadFileAsString(
				std::filesystem::path("./DefaultResources") / "Shaders/Fragment/Lighting/EnvironmentalMapPrefilter.frag");
		fragShader = CreateResource<Shader>("PREFILTER_CONSTRUCT_FRAG");
		fragShader->Set(ShaderType::Fragment, fragShaderCode);
	}
#pragma endregion
}

void Resources::LoadPrimitives() const
{
	{
		VertexAttributes attributes {};
		attributes.m_texCoord = true;
		Vertex vertex{};
		std::vector<Vertex> vertices;

		vertex.m_position = { -1, 1, 0 };
		vertex.m_texCoord = { 0, 1 };
		vertices.emplace_back(vertex);

		vertex.m_position = { 1, 1, 0 };
		vertex.m_texCoord = { 1, 1 };
		vertices.emplace_back(vertex);

		vertex.m_position = { -1, -1, 0 };
		vertex.m_texCoord = { 0, 0 };
		vertices.emplace_back(vertex);

		vertex.m_position = { 1, -1, 0 };
		vertex.m_texCoord = { 1, 0 };
		vertices.emplace_back(vertex);

		std::vector<glm::uvec3> triangles = { {0,2,3}, {0,3,1} };
		const auto texPassThrough = CreateResource<Mesh>("PRIMITIVE_TEX_PASS_THROUGH");
		texPassThrough->SetVertices(attributes, vertices, triangles);
	}
	{
		VertexAttributes attributes{};
		Vertex vertex{};
		std::vector<Vertex> vertices;

		vertex.m_position = { -1, -1, -1 };
		vertices.emplace_back(vertex);//0: 

		vertex.m_position = { 1, 1, -1 };
		vertices.emplace_back(vertex);//1: 

		vertex.m_position = { 1, -1, -1};
		vertices.emplace_back(vertex);//2: 

		vertex.m_position = { -1, 1, -1};
		vertices.emplace_back(vertex);//3: 


		vertex.m_position = { -1, -1, 1 };
		vertices.emplace_back(vertex);//4: 

		vertex.m_position = {1, -1,1 };
		vertices.emplace_back(vertex);//5: 

		vertex.m_position = { 1, 1, 1 };
		vertices.emplace_back(vertex);//6: 

		vertex.m_position = { -1, 1, 1 };
		vertices.emplace_back(vertex);//7: 


		std::vector<glm::uvec3> triangles = {
			{0,1,2}, {1, 0, 3},//OK
			{4,5,6}, {6, 7, 4},//OK
			{7,3,0}, {0, 4, 7},
			{6,2,1}, {2, 6, 5},
			{0,2,5}, {5, 4, 0},
			{3,6,1}, {6, 3, 7},
		};
		const auto renderingCube = CreateResource<Mesh>("PRIMITIVE_RENDERING_CUBE");
		renderingCube->SetVertices(attributes, vertices, triangles);
	}
	{
		const auto quad = CreateResource<Mesh>("PRIMITIVE_QUAD");
		quad->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/quad.evemesh");
	}
	{
		const auto sphere = CreateResource<Mesh>("PRIMITIVE_SPHERE");
		sphere->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/sphere.evemesh");
	}
	{
		const auto cube = CreateResource<Mesh>("PRIMITIVE_CUBE");
		cube->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/cube.evemesh");
	}
	{
		const auto cone = CreateResource<Mesh>("PRIMITIVE_CONE");
		cone->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/cone.evemesh");
	}
	{
		const auto cylinder = CreateResource<Mesh>("PRIMITIVE_CYLINDER");
		cylinder->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/cylinder.evemesh");
	}
	{
		const auto torus = CreateResource<Mesh>("PRIMITIVE_TORUS");
		torus->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/torus.evemesh");
	}
	{
		const auto monkey = CreateResource<Mesh>("PRIMITIVE_MONKEY");
		monkey->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/monkey.evemesh");
	}
	{
		const auto capsule = CreateResource<Mesh>("PRIMITIVE_CAPSULE");
		capsule->LoadInternal(std::filesystem::path("./DefaultResources") / "Primitives/capsule.evemesh");
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
	

	const auto missingTexture = CreateResource<Texture2D>("TEXTURE_MISSING");
	missingTexture->LoadInternal(std::filesystem::path("./DefaultResources") / "Textures/texture-missing.png");
}

void Resources::InitializeEnvironmentalMap()
{
	auto& resources = GetInstance();
	resources.LoadPrimitives();
	MeshStorage::GetInstance().UploadData();
	const auto defaultEnvironmentalMapTexture = CreateResource<Texture2D>("DEFAULT_ENVIRONMENTAL_MAP_TEXTURE");
	defaultEnvironmentalMapTexture->LoadInternal(std::filesystem::path("./DefaultResources") / "Textures/Cubemaps/GrandCanyon/GCanyon_C_YumaPoint_3k.hdr");
	
	const auto defaultSkyboxTexture = CreateResource<Texture2D>("DEFAULT_SKYBOX_TEXTURE");
	defaultSkyboxTexture->LoadInternal(std::filesystem::path("./DefaultResources") / "Textures/Cubemaps/GrandCanyon/GCanyon_C_YumaPoint_Env.hdr");
	const auto defaultSkybox = CreateResource<Cubemap>("DEFAULT_SKYBOX");
	defaultSkybox->Initialize(256);
	defaultSkybox->ConvertFromEquirectangularTexture(defaultSkyboxTexture);

	const auto defaultEnvironmentalMap = CreateResource<EnvironmentalMap>("DEFAULT_ENVIRONMENTAL_MAP");
	defaultEnvironmentalMap->ConstructFromTexture2D(defaultEnvironmentalMapTexture);
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