#pragma once
#include "AssetRef.hpp"
#include "IAsset.hpp"
#include "MaterialProperties.hpp"
namespace EvoEngine
{
	class Material : public IAsset
	{
		AssetRef m_albedoTexture;
		AssetRef m_normalTexture;
		AssetRef m_metallicTexture;
		AssetRef m_roughnessTexture;
		AssetRef m_aoTexture;

		MaterialProperties m_materialProperties;
	public:
	};
}
