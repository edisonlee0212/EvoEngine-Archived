#pragma once
#include "AssetRef.hpp"
#include "GraphicsResources.hpp"
#include "IAsset.hpp"

namespace EvoEngine
{
	enum class ShaderType
	{
		Vertex,
		TessellationControl,
		TessellationEvaluation,
		Geometry,
		Fragment,
		Compute,
		Unknown
	};
	class Shader : public IAsset
	{
		std::unique_ptr<ShaderEXT> m_shaderEXT;
		std::string m_code = {};
		ShaderType m_shaderType = ShaderType::Unknown;
	public:
		void Set(ShaderType shaderType, const std::string& shaderCode);
		[[nodiscard]] const std::unique_ptr<ShaderEXT>& GetShaderEXT() const;
	};

	class ShaderProgram : public IAsset
	{
	public:
		AssetRef m_vertexShader;
		AssetRef m_tessellationControlShader;
		AssetRef m_tessellationEvaluationShader;
		AssetRef m_geometryShader;
		AssetRef m_fragmentShader;
		AssetRef m_computeShader;

	};
}
