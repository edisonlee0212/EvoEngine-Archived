#pragma once
#include "GraphicsResources.hpp"
#include "IAsset.hpp"

namespace evo_engine
{
	enum class ShaderType
	{
		Vertex,
		TessellationControl,
		TessellationEvaluation,
		Geometry,
		Task,
		Mesh,
		Fragment,
		Compute,
		Unknown
	};
	class Shader final : public IAsset
	{
		friend class GraphicsPipeline;
		std::unique_ptr<ShaderModule> m_shaderModule = {};
		std::string m_code = {};
		ShaderType m_shaderType = ShaderType::Unknown;
	public:
		[[nodiscard]] bool Compiled() const;
		void Set(ShaderType shaderType, const std::string& shaderCode);
		[[nodiscard]] const std::unique_ptr<ShaderModule>& GetShaderModule() const;
	};
}
