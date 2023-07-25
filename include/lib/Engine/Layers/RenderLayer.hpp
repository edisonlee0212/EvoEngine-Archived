#pragma once
#include "Camera.hpp"
#include "ILayer.hpp"
#include "Material.hpp"
#include "Mesh.hpp"

namespace EvoEngine
{
	class RenderLayer : public ILayer {
		friend class Resources;
		friend class ShaderProgram;
		
		
		void OnCreate() override;
		void OnDestroy() override;
		void PreUpdate() override;
		void LateUpdate() override;
		void CreateRenderPasses();
		std::unordered_map<std::string, std::unique_ptr<RenderPass>> m_renderPasses;
	public:
		[[nodiscard]] const std::unique_ptr<RenderPass>& GetRenderPass(const std::string& name);

		bool m_allowAutoResize = true;

		EnvironmentInfoBlock m_environmentInfoBlock = {};
		RenderInfoBlock m_renderInfoBlock = {};
		MaterialInfoBlock m_materialInfoBlock = {};
		CameraInfoBlock m_cameraInfoBlock = {};

		std::shared_ptr<Mesh> m_mesh = {};
	};
}