#pragma once
#include "ISingleton.hpp"
#include "Material.hpp"
#include "Mesh.hpp"

namespace EvoEngine
{
	struct GizmoSettings {
		DrawSettings m_drawSettings;
		enum class ColorMode {
			Default,
			VertexColor,
			NormalColor
		} m_colorMode = ColorMode::Default;
		bool m_depthTest = false;
		bool m_depthWrite = false;
		void ApplySettings(GraphicsPipelineStates& globalPipelineState) const;
	};
	struct GizmosPushConstant
	{
		glm::mat4 m_model;
		glm::vec4 m_color;
		float m_size;
		uint32_t m_cameraIndex;
	};
	class Gizmos : public ISingleton<Gizmos>
	{
	public:
		static void DrawGizmoMeshInstancedColored(
			const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<ParticleInfoList>& instancedData,
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		static void DrawGizmoMesh(
			const std::shared_ptr<Mesh>& mesh,
			const glm::vec4& color = glm::vec4(1.0f),
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		static void DrawGizmoCubes(
			const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<ParticleInfoList>& instancedData,
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		static void DrawGizmoCube(
			const glm::vec4& color = glm::vec4(1.0f),
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		static void DrawGizmoSpheres(
			const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<ParticleInfoList>& instancedData,
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		static void DrawGizmoSphere(
			const glm::vec4& color = glm::vec4(1.0f),
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		static void DrawGizmoCylinders(
			const std::shared_ptr<Mesh>& mesh, const std::shared_ptr<ParticleInfoList>& instancedData,
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});

		static void DrawGizmoCylinder(
			const glm::vec4& color = glm::vec4(1.0f),
			const glm::mat4& model = glm::mat4(1.0f),
			const float& size = 1.0f, const GizmoSettings& gizmoSettings = {});
	};
}
