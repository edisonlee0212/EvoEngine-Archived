#pragma once

#include "Mesh.hpp"
#include "IPrivateComponent.hpp"
#include "PrivateComponentRef.hpp"

namespace evo_engine {
	class Lod
	{
	public:
		int m_index = 0;
		std::vector<PrivateComponentRef> m_renderers;
		float m_lodOffset = 0.f;
		float m_transitionWidth = 0.f;
		bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer);
	};

	class LodGroup : public IPrivateComponent {
	public:
		std::vector<Lod> m_lods;
		bool m_overrideLodFactor = false;
		float m_lodFactor = 0.f;
		void Serialize(YAML::Emitter& out) const override;
		void Deserialize(const YAML::Node& in) override;
		bool OnInspect(const std::shared_ptr<EditorLayer>& editorLayer) override;
		void Relink(const std::unordered_map<Handle, Handle>& map, const std::shared_ptr<Scene>& scene) override;
	};
}
