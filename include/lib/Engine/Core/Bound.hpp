#pragma once
namespace EvoEngine
{
	struct Bound
	{
		glm::vec3 m_min = glm::vec3(FLT_MIN);
		glm::vec3 m_max = glm::vec3(FLT_MAX);
		[[nodiscard]] glm::vec3 Size() const;
		[[nodiscard]] glm::vec3 Center() const;
		[[nodiscard]] bool InBound(const glm::vec3& position) const;
		void ApplyTransform(const glm::mat4& transform);
		void PopulateCorners(std::vector<glm::vec3>& corners) const;
	};
}